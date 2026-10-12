#!/usr/bin/env python3
"""
Konvertiert Real-ESRGAN-Gewichte (.pth) in ONNX-Modelle für Bildwerk – ganz ohne PyTorch.

Unterstützte Architekturen (werden automatisch erkannt):
  * SRVGGNetCompact  – realesr-general-x4v3, realesr-general-wdn-x4v3, realesr-animevideov3
  * RRDBNet          – RealESRGAN_x4plus, RealESRGAN_x2plus, RealESRGAN_x4plus_anime_6B

Benötigt:  pip install onnx numpy
Aufruf:    python pth_to_onnx.py RealESRGAN_x4plus.pth models/RealESRGAN_x4plus.onnx

Das erzeugte Modell erwartet einen Tensor "input" [1,3,H,W] (RGB, float32, 0..1)
und liefert "output" [1,3,H*s,W*s]. Der Skalierungsfaktor wird als Metadatum "scale" abgelegt.
"""
import pickle
import re
import sys
import zipfile
from collections import OrderedDict

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

# ---------------------------------------------------------------- .pth laden (ohne torch)

_DTYPES = {
    "FloatStorage": np.float32,
    "HalfStorage": np.float16,
    "DoubleStorage": np.float64,
    "LongStorage": np.int64,
    "IntStorage": np.int32,
}


class _StorageType:
    def __init__(self, name):
        self.dtype = _DTYPES[name]


def _rebuild_tensor_v2(storage, offset, size, stride, *_args):
    arr = storage
    if len(size) == 0:
        return arr[offset].copy()
    itemsize = arr.dtype.itemsize
    view = np.lib.stride_tricks.as_strided(
        arr[offset:], shape=tuple(size), strides=tuple(s * itemsize for s in stride)
    )
    return np.ascontiguousarray(view)


def load_pth(path):
    zf = zipfile.ZipFile(path)
    pkl_name = next(n for n in zf.namelist() if n.endswith("data.pkl"))
    prefix = pkl_name[: -len("data.pkl")]

    class SafeUnpickler(pickle.Unpickler):
        # Nur die wenigen Klassen erlauben, die ein state_dict braucht.
        def find_class(self, module, name):
            if module == "collections" and name == "OrderedDict":
                return OrderedDict
            if module == "torch._utils" and name == "_rebuild_tensor_v2":
                return _rebuild_tensor_v2
            if module == "torch" and name in _DTYPES:
                return _StorageType(name)
            raise pickle.UnpicklingError(f"Nicht erlaubt: {module}.{name}")

        def persistent_load(self, pid):
            _tag, stype, key, _loc, _numel = pid
            raw = zf.read(f"{prefix}data/{key}")
            return np.frombuffer(raw, dtype=stype.dtype)

    obj = SafeUnpickler(zf.open(pkl_name)).load()
    for k in ("params_ema", "params", "state_dict"):
        if isinstance(obj, dict) and k in obj:
            obj = obj[k]
            break
    return {k: np.asarray(v, dtype=np.float32) for k, v in obj.items()}


# ---------------------------------------------------------------- ONNX-Graph-Baukasten


class Graph:
    def __init__(self, sd):
        self.sd = sd
        self.nodes, self.inits = [], []
        self.n = 0

    def name(self, base):
        self.n += 1
        return f"{base}_{self.n}"

    def const(self, arr, base="c"):
        nm = self.name(base)
        self.inits.append(numpy_helper.from_array(np.asarray(arr), nm))
        return nm

    def op(self, op_type, inputs, **attrs):
        out = self.name(op_type.lower())
        self.nodes.append(helper.make_node(op_type, inputs, [out], **attrs))
        return out

    def conv(self, x, key, weight=None):
        w = self.sd[key + ".weight"] if weight is None else weight
        b = self.sd[key + ".bias"]
        k = w.shape[-1]
        return self.op(
            "Conv",
            [x, self.const(w, key + "_w"), self.const(b, key + "_b")],
            kernel_shape=[k, k],
            pads=[k // 2] * 4,
        )

    def lrelu(self, x):
        return self.op("LeakyRelu", [x], alpha=0.2)

    def prelu(self, x, key):
        slope = self.sd[key + ".weight"].reshape(-1, 1, 1)
        return self.op("PRelu", [x, self.const(slope, key + "_slope")])

    def add(self, a, b):
        return self.op("Add", [a, b])

    def scale(self, x, s):
        return self.op("Mul", [x, self.const(np.array(s, dtype=np.float32))])

    def up2(self, x, factor=2):
        scales = self.const(np.array([1, 1, factor, factor], dtype=np.float32), "scales")
        return self.op(
            "Resize",
            [x, "", scales],
            mode="nearest",
            coordinate_transformation_mode="asymmetric",
            nearest_mode="floor",
        )


def build_srvgg(sd):
    convs = sorted({int(m.group(1)) for k in sd if (m := re.match(r"body\.(\d+)\.bias", k))})
    last = convs[-1]
    out_ch = sd[f"body.{last}.weight"].shape[0]
    scale = int(round((out_ch / 3) ** 0.5))
    g = Graph(sd)
    x = "input"
    h = x
    for idx in convs:
        h = g.conv(h, f"body.{idx}")
        if idx != last:
            h = g.prelu(h, f"body.{idx + 1}")
    h = g.op("DepthToSpace", [h], blocksize=scale, mode="CRD")  # = torch.pixel_shuffle
    base = g.up2(x, scale)
    return g, g.add(h, base), scale


def build_rrdb(sd):
    n_blocks = len({k.split(".")[1] for k in sd if k.startswith("body.")})
    in_ch = sd["conv_first.weight"].shape[1]
    g = Graph(sd)
    x = "input"
    w_first = sd["conv_first.weight"]
    if in_ch == 12:  # x2-Modell: pixel_unshuffle(2) vorneweg
        scale = 2
        r, c = 2, 3
        # torch-Kanalreihenfolge c*r*r + i*r + j  ->  ONNX SpaceToDepth (i*r + j)*C + c
        w_new = np.empty_like(w_first)
        for ch in range(c):
            for i in range(r):
                for j in range(r):
                    w_new[:, (i * r + j) * c + ch] = w_first[:, ch * r * r + i * r + j]
        x = g.op("SpaceToDepth", [x], blocksize=2)
        w_first = w_new
    elif in_ch == 48:
        raise SystemExit("x1-RRDB-Modelle werden nicht unterstützt")
    else:
        scale = 4
    feat = g.conv(x, "conv_first", weight=w_first)

    h = feat
    for b in range(n_blocks):
        block_in = h
        for r_i in (1, 2, 3):
            key = f"body.{b}.rdb{r_i}"
            rdb_in = h
            outs = [h]
            for ci in range(1, 6):
                inp = outs[0] if len(outs) == 1 else g.op("Concat", outs, axis=1)
                y = g.conv(inp, f"{key}.conv{ci}")
                if ci < 5:
                    y = g.lrelu(y)
                    outs.append(y)
            h = g.add(g.scale(y, 0.2), rdb_in)
        h = g.add(g.scale(h, 0.2), block_in)

    feat = g.add(feat, g.conv(h, "conv_body"))
    feat = g.lrelu(g.conv(g.up2(feat), "conv_up1"))
    feat = g.lrelu(g.conv(g.up2(feat), "conv_up2"))
    out = g.conv(g.lrelu(g.conv(feat, "conv_hr")), "conv_last")
    return g, out, scale


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    src, dst = sys.argv[1], sys.argv[2]
    sd = load_pth(src)
    if "conv_first.weight" in sd:
        g, out, scale = build_rrdb(sd)
        arch = "RRDBNet"
    elif "body.0.weight" in sd:
        g, out, scale = build_srvgg(sd)
        arch = "SRVGGNetCompact"
    else:
        raise SystemExit("Unbekannte Architektur")

    g.nodes.append(helper.make_node("Identity", [out], ["output"]))
    graph = helper.make_graph(
        g.nodes,
        "bildwerk",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, "h", "w"])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, "H", "W"])],
        g.inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    helper.set_model_props(model, {"scale": str(scale), "architecture": arch})
    onnx.checker.check_model(model)
    onnx.save(model, dst)
    print(f"{src} -> {dst}  ({arch}, x{scale}, {len(g.nodes)} Knoten)")


if __name__ == "__main__":
    main()
