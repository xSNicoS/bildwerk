// Bettet unter Windows Programmsymbol und Versionsinformationen in die .exe ein.
fn main() {
    println!("cargo:rerun-if-changed=assets/icon.ico");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }
    let mut res = winresource::WindowsResource::new();
    // Beim Kreuzkompilieren unter Linux das MinGW-windres verwenden
    if std::env::var("HOST").map(|h| !h.contains("windows")).unwrap_or(false) {
        res.set_windres_path("x86_64-w64-mingw32-windres");
        res.set_ar_path("x86_64-w64-mingw32-ar");
    }
    res.set_icon("assets/icon.ico")
        .set("ProductName", "Bildwerk")
        .set("FileDescription", "Bildwerk – KI-Bildverbesserung")
        .set("CompanyName", "Bildwerk")
        .set("LegalCopyright", "MIT-Lizenz");
    if let Err(e) = res.compile() {
        println!("cargo:warning=Symbol konnte nicht eingebettet werden: {e}");
    }
}
