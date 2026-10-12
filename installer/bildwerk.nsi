; Bildwerk – Windows-Installer (NSIS 3)
; Wird von tools/build_windows.sh aufgerufen:
;   makensis -DVERSION=0.2.0 -DSTAGE=<Programmordner> -DOUTFILE=<Setup.exe> installer/bildwerk.nsi

Target amd64-unicode
ManifestDPIAware true
SetCompressor /SOLID lzma
SetCompressorDictSize 64

!ifndef VERSION
  !define VERSION "0.0.0"
!endif
!ifndef STAGE
  !error "STAGE (Ordner mit Programmdateien) fehlt"
!endif
!ifndef OUTFILE
  !define OUTFILE "Bildwerk-Setup-${VERSION}.exe"
!endif

!define APP       "Bildwerk"
!define EXE       "Bildwerk.exe"
!define PUBLISHER "Bildwerk"
!define REGKEY    "Software\Bildwerk"
!define UNINSTKEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\Bildwerk"

Name "${APP}"
Caption "${APP} ${VERSION} – Installation"
UninstallCaption "${APP} – Deinstallation"
OutFile "${OUTFILE}"
InstallDir "$PROGRAMFILES64\${APP}"
InstallDirRegKey HKLM "${REGKEY}" "InstallDir"
RequestExecutionLevel admin
BrandingText "${APP} ${VERSION}"
ShowInstDetails nevershow
ShowUninstDetails nevershow

VIProductVersion "${VERSION}.0"
VIAddVersionKey /LANG=1031 "ProductName" "${APP}"
VIAddVersionKey /LANG=1031 "FileDescription" "${APP} – Installation"
VIAddVersionKey /LANG=1031 "ProductVersion" "${VERSION}"
VIAddVersionKey /LANG=1031 "FileVersion" "${VERSION}"
VIAddVersionKey /LANG=1031 "CompanyName" "${PUBLISHER}"
VIAddVersionKey /LANG=1031 "LegalCopyright" "MIT-Lizenz"

!include "MUI2.nsh"
!include "x64.nsh"
!include "FileFunc.nsh"
!include "WinVer.nsh"

; ------------------------------------------------------------------ Aussehen

!define MUI_ICON   "icon.ico"
!define MUI_UNICON "icon.ico"
!define MUI_WELCOMEFINISHPAGE_BITMAP   "welcome.bmp"
!define MUI_UNWELCOMEFINISHPAGE_BITMAP "welcome.bmp"
!define MUI_HEADERIMAGE
!define MUI_HEADERIMAGE_RIGHT
!define MUI_HEADERIMAGE_BITMAP   "header.bmp"
!define MUI_HEADERIMAGE_UNBITMAP "header.bmp"
!define MUI_ABORTWARNING
!define MUI_COMPONENTSPAGE_SMALLDESC

!define MUI_WELCOMEPAGE_TITLE "Willkommen bei Bildwerk"
!define MUI_WELCOMEPAGE_TEXT "Bildwerk verbessert Fotos mit künstlicher Intelligenz: vergrößern, entrauschen, Gesichter restaurieren und nachschärfen – komplett offline auf diesem Computer.$\r$\n$\r$\nAlle KI-Modelle sind enthalten. Benötigt werden etwa 550 MB Speicherplatz.$\r$\n$\r$\nKlicken Sie auf „Weiter“, um fortzufahren."

!define MUI_FINISHPAGE_TITLE "Bildwerk ist installiert"
!define MUI_FINISHPAGE_TEXT "Sie finden Bildwerk im Startmenü.$\r$\n$\r$\nTipp: Ziehen Sie ein Foto einfach ins Programmfenster oder klicken Sie im Explorer mit der rechten Maustaste auf ein Bild und wählen Sie „Mit Bildwerk verbessern“."
!define MUI_FINISHPAGE_RUN "$INSTDIR\${EXE}"
!define MUI_FINISHPAGE_RUN_TEXT "Bildwerk jetzt starten"
!define MUI_FINISHPAGE_NOREBOOTSUPPORT

; ------------------------------------------------------------------ Seiten

!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_LICENSE "LIZENZ.txt"
!insertmacro MUI_PAGE_COMPONENTS
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

!insertmacro MUI_LANGUAGE "German"

; ------------------------------------------------------------------ Prüfungen

Function .onInit
  ${IfNot} ${RunningX64}
    MessageBox MB_OK|MB_ICONSTOP "Bildwerk benötigt ein 64-Bit-Windows."
    Abort
  ${EndIf}
  ${IfNot} ${AtLeastWin10}
    MessageBox MB_YESNO|MB_ICONEXCLAMATION "Bildwerk ist für Windows 10 und 11 gedacht. Trotzdem installieren?" IDYES +2
    Abort
  ${EndIf}
  SetRegView 64

  ; Läuft Bildwerk gerade? Dann können die Dateien nicht ersetzt werden.
  FindWindow $0 "" "Bildwerk"
  ${If} $0 != 0
    MessageBox MB_OKCANCEL|MB_ICONEXCLAMATION "Bildwerk ist noch geöffnet. Bitte schließen Sie das Programm und klicken Sie dann auf OK." IDOK +2
    Abort
  ${EndIf}
FunctionEnd

Function un.onInit
  SetRegView 64
FunctionEnd

; ------------------------------------------------------------------ Bestandteile

Section "Bildwerk (erforderlich)" SecMain
  SectionIn RO
  SetShellVarContext all

  ; Reste einer älteren Version entfernen (Modelle werden komplett ersetzt)
  RMDir /r "$INSTDIR\models"

  SetOutPath "$INSTDIR"
  File /r "${STAGE}\*.*"
  File "icon.ico"

  ; Startmenü
  CreateShortcut "$SMPROGRAMS\${APP}.lnk" "$INSTDIR\${EXE}" "" "$INSTDIR\${EXE}" 0 SW_SHOWNORMAL "" "KI-Bildverbesserung"

  ; „Öffnen mit“-Liste und Aufruf über Win+R
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\App Paths\${EXE}" "" "$INSTDIR\${EXE}"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\App Paths\${EXE}" "Path" "$INSTDIR"
  WriteRegStr HKLM "Software\Classes\Applications\${EXE}" "FriendlyAppName" "${APP}"
  WriteRegStr HKLM "Software\Classes\Applications\${EXE}\DefaultIcon" "" "$INSTDIR\${EXE},0"
  WriteRegStr HKLM "Software\Classes\Applications\${EXE}\shell\open\command" "" '"$INSTDIR\${EXE}" "%1"'
  !macro SupportType EXT
    WriteRegStr HKLM "Software\Classes\Applications\${EXE}\SupportedTypes" "${EXT}" ""
    WriteRegStr HKLM "Software\Classes\${EXT}\OpenWithList\${EXE}" "" ""
  !macroend
  !insertmacro SupportType ".jpg"
  !insertmacro SupportType ".jpeg"
  !insertmacro SupportType ".png"
  !insertmacro SupportType ".tif"
  !insertmacro SupportType ".tiff"
  !insertmacro SupportType ".webp"
  !insertmacro SupportType ".bmp"

  ; Deinstallation in „Apps & Features“
  WriteUninstaller "$INSTDIR\Deinstallieren.exe"
  WriteRegStr HKLM "${REGKEY}" "InstallDir" "$INSTDIR"
  WriteRegStr HKLM "${UNINSTKEY}" "DisplayName" "${APP}"
  WriteRegStr HKLM "${UNINSTKEY}" "DisplayVersion" "${VERSION}"
  WriteRegStr HKLM "${UNINSTKEY}" "Publisher" "${PUBLISHER}"
  WriteRegStr HKLM "${UNINSTKEY}" "DisplayIcon" "$INSTDIR\${EXE},0"
  WriteRegStr HKLM "${UNINSTKEY}" "InstallLocation" "$INSTDIR"
  WriteRegStr HKLM "${UNINSTKEY}" "UninstallString" '"$INSTDIR\Deinstallieren.exe"'
  WriteRegStr HKLM "${UNINSTKEY}" "QuietUninstallString" '"$INSTDIR\Deinstallieren.exe" /S'
  WriteRegDWORD HKLM "${UNINSTKEY}" "NoModify" 1
  WriteRegDWORD HKLM "${UNINSTKEY}" "NoRepair" 1
  ${GetSize} "$INSTDIR" "/S=0K" $0 $1 $2
  IntFmt $0 "0x%08X" $0
  WriteRegDWORD HKLM "${UNINSTKEY}" "EstimatedSize" "$0"
SectionEnd

Section "Verknüpfung auf dem Desktop" SecDesktop
  SetShellVarContext all
  CreateShortcut "$DESKTOP\${APP}.lnk" "$INSTDIR\${EXE}" "" "$INSTDIR\${EXE}" 0 SW_SHOWNORMAL "" "KI-Bildverbesserung"
SectionEnd

Section "Kontextmenü „Mit Bildwerk verbessern“" SecContext
  WriteRegStr HKLM "Software\Classes\SystemFileAssociations\image\shell\Bildwerk" "" "Mit Bildwerk verbessern"
  WriteRegStr HKLM "Software\Classes\SystemFileAssociations\image\shell\Bildwerk" "Icon" '"$INSTDIR\${EXE}",0'
  WriteRegStr HKLM "Software\Classes\SystemFileAssociations\image\shell\Bildwerk\command" "" '"$INSTDIR\${EXE}" "%1"'
SectionEnd

Section "-Abschluss"
  ; Explorer über geänderte Dateizuordnungen informieren
  System::Call 'shell32::SHChangeNotify(i 0x08000000, i 0, p 0, p 0)'
SectionEnd

!insertmacro MUI_FUNCTION_DESCRIPTION_BEGIN
  !insertmacro MUI_DESCRIPTION_TEXT ${SecMain}    "Das Programm mit allen KI-Modellen (Vergrößern, Entrauschen, Gesichter) und Grafikkarten-Unterstützung."
  !insertmacro MUI_DESCRIPTION_TEXT ${SecDesktop} "Legt ein Bildwerk-Symbol auf dem Desktop an."
  !insertmacro MUI_DESCRIPTION_TEXT ${SecContext} "Rechtsklick auf ein Bild im Explorer → „Mit Bildwerk verbessern“. (Unter Windows 11 unter „Weitere Optionen anzeigen“.)"
!insertmacro MUI_FUNCTION_DESCRIPTION_END

; ------------------------------------------------------------------ Deinstallation

Section "Uninstall"
  SetShellVarContext all

  Delete "$SMPROGRAMS\${APP}.lnk"
  Delete "$DESKTOP\${APP}.lnk"

  ; Nur eigene Dateien löschen – nie blind den ganzen Ordner
  Delete "$INSTDIR\${EXE}"
  Delete "$INSTDIR\icon.ico"
  Delete "$INSTDIR\onnxruntime.dll"
  Delete "$INSTDIR\onnxruntime_providers_shared.dll"
  Delete "$INSTDIR\DirectML.dll"
  Delete "$INSTDIR\msvcp140.dll"
  Delete "$INSTDIR\msvcp140_1.dll"
  Delete "$INSTDIR\vcruntime140.dll"
  Delete "$INSTDIR\vcruntime140_1.dll"
  Delete "$INSTDIR\LIZENZ.txt"
  Delete "$INSTDIR\LIESMICH.txt"
  RMDir /r "$INSTDIR\models"
  RMDir /r "$INSTDIR\licenses"
  Delete "$INSTDIR\Deinstallieren.exe"
  RMDir "$INSTDIR"

  DeleteRegKey HKLM "Software\Classes\SystemFileAssociations\image\shell\Bildwerk"
  DeleteRegKey HKLM "Software\Classes\Applications\${EXE}"
  DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\App Paths\${EXE}"
  !macro RemoveType EXT
    DeleteRegKey HKLM "Software\Classes\${EXT}\OpenWithList\${EXE}"
  !macroend
  !insertmacro RemoveType ".jpg"
  !insertmacro RemoveType ".jpeg"
  !insertmacro RemoveType ".png"
  !insertmacro RemoveType ".tif"
  !insertmacro RemoveType ".tiff"
  !insertmacro RemoveType ".webp"
  !insertmacro RemoveType ".bmp"
  DeleteRegKey HKLM "${UNINSTKEY}"
  DeleteRegKey HKLM "${REGKEY}"

  System::Call 'shell32::SHChangeNotify(i 0x08000000, i 0, p 0, p 0)'
SectionEnd
