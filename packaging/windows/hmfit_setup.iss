; HM Fit - Inno Setup installer script
; Build example:
;   iscc /DMyAppVersion=0.1.0-beta.1 /DMyDistDir=dist\hmfit_pyside6 packaging\windows\hmfit_setup.iss

#define MyAppName "HM Fit"
#define MyAppPublisher "HM Fit Team"
#define MyAppURL "https://github.com/PJGV333/GUI_HM_fit"
#define MyAppExeName "hmfit_pyside6.exe"

#ifndef MyAppVersion
  #define MyAppVersion "0.1.0-beta.1"
#endif

#ifndef MyDistDir
  #define MyDistDir "..\\..\\dist\\hmfit_pyside6"
#endif

#ifndef MyOutputDir
  #define MyOutputDir "..\\..\\dist\\installer"
#endif

#ifndef MyOutputBaseFilename
  #define MyOutputBaseFilename "hmfit_setup"
#endif

[Setup]
AppId={{F16F5F5C-79ED-40A3-A05E-61B07D0EA8D8}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\HM Fit
DefaultGroupName=HM Fit
AllowNoIcons=yes
OutputDir={#MyOutputDir}
OutputBaseFilename={#MyOutputBaseFilename}
Compression=lzma2/ultra64
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest
WizardStyle=modern
DisableDirPage=no
DisableProgramGroupPage=no
UninstallDisplayIcon={app}\{#MyAppExeName}
CloseApplications=yes
RestartApplications=no

[Languages]
Name: "spanish"; MessagesFile: "compiler:Languages\Spanish.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "{#MyDistDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\HM Fit"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Desinstalar HM Fit"; Filename: "{uninstallexe}"
Name: "{autodesktop}\HM Fit"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Ejecutar HM Fit"; Flags: nowait postinstall skipifsilent
