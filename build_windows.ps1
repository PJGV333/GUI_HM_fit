param(
    [ValidateSet('All', 'Portable', 'Installer')]
    [string]$Target = 'All',

    [string]$Version,

    [string]$PythonExe,

    [string]$ISCCPath,

    [switch]$PortableOneFile
)

$ErrorActionPreference = 'Stop'

function Invoke-External {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,

        [Parameter(Mandatory = $true)]
        [string[]]$ArgumentList
    )

    Write-Host ">" $FilePath ($ArgumentList -join ' ')
    & $FilePath @ArgumentList

    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $FilePath"
    }
}

function Resolve-PythonExe {
    param(
        [string]$RepoRoot,
        [string]$ExplicitPythonExe
    )

    if ($ExplicitPythonExe) {
        return $ExplicitPythonExe
    }

    $venvPython = Join-Path $RepoRoot '.venv\Scripts\python.exe'
    if (Test-Path $venvPython) {
        return $venvPython
    }

    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCmd) {
        return $pythonCmd.Source
    }

    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        return $pyLauncher.Source
    }

    throw 'No Python executable found. Activate the venv or pass -PythonExe.'
}

function Resolve-Version {
    param(
        [string]$RepoRoot,
        [string]$ExplicitVersion
    )

    if ($ExplicitVersion) {
        return $ExplicitVersion
    }

    $versionFile = Join-Path $RepoRoot 'hmfit_gui_qt\version.py'
    $match = Select-String -Path $versionFile -Pattern '^\s*VERSION\s*=\s*"([^"]+)"' | Select-Object -First 1
    if (-not $match) {
        throw "Could not read VERSION from $versionFile"
    }

    return $match.Matches[0].Groups[1].Value
}

function Resolve-ISCCPath {
    param(
        [string]$ExplicitISCCPath
    )

    if ($ExplicitISCCPath) {
        return $ExplicitISCCPath
    }

    $isccCmd = Get-Command iscc -ErrorAction SilentlyContinue
    if ($isccCmd) {
        return $isccCmd.Source
    }

    $candidates = @(
        'C:\Program Files (x86)\Inno Setup 6\ISCC.exe',
        'C:\Program Files\Inno Setup 6\ISCC.exe'
    )

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return $candidate
        }
    }

    throw 'ISCC.exe not found. Install Inno Setup 6 or pass -ISCCPath.'
}

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

$python = Resolve-PythonExe -RepoRoot $repoRoot -ExplicitPythonExe $PythonExe
$resolvedVersion = Resolve-Version -RepoRoot $repoRoot -ExplicitVersion $Version
$portableOutput = $null
$installerOutput = $null

Write-Host "Repo root:" $repoRoot
Write-Host "Python:" $python
Write-Host "Version:" $resolvedVersion

if ($Target -in @('All', 'Portable')) {
    if ($PortableOneFile) {
        Invoke-External -FilePath $python -ArgumentList @(
            '-m', 'PyInstaller',
            '--noconfirm',
            '--clean',
            '--windowed',
            '--onefile',
            '--name', 'hmfit_pyside6_portable',
            'hmfit_pyside6_entry.py'
        )

        $portableOutput = Join-Path $repoRoot 'dist\hmfit_pyside6_portable.exe'
    }
    else {
        Invoke-External -FilePath $python -ArgumentList @(
            '-m', 'PyInstaller',
            '--noconfirm',
            '--clean',
            'hmfit_pyside6.spec'
        )

        $singleFilePortable = Join-Path $repoRoot 'dist\hmfit_pyside6.exe'
        $onedirPortable = Join-Path $repoRoot 'dist\hmfit_pyside6\hmfit_pyside6.exe'

        if (Test-Path $singleFilePortable) {
            $portableOutput = $singleFilePortable
        }
        elseif (Test-Path $onedirPortable) {
            $portableOutput = $onedirPortable
        }
        else {
            throw 'Portable build finished but no executable was found in dist\.'
        }
    }
}

if ($Target -in @('All', 'Installer')) {
    $iscc = Resolve-ISCCPath -ExplicitISCCPath $ISCCPath
    $distDir = Join-Path $repoRoot 'dist\hmfit_pyside6'
    $issFile = Join-Path $repoRoot 'packaging\windows\hmfit_setup.iss'

    Invoke-External -FilePath $python -ArgumentList @(
        '-m', 'PyInstaller',
        '--noconfirm',
        '--clean',
        '--windowed',
        '--onedir',
        '--name', 'hmfit_pyside6',
        'hmfit_pyside6_entry.py'
    )

    if (-not (Test-Path $distDir)) {
        throw "Expected installer staging directory was not created: $distDir"
    }

    Invoke-External -FilePath $iscc -ArgumentList @(
        "/DMyAppVersion=$resolvedVersion",
        "/DMyDistDir=$distDir",
        $issFile
    )

    $installerOutput = Join-Path $repoRoot 'dist\installer\hmfit_setup.exe'
}

Write-Host ''
Write-Host 'Build outputs:'

if ($portableOutput) {
    Write-Host "Portable:" $portableOutput
    if (-not $PortableOneFile -and $portableOutput -like '*\dist\hmfit_pyside6\hmfit_pyside6.exe') {
        Write-Host 'Note: the current spec produced an onedir build. Use -PortableOneFile for a single-file portable exe.'
    }
}

if ($installerOutput) {
    Write-Host "Installer:" $installerOutput
}
