# Setup NextWatch Jupyter environment (Windows PowerShell)
$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RootDir

$MinMajor = 3
$MinMinor = 10

$PythonCmd = $null
foreach ($candidate in @("python", "py")) {
    try {
        $null = Get-Command $candidate -ErrorAction Stop
        $PythonCmd = $candidate
        break
    } catch {
        continue
    }
}

if (-not $PythonCmd) {
    Write-Error "Erreur: Python n'est pas installé. Installez Python $MinMajor.$MinMinor+ puis relancez ce script."
    exit 1
}

$VersionInfo = & $PythonCmd -c "import sys; print(sys.version_info.major); print(sys.version_info.minor)"
$VersionLines = $VersionInfo -split "`n" | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" }
$Major = [int]$VersionLines[0]
$Minor = [int]$VersionLines[1]
$Version = "$Major.$Minor"

if ($Major -lt $MinMajor -or ($Major -eq $MinMajor -and $Minor -lt $MinMinor)) {
    Write-Error "Erreur: Python $MinMajor.$MinMinor+ requis (trouvé: $Version)."
    exit 1
}

$VenvDir = Join-Path $RootDir ".venv"

if (-not (Test-Path $VenvDir)) {
    Write-Host "Création du venv dans .venv ..."
    & $PythonCmd -m venv $VenvDir
} else {
    Write-Host "Venv déjà présent: .venv"
}

$ActivateScript = Join-Path $VenvDir "Scripts\Activate.ps1"
. $ActivateScript

Write-Host "Mise à jour de pip ..."
python -m pip install --upgrade pip

Write-Host "Installation des dépendances ..."
python -m pip install -r requirements.txt

Write-Host "Enregistrement du kernel Jupyter 'NextWatch' ..."
python -m ipykernel install --user --name nextwatch --display-name "NextWatch"

Write-Host ""
Write-Host "Installation terminée."
Write-Host "Activez l'environnement:  .\.venv\Scripts\Activate.ps1"
Write-Host "Lancez Jupyter:            jupyter notebook"
Write-Host "Dans Jupyter, choisissez le kernel: NextWatch"
