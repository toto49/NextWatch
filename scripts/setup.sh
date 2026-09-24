#!/usr/bin/env bash
# Setup NextWatch Jupyter environment (Linux / macOS)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MIN_MAJOR=3
MIN_MINOR=10

if command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
elif command -v python >/dev/null 2>&1; then
  PYTHON=python
else
  echo "Erreur: Python n'est pas installé. Installez Python ${MIN_MAJOR}.${MIN_MINOR}+ puis relancez ce script." >&2
  exit 1
fi

VERSION="$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
MAJOR="$("$PYTHON" -c 'import sys; print(sys.version_info.major)')"
MINOR="$("$PYTHON" -c 'import sys; print(sys.version_info.minor)')"

if [[ "$MAJOR" -lt "$MIN_MAJOR" ]] || { [[ "$MAJOR" -eq "$MIN_MAJOR" ]] && [[ "$MINOR" -lt "$MIN_MINOR" ]]; }; then
  echo "Erreur: Python ${MIN_MAJOR}.${MIN_MINOR}+ requis (trouvé: ${VERSION})." >&2
  exit 1
fi

VENV_DIR="$ROOT_DIR/.venv"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Création du venv dans .venv ..."
  "$PYTHON" -m venv "$VENV_DIR"
else
  echo "Venv déjà présent: .venv"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "Mise à jour de pip ..."
python -m pip install --upgrade pip

echo "Installation des dépendances ..."
python -m pip install -r requirements.txt

echo "Enregistrement du kernel Jupyter 'NextWatch' ..."
python -m ipykernel install --user --name nextwatch --display-name "NextWatch"

echo ""
echo "Installation terminée."
echo "Activez l'environnement:  source .venv/bin/activate"
echo "Lancez Jupyter:            jupyter notebook"
echo "Dans Jupyter, choisissez le kernel: NextWatch"
