#!/usr/bin/env bash
set -euo pipefail

CONTROL_PLANE="https://vedock.ecorims.com"
INSTALL_ROOT="${1:-${XDG_DATA_HOME:-$HOME/.local/share}/Vedock}"
BIN_ROOT="${HOME}/.local/bin"

printf '\n\033[1;34mVedock connected compute\033[0m\n'
printf 'Web tasks from %s. Training on this computer.\n\n' "$CONTROL_PLANE"

if ! command -v python3 >/dev/null 2>&1 || ! python3 -c 'import sys;raise SystemExit(sys.version_info < (3,11))'; then
  if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update -qq
    sudo apt-get install -y python3 python3-venv python3-pip python3-gi gir1.2-webkit2-4.1
  elif command -v dnf >/dev/null 2>&1; then
    sudo dnf install -y python3 python3-pip python3-gobject webkit2gtk4.1
  else
    printf 'Python 3.11+ is required. Install it with your distribution package manager and retry.\n' >&2
    exit 1
  fi
fi

mkdir -p "$INSTALL_ROOT" "$BIN_ROOT"
ARCHIVE="$INSTALL_ROOT/vedock-client.zip"
if command -v curl >/dev/null 2>&1; then
  curl -fL --progress-bar "$CONTROL_PLANE/downloads/vedock-client.zip" -o "$ARCHIVE"
elif command -v wget >/dev/null 2>&1; then
  wget -O "$ARCHIVE" "$CONTROL_PLANE/downloads/vedock-client.zip"
else
  printf 'curl or wget is required.\n' >&2
  exit 1
fi

python3 -m zipfile -e "$ARCHIVE" "$INSTALL_ROOT/client"
rm -f "$ARCHIVE"
python3 -m venv "$INSTALL_ROOT/runtime"
"$INSTALL_ROOT/runtime/bin/python" -m pip install --disable-pip-version-check --upgrade pip
"$INSTALL_ROOT/runtime/bin/python" -m pip install --disable-pip-version-check -r "$INSTALL_ROOT/client/vedock-client/requirements-client.txt"
"$INSTALL_ROOT/runtime/bin/python" -m pip install --disable-pip-version-check --no-deps -e "$INSTALL_ROOT/client/vedock-client"
ln -sfn "$INSTALL_ROOT/runtime/bin/vedock" "$BIN_ROOT/vedock"

APPLICATIONS="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
mkdir -p "$APPLICATIONS"
cat > "$APPLICATIONS/vedock.desktop" <<EOF
[Desktop Entry]
Name=Vedock
Comment=Connected local AI training
Exec=$INSTALL_ROOT/runtime/bin/vedock ui
Icon=$INSTALL_ROOT/client/vedock-client/vedock_cli/assets/logo.png
Terminal=false
Type=Application
Categories=Development;Science;
EOF

printf '\n\033[1;32mVedock installed.\033[0m\n'
printf 'If needed, add %s to PATH. Then run: vedock login\n' "$BIN_ROOT"
