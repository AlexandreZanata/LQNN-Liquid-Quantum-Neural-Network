#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo "  LQNN v3 - Desktop App Builder (Tauri)"
echo "============================================"
echo

check_cmd() {
    if ! command -v "$1" &>/dev/null; then
        echo "[ERROR] '$1' is not installed."
        echo "  $2"
        exit 1
    fi
}

echo "[1/5] Checking prerequisites..."
check_cmd "cargo"  "Install Rust: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
check_cmd "node"   "Install Node.js: https://nodejs.org/ or 'sudo apt install nodejs'"
check_cmd "npm"    "Install npm: comes with Node.js"

echo "[2/5] Checking system libraries..."
MISSING_LIBS=""
for lib in libwebkit2gtk-4.1-dev libappindicator3-dev librsvg2-dev patchelf; do
    if ! dpkg -s "$lib" &>/dev/null 2>&1; then
        MISSING_LIBS="$MISSING_LIBS $lib"
    fi
done

if [ -n "$MISSING_LIBS" ]; then
    echo "[WARN] Missing system libraries:$MISSING_LIBS"
    echo "  Install with: sudo apt install$MISSING_LIBS"
    read -rp "  Install now? [y/N] " yn
    if [[ "$yn" =~ ^[Yy]$ ]]; then
        sudo apt update && sudo apt install -y $MISSING_LIBS
    else
        echo "  Skipping -- build may fail without these libraries."
    fi
fi

echo "[3/5] Installing npm dependencies..."
npm install

echo "[4/5] Building Tauri application..."
npx tauri build

echo "[5/5] Done!"
echo
echo "Build artifacts:"
find src-tauri/target/release/bundle -name "*.deb" -o -name "*.AppImage" 2>/dev/null | while read -r f; do
    echo "  -> $f ($(du -h "$f" | cut -f1))"
done
echo
echo "To install the .deb:"
echo "  sudo dpkg -i src-tauri/target/release/bundle/deb/*.deb"
echo
echo "To run the AppImage directly:"
echo "  chmod +x src-tauri/target/release/bundle/appimage/*.AppImage"
echo "  ./src-tauri/target/release/bundle/appimage/*.AppImage"
