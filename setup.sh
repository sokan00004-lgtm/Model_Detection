#!/usr/bin/env bash
# One-command setup for the YOLO26n-seg Flutter web dashboard.
#
#   ./setup.sh           full setup: export model + prepare Flutter + run in Chrome
#   ./setup.sh --no-run  same, but do not start the app at the end
#   ./setup.sh --skip-model   skip the model export (you already have the .onnx)
#
# Needs: python3 (3.10-3.13 recommended) and Flutter on PATH.
set -euo pipefail
cd "$(dirname "$0")"

RUN=1
SKIP_MODEL=0
for arg in "$@"; do
  case "$arg" in
    --no-run) RUN=0 ;;
    --skip-model) SKIP_MODEL=1 ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
  esac
done

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

command -v flutter >/dev/null || { echo "Flutter not found on PATH. Install it from https://docs.flutter.dev/get-started/install"; exit 1; }

# ---------------------------------------------------------------- model ----
MODEL=web/models/yolo26n-seg.onnx
if [ "$SKIP_MODEL" -eq 0 ] && [ ! -f "$MODEL" ]; then
  command -v python3 >/dev/null || { echo "python3 not found"; exit 1; }

  say "Creating Python venv (.venv-export) with CPU-only PyTorch (small download)"
  python3 -m venv .venv-export
  # shellcheck disable=SC1091
  source .venv-export/bin/activate
  pip install --no-cache-dir --upgrade pip
  pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
  pip install --no-cache-dir ultralytics onnx onnxslim

  say "Exporting yolo26n-seg to ONNX"
  python export_model.py
  deactivate
else
  say "Model step skipped (found $MODEL or --skip-model)"
fi

[ -f "$MODEL" ] || { echo "Missing $MODEL. Run the export or copy your .onnx there."; exit 1; }

# -------------------------------------------------------------- flutter ----
say "Preparing Flutter web scaffolding"
# flutter create may overwrite web/index.html, so keep our copy and restore it.
cp web/index.html .index.html.bak
flutter create --platforms=web --project-name yolo26_seg_dashboard . >/dev/null
cp .index.html.bak web/index.html
rm -f .index.html.bak
rm -f test/widget_test.dart   # default template test references a MyApp we replaced

flutter pub get

if [ "$RUN" -eq 1 ]; then
  say "Starting app in Chrome (press q to quit)"
  flutter run -d chrome
else
  say "Done. Start it with:  flutter run -d chrome"
fi
