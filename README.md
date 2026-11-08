# YOLO26n-seg Dashboard (Flutter web)

Instance segmentation in the browser. Flutter/Dart does the UI, image
preprocessing, mask decoding and drawing; `onnxruntime-web` (WASM) runs the model.
Nothing is uploaded: inference happens locally in the browser tab.

## Quick start (new folder)

```bash
unzip yolo26_seg_dashboard.zip
cd yolo26_seg_dashboard
chmod +x setup.sh
./setup.sh
```

`setup.sh` creates a CPU-only Python venv, exports `yolo26n-seg` to ONNX into
`web/models/`, adds the Flutter web scaffolding, runs `flutter pub get` and starts
the app in Chrome. Options: `--no-run`, `--skip-model`.

Manual equivalent:

```bash
python3 -m venv .venv-export && source .venv-export/bin/activate
pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install --no-cache-dir ultralytics onnx onnxslim
python export_model.py && deactivate

cp web/index.html /tmp/index.html.bak
flutter create --platforms=web --project-name yolo26_seg_dashboard .
cp /tmp/index.html.bak web/index.html && rm -f test/widget_test.dart
flutter pub get
flutter run -d chrome
```

## Add to an existing Flutter project instead

```bash
cp lib/*.dart  /path/to/your_project/lib/
cp web/index.html web/yolo_bridge.js /path/to/your_project/web/
cp export_model.py /path/to/your_project/
cd /path/to/your_project
rm -f test/widget_test.dart
flutter pub add web
python export_model.py        # inside a venv with ultralytics
flutter run -d chrome
```

## How it works

| File | Role |
|---|---|
| `web/yolo_bridge.js` | Loads the ONNX model and runs it via onnxruntime-web |
| `lib/yolo_bridge.dart` | `dart:js_interop` bindings to the bridge |
| `lib/yolo26_seg.dart` | Letterbox preprocessing, box + mask decoding, NMS when needed |
| `lib/main.dart` | Dashboard UI: canvas overlay, threshold slider, metrics, class counts |
| `lib/image_picker_web.dart` | Browser file picker using `package:web` |

The decoder accepts both ONNX layouts:
- classic `[1, 116, 8400]` (4 box + 80 classes + 32 mask coefficients, NMS done in Dart)
- end-to-end `[1, N, 38]` (NMS already done by the model)

Moving the confidence slider re-decodes the cached model output, so it does not
re-run inference.

## Troubleshooting

- **"Disk quota exceeded" during pip:** run `pip cache purge` and use the CPU-only
  PyTorch command above.
- **Old error text still shows:** you are running a different copy of the project, or
  the browser cached the old build. Run `flutter clean`, then `flutter run -d chrome`,
  and hard-refresh with `Ctrl+Shift+R`.
- **"Could not load web/models/yolo26n-seg.onnx":** the export step did not finish.
- **No detections:** lower the confidence slider to about 0.10.
- onnxruntime-web files load from jsDelivr, so the first load needs internet.
