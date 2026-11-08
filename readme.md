# YOLO26n-seg Dashboard (Flutter web)

Instance segmentation in the browser. Flutter/Dart does the UI, image
preprocessing, mask decoding and drawing; `onnxruntime-web` (WASM) runs the model.
Nothing is uploaded: inference happens locally in the browser tab.

## Setup

```bash
# 1. Export the model (needs Python + ultralytics) and copy it to web/models/
pip install -U ultralytics onnx onnxslim
python export_model.py

# 2. Add the missing Flutter web scaffolding (icons, manifest). Keeps existing files.
flutter create --platforms=web --project-name yolo26_seg_dashboard .
flutter pub get

# 3. Run
flutter run -d chrome
# or: flutter build web && serve build/web
```

After `flutter create`, check that `web/index.html` still contains the two
`<script>` tags for onnxruntime-web and `yolo_bridge.js` (re-copy if it was replaced).

## How it works

| File | Role |
|---|---|
| `web/yolo_bridge.js` | Loads the ONNX model and runs it via onnxruntime-web |
| `lib/yolo_bridge.dart` | `dart:js_interop` bindings to the bridge |
| `lib/yolo26_seg.dart` | Letterbox preprocessing, box + mask decoding (no NMS needed) |
| `lib/main.dart` | Dashboard UI: canvas overlay, threshold slider, metrics, class counts |
| `lib/image_picker_web.dart` | Browser file picker using `package:web` |

Moving the confidence slider re-decodes the cached model output, so it does not
re-run inference.

## Notes

- The model is expected to be an end-to-end YOLO26-seg export with outputs
  `[1, N, 38]` and `[1, 32, H, W]`. If the shape differs, the dashboard shows an
  error with the shapes it received.
- onnxruntime-web and its WASM files load from jsDelivr; the first load needs internet.
  To work offline, copy `ort.min.js` and the `ort-wasm*.wasm` files from the
  `onnxruntime-web` npm package into `web/` and update `index.html` and `wasmPaths`.
- Preprocessing resizes with Flutter's canvas (bilinear/mipmap), which can differ
  slightly from Ultralytics' OpenCV resize. Scores may vary by a small amount.
