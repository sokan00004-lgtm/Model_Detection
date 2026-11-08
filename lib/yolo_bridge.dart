import 'dart:js_interop';
import 'dart:typed_data';

// --- JS interop declarations (see web/yolo_bridge.js) ----------------------

@JS('yoloBridge')
external _YoloBridge get _yoloBridge;

extension type _YoloBridge._(JSObject _) implements JSObject {
  external JSPromise<JSString> load(JSString url);
  external JSPromise<_JsOutputs> run(JSFloat32Array data, JSNumber size);
}

extension type _JsOutputs._(JSObject _) implements JSObject {
  external JSFloat32Array get predData;
  external JSArray<JSNumber> get predDims;
  external JSFloat32Array get protoData;
  external JSArray<JSNumber> get protoDims;
}

// --- Plain Dart API ---------------------------------------------------------

/// Raw model outputs, already converted to Dart types.
class RawOutputs {
  const RawOutputs({
    required this.pred,
    required this.predDims,
    required this.proto,
    required this.protoDims,
  });

  /// Detections tensor, shape [1, N, 38]:
  /// x1, y1, x2, y2, score, classId, 32 mask coefficients.
  final Float32List pred;
  final List<int> predDims;

  /// Mask prototypes, shape [1, 32, H, W].
  final Float32List proto;
  final List<int> protoDims;
}

class YoloRuntime {
  /// Loads the ONNX model from [url]. Returns "inputs -> outputs" info.
  Future<String> load(String url) async {
    final info = await _yoloBridge.load(url.toJS).toDart;
    return info.toDart;
  }

  /// Runs the model on a CHW float tensor of size 3 x [size] x [size].
  Future<RawOutputs> run(Float32List input, int size) async {
    final r = await _yoloBridge.run(input.toJS, size.toJS).toDart;
    return RawOutputs(
      pred: r.predData.toDart,
      predDims: r.predDims.toDart.map((e) => e.toDartInt).toList(),
      proto: r.protoData.toDart,
      protoDims: r.protoDims.toDart.map((e) => e.toDartInt).toList(),
    );
  }
}
