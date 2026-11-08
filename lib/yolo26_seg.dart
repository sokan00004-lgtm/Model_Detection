import 'dart:async';
import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'coco_classes.dart';
import 'yolo_bridge.dart';

/// Square network input size the ONNX model was exported with.
const int kInputSize = 640;

const List<int> _palette = [
  0xFFEF4444, 0xFF3B82F6, 0xFF22C55E, 0xFFF59E0B, 0xFFA855F7,
  0xFF06B6D4, 0xFFEC4899, 0xFF84CC16, 0xFFF97316, 0xFF6366F1,
  0xFF14B8A6, 0xFFE11D48, 0xFF0EA5E9, 0xFFEAB308, 0xFF8B5CF6,
  0xFF10B981, 0xFFD946EF, 0xFF64748B, 0xFFFB7185, 0xFF2DD4BF,
];

/// Geometry of the letterbox (resize + gray padding) applied to the source image.
class Letterbox {
  const Letterbox({
    required this.srcW,
    required this.srcH,
    required this.scale,
    required this.newW,
    required this.newH,
    required this.padX,
    required this.padY,
  });

  final int srcW, srcH;
  final double scale;
  final int newW, newH;
  final double padX, padY;
}

class Preprocessed {
  const Preprocessed(this.tensor, this.letterbox);
  final Float32List tensor;
  final Letterbox letterbox;
}

class Detection {
  const Detection({
    required this.box,
    required this.lbBox,
    required this.score,
    required this.classId,
    required this.coeffs,
  });

  /// Box in source-image pixels.
  final ui.Rect box;

  /// Box in the 640x640 letterboxed network space.
  final ui.Rect lbBox;
  final double score;
  final int classId;
  final Float32List coeffs;

  String get label =>
      classId >= 0 && classId < kCocoClasses.length
          ? kCocoClasses[classId]
          : 'class $classId';

  int get colorValue => _palette[classId.abs() % _palette.length];
  ui.Color get color => ui.Color(colorValue);
}

class SegResult {
  const SegResult(this.detections, this.maskImage);

  /// Sorted by score, highest first.
  final List<Detection> detections;

  /// RGBA overlay in letterboxed space (proto resolution, e.g. 160x160).
  final ui.Image? maskImage;
}

double _clampD(double v, double lo, double hi) =>
    v < lo ? lo : (v > hi ? hi : v);

double _iou(ui.Rect a, ui.Rect b) {
  final l = math.max(a.left, b.left);
  final t = math.max(a.top, b.top);
  final r = math.min(a.right, b.right);
  final bt = math.min(a.bottom, b.bottom);
  final iw = r - l;
  final ih = bt - t;
  if (iw <= 0 || ih <= 0) return 0;
  final inter = iw * ih;
  final union = a.width * a.height + b.width * b.height - inter;
  return union <= 0 ? 0 : inter / union;
}

class Yolo26Seg {
  final YoloRuntime _runtime = YoloRuntime();

  Future<String> load(String url) => _runtime.load(url);

  Future<RawOutputs> run(Float32List tensor) =>
      _runtime.run(tensor, kInputSize);

  /// Letterboxes [image] to 640x640 and converts it to a CHW float tensor.
  Future<Preprocessed> preprocess(ui.Image image) async {
    const s = kInputSize;
    final scale = math.min(s / image.width, s / image.height);
    final newW = (image.width * scale).round();
    final newH = (image.height * scale).round();
    final padX = ((s - newW) / 2).floorToDouble();
    final padY = ((s - newH) / 2).floorToDouble();

    final recorder = ui.PictureRecorder();
    final canvas = ui.Canvas(recorder);
    canvas.drawRect(
      ui.Rect.fromLTWH(0, 0, s.toDouble(), s.toDouble()),
      ui.Paint()..color = const ui.Color(0xFF727272), // 114 gray, like Ultralytics
    );
    canvas.drawImageRect(
      image,
      ui.Rect.fromLTWH(0, 0, image.width.toDouble(), image.height.toDouble()),
      ui.Rect.fromLTWH(padX, padY, newW.toDouble(), newH.toDouble()),
      ui.Paint()..filterQuality = ui.FilterQuality.medium,
    );
    final picture = recorder.endRecording();
    final boxed = await picture.toImage(s, s);
    final data = await boxed.toByteData(format: ui.ImageByteFormat.rawRgba);
    picture.dispose();
    boxed.dispose();
    if (data == null) throw StateError('Could not read image pixels');

    final px = data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);
    const plane = s * s;
    final tensor = Float32List(3 * plane);
    for (var i = 0; i < plane; i++) {
      final j = i * 4;
      tensor[i] = px[j] / 255.0;
      tensor[plane + i] = px[j + 1] / 255.0;
      tensor[2 * plane + i] = px[j + 2] / 255.0;
    }

    return Preprocessed(
      tensor,
      Letterbox(
        srcW: image.width,
        srcH: image.height,
        scale: scale,
        newW: newW,
        newH: newH,
        padX: padX,
        padY: padY,
      ),
    );
  }

  /// Turns raw model outputs into detections + one combined mask overlay.
  ///
  /// YOLO26 is NMS-free (end-to-end), so there is no NMS step: every row of the
  /// [1, N, 38] tensor is already a final detection candidate.
  Future<SegResult> postprocess(
    RawOutputs raw,
    Letterbox lb, {
    required double confThreshold,
  }) async {
    final nm = raw.protoDims[1];
    final ph = raw.protoDims[2];
    final pw = raw.protoDims[3];
    if (nm != 32) throw StateError('Expected 32 mask prototypes, got $nm');

    final detections = _decode(raw, lb, confThreshold, nm);
    detections.sort((a, b) => b.score.compareTo(a.score));
    if (detections.isEmpty) return const SegResult([], null);

    // 2) Decode instance masks: sigmoid(coeffs @ protos), cropped to the box.
    //    sigmoid(v) > 0.5  <=>  v > 0, so the sigmoid itself can be skipped.
    final proto = raw.proto;
    final plane = ph * pw;
    final rx = pw / kInputSize;
    final ry = ph / kInputSize;
    final overlay = Uint8List(plane * 4);
    const alpha = 120;

    // Paint low-score first so the most confident instance ends up on top.
    for (final d in detections.reversed) {
      final mx1 = (d.lbBox.left * rx).floor().clamp(0, pw - 1).toInt();
      final my1 = (d.lbBox.top * ry).floor().clamp(0, ph - 1).toInt();
      final mx2 = (d.lbBox.right * rx).ceil().clamp(0, pw).toInt();
      final my2 = (d.lbBox.bottom * ry).ceil().clamp(0, ph).toInt();
      final c = d.colorValue;
      final r = ((c >> 16) & 0xFF) * alpha ~/ 255; // premultiplied
      final g = ((c >> 8) & 0xFF) * alpha ~/ 255;
      final b = (c & 0xFF) * alpha ~/ 255;
      final coeffs = d.coeffs;

      for (var y = my1; y < my2; y++) {
        for (var x = mx1; x < mx2; x++) {
          final base = y * pw + x;
          var v = 0.0;
          for (var k = 0; k < nm; k++) {
            v += coeffs[k] * proto[k * plane + base];
          }
          if (v > 0) {
            final q = base * 4;
            overlay[q] = r;
            overlay[q + 1] = g;
            overlay[q + 2] = b;
            overlay[q + 3] = alpha;
          }
        }
      }
    }

    final maskImage = await _imageFromRgba(overlay, pw, ph);
    return SegResult(detections, maskImage);
  }

  /// Decodes the predictions tensor into detections. Supports two layouts:
  ///  * end-to-end export  [1, N, 6 + 32]: x1,y1,x2,y2,score,class,coeffs
  ///    (NMS already applied by the model)
  ///  * classic export     [1, 4 + nc + 32, A]: cx,cy,w,h, class scores, coeffs
  ///    (needs NMS, done here)
  List<Detection> _decode(RawOutputs raw, Letterbox lb, double conf, int nm) {
    final dims = raw.predDims;
    if (dims.length != 3) {
      throw StateError('Unexpected predictions shape $dims');
    }
    final p = raw.pred;
    final out = <Detection>[];
    final size = kInputSize.toDouble();

    void add(double x1, double y1, double x2, double y2, double score, int cls,
        Float32List coeffs) {
      x1 = _clampD(x1, 0, size);
      y1 = _clampD(y1, 0, size);
      x2 = _clampD(x2, 0, size);
      y2 = _clampD(y2, 0, size);
      if (x2 <= x1 || y2 <= y1) return;
      out.add(Detection(
        box: ui.Rect.fromLTRB(
          _clampD((x1 - lb.padX) / lb.scale, 0, lb.srcW.toDouble()),
          _clampD((y1 - lb.padY) / lb.scale, 0, lb.srcH.toDouble()),
          _clampD((x2 - lb.padX) / lb.scale, 0, lb.srcW.toDouble()),
          _clampD((y2 - lb.padY) / lb.scale, 0, lb.srcH.toDouble()),
        ),
        lbBox: ui.Rect.fromLTRB(x1, y1, x2, y2),
        score: score,
        classId: cls,
        coeffs: coeffs,
      ));
    }

    // --- End-to-end layout: [1, N, 38] ---------------------------------
    if (dims[2] == 6 + nm) {
      final n = dims[1];
      final stride = dims[2];
      var maxCoord = 0.0;
      for (var i = 0; i < n; i++) {
        final o = i * stride;
        if (p[o + 4] < conf) continue;
        maxCoord = math.max(maxCoord, math.max(p[o + 2], p[o + 3]));
      }
      final s = (maxCoord > 0 && maxCoord <= 2.0) ? size : 1.0;
      for (var i = 0; i < n; i++) {
        final o = i * stride;
        if (p[o + 4] < conf) continue;
        add(p[o] * s, p[o + 1] * s, p[o + 2] * s, p[o + 3] * s, p[o + 4],
            p[o + 5].round(), Float32List.fromList(p.sublist(o + 6, o + 6 + nm)));
      }
      return out;
    }

    // --- Classic layout: [1, 4 + nc + 32, A] (e.g. [1, 116, 8400]) -------
    if (dims[1] > 4 + nm && dims[1] < dims[2]) {
      final channels = dims[1];
      final a = dims[2];
      final nc = channels - 4 - nm;
      for (var i = 0; i < a; i++) {
        var best = 0;
        var bestScore = p[4 * a + i];
        for (var k = 1; k < nc; k++) {
          final sc = p[(4 + k) * a + i];
          if (sc > bestScore) {
            bestScore = sc;
            best = k;
          }
        }
        if (bestScore < conf) continue;
        final cx = p[i], cy = p[a + i], w = p[2 * a + i], h = p[3 * a + i];
        final coeffs = Float32List(nm);
        for (var k = 0; k < nm; k++) {
          coeffs[k] = p[(4 + nc + k) * a + i];
        }
        add(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, bestScore, best,
            coeffs);
      }

      // Class-aware NMS (IoU 0.7, max 100 detections).
      out.sort((x, y) => y.score.compareTo(x.score));
      final kept = <Detection>[];
      for (final d in out) {
        if (kept.length >= 100) break;
        var keep = true;
        for (final k in kept) {
          if (k.classId == d.classId && _iou(k.lbBox, d.lbBox) > 0.7) {
            keep = false;
            break;
          }
        }
        if (keep) kept.add(d);
      }
      return kept;
    }

    throw StateError(
      'Unsupported predictions shape $dims. Expected [1, N, 38] (end-to-end) '
      'or [1, 4 + classes + 32, anchors] (classic).',
    );
  }

  Future<ui.Image> _imageFromRgba(Uint8List pixels, int w, int h) {
    final completer = Completer<ui.Image>();
    ui.decodeImageFromPixels(pixels, w, h, ui.PixelFormat.rgba8888, completer.complete);
    return completer.future;
  }
}
