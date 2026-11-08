import 'dart:math' as math;
import 'dart:ui' as ui;

import 'package:flutter/material.dart';

import 'image_picker_web.dart';
import 'yolo26_seg.dart';
import 'yolo_bridge.dart';

void main() => runApp(const DashboardApp());

class DashboardApp extends StatelessWidget {
  const DashboardApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'YOLO26n-seg Dashboard Demo001',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorSchemeSeed: const Color(0xFF14B8A6),
        brightness: Brightness.dark,
        useMaterial3: true,
      ),
      home: const DashboardPage(),
    );
  }
}

class DashboardPage extends StatefulWidget {
  const DashboardPage({super.key});

  @override
  State<DashboardPage> createState() => _DashboardPageState();
}

class _DashboardPageState extends State<DashboardPage> {
  static const _modelUrl = 'models/yolo26n-seg.onnx';

  final _yolo = Yolo26Seg();

  bool _modelReady = false;
  String? _modelError;
  int _loadMs = 0;

  ui.Image? _image;
  RawOutputs? _raw;
  Letterbox? _lb;
  SegResult? _result;
  String? _runError;

  double _conf = 0.25;
  bool _showMasks = true;
  bool _showBoxes = true;
  bool _busy = false;
  int _preMs = 0, _inferMs = 0, _postMs = 0;

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  Future<void> _loadModel() async {
    final sw = Stopwatch()..start();
    try {
      await _yolo.load(_modelUrl);
      if (!mounted) return;
      setState(() {
        _modelReady = true;
        _loadMs = sw.elapsedMilliseconds;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() => _modelError = '$e');
    }
  }

  void _disposeLater(ui.Image? img) {
    if (img == null) return;
    WidgetsBinding.instance.addPostFrameCallback((_) => img.dispose());
  }

  Future<void> _pickImage() async {
    final bytes = await pickImageBytes();
    if (bytes == null) return;
    try {
      final codec = await ui.instantiateImageCodec(bytes);
      final frame = await codec.getNextFrame();
      codec.dispose();
      final oldImage = _image;
      final oldMask = _result?.maskImage;
      setState(() {
        _image = frame.image;
        _result = null;
        _raw = null;
        _runError = null;
      });
      _disposeLater(oldImage);
      _disposeLater(oldMask);
      await _runInference();
    } catch (e) {
      if (!mounted) return;
      setState(() => _runError = 'Could not read that image: $e');
    }
  }

  Future<void> _runInference() async {
    final img = _image;
    if (img == null || !_modelReady || _busy) return;
    setState(() => _busy = true);
    try {
      final sw = Stopwatch()..start();
      final pre = await _yolo.preprocess(img);
      final t1 = sw.elapsedMilliseconds;
      final raw = await _yolo.run(pre.tensor);
      final t2 = sw.elapsedMilliseconds;
      final res = await _yolo.postprocess(
        raw,
        pre.letterbox,
        confThreshold: _conf,
      );
      final t3 = sw.elapsedMilliseconds;
      if (!mounted) return;
      final oldMask = _result?.maskImage;
      setState(() {
        _raw = raw;
        _lb = pre.letterbox;
        _result = res;
        _preMs = t1;
        _inferMs = t2 - t1;
        _postMs = t3 - t2;
        _runError = null;
      });
      _disposeLater(oldMask);
    } catch (e) {
      if (!mounted) return;
      setState(() => _runError = '$e');
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  /// Re-decodes detections/masks from the cached raw output (no new inference).
  Future<void> _repost() async {
    final raw = _raw;
    final lb = _lb;
    if (raw == null || lb == null) return;
    final sw = Stopwatch()..start();
    try {
      final res = await _yolo.postprocess(raw, lb, confThreshold: _conf);
      if (!mounted) return;
      final oldMask = _result?.maskImage;
      setState(() {
        _result = res;
        _postMs = sw.elapsedMilliseconds;
      });
      _disposeLater(oldMask);
    } catch (e) {
      if (!mounted) return;
      setState(() => _runError = '$e');
    }
  }

  // ---------------------------------------------------------------- build --

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('YOLO26n-seg · Dashboard Demo001'),
        actions: [
          Padding(
            padding: const EdgeInsets.only(right: 16),
            child: Center(child: _statusChip()),
          ),
        ],
      ),
      body: LayoutBuilder(
        builder: (context, constraints) {
          final wide = constraints.maxWidth >= 960;
          if (wide) {
            return Padding(
              padding: const EdgeInsets.all(16),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  Expanded(child: _canvasCard()),
                  const SizedBox(width: 16),
                  SizedBox(
                    width: 360,
                    child: SingleChildScrollView(child: _panel()),
                  ),
                ],
              ),
            );
          }
          return SingleChildScrollView(
            padding: const EdgeInsets.all(12),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                SizedBox(height: 380, child: _canvasCard()),
                const SizedBox(height: 12),
                _panel(),
              ],
            ),
          );
        },
      ),
    );
  }

  Widget _statusChip() {
    if (_modelError != null) {
      return const Chip(
        avatar: Icon(Icons.error_outline, size: 18),
        label: Text('Model error'),
      );
    }
    if (!_modelReady) {
      return const Chip(
        avatar: SizedBox(
          width: 16,
          height: 16,
          child: CircularProgressIndicator(strokeWidth: 2),
        ),
        label: Text('Loading model…'),
      );
    }
    return Chip(
      avatar: const Icon(Icons.check_circle_outline, size: 18),
      label: Text('Ready · loaded in $_loadMs ms'),
    );
  }

  Widget _canvasCard() {
    final img = _image;
    final Widget body;
    if (img == null) {
      body = Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Icon(Icons.image_search, size: 56),
            const SizedBox(height: 12),
            const Text('Choose an image to run instance segmentation'),
            const SizedBox(height: 16),
            FilledButton.icon(
              onPressed: _modelReady ? _pickImage : null,
              icon: const Icon(Icons.upload_file),
              label: const Text('Choose image'),
            ),
          ],
        ),
      );
    } else {
      body = Stack(
        children: [
          Center(
            child: AspectRatio(
              aspectRatio: img.width / img.height,
              child: CustomPaint(
                painter: ResultPainter(
                  image: img,
                  result: _result,
                  letterbox: _lb,
                  showMasks: _showMasks,
                  showBoxes: _showBoxes,
                ),
              ),
            ),
          ),
          if (_busy)
            const Positioned.fill(
              child: ColoredBox(
                color: Color(0x66000000),
                child: Center(child: CircularProgressIndicator()),
              ),
            ),
        ],
      );
    }
    return Card(
      clipBehavior: Clip.antiAlias,
      child: Padding(padding: const EdgeInsets.all(12), child: body),
    );
  }

  Widget _panel() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        _controlsCard(),
        const SizedBox(height: 12),
        _metricsCard(),
        const SizedBox(height: 12),
        _classesCard(),
        const SizedBox(height: 12),
        _detectionsCard(),
      ],
    );
  }

  Widget _controlsCard() {
    final scheme = Theme.of(context).colorScheme;
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Controls', style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 12),
            FilledButton.icon(
              onPressed: (_modelReady && !_busy) ? _pickImage : null,
              icon: const Icon(Icons.upload_file),
              label: Text(
                  _image == null ? 'Choose image' : 'Choose another image'),
            ),
            const SizedBox(height: 16),
            Text('Confidence threshold: ${_conf.toStringAsFixed(2)}'),
            Slider(
              min: 0.05,
              max: 0.95,
              divisions: 18,
              value: _conf,
              label: _conf.toStringAsFixed(2),
              onChanged: (v) => setState(() => _conf = v),
              onChangeEnd: (_) => _repost(),
            ),
            SwitchListTile(
              dense: true,
              contentPadding: EdgeInsets.zero,
              title: const Text('Show masks'),
              value: _showMasks,
              onChanged: (v) => setState(() => _showMasks = v),
            ),
            SwitchListTile(
              dense: true,
              contentPadding: EdgeInsets.zero,
              title: const Text('Show boxes & labels'),
              value: _showBoxes,
              onChanged: (v) => setState(() => _showBoxes = v),
            ),
            if (_modelError != null) ...[
              const SizedBox(height: 8),
              Text(
                'Could not load web/$_modelUrl.\n'
                'Export the model with export_model.py and put the .onnx file in web/models/.\n\n'
                '$_modelError',
                style: TextStyle(color: scheme.error),
              ),
            ],
            if (_runError != null) ...[
              const SizedBox(height: 8),
              Text(_runError!, style: TextStyle(color: scheme.error)),
            ],
          ],
        ),
      ),
    );
  }

  Widget _metricsCard() {
    final count = _result?.detections.length ?? 0;
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Metrics', style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 12),
            Wrap(
              spacing: 10,
              runSpacing: 10,
              children: [
                _MetricTile(label: 'Detections', value: '$count'),
                _MetricTile(label: 'Inference', value: '$_inferMs ms'),
                _MetricTile(label: 'Preprocess', value: '$_preMs ms'),
                _MetricTile(label: 'Postprocess', value: '$_postMs ms'),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _classesCard() {
    final dets = _result?.detections ?? const <Detection>[];
    final counts = <String, int>{};
    final colors = <String, Color>{};
    for (final d in dets) {
      counts[d.label] = (counts[d.label] ?? 0) + 1;
      colors[d.label] = d.color;
    }
    final entries = counts.entries.toList()
      ..sort((a, b) => b.value.compareTo(a.value));
    final maxCount = entries.isEmpty ? 1 : entries.first.value;

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Classes', style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 12),
            if (entries.isEmpty)
              const Text('No detections yet')
            else
              for (final e in entries)
                Padding(
                  padding: const EdgeInsets.only(bottom: 10),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [Text(e.key), Text('${e.value}')],
                      ),
                      const SizedBox(height: 4),
                      LinearProgressIndicator(
                        value: e.value / maxCount,
                        color: colors[e.key],
                        minHeight: 6,
                        borderRadius: BorderRadius.circular(3),
                      ),
                    ],
                  ),
                ),
          ],
        ),
      ),
    );
  }

  Widget _detectionsCard() {
    final dets = (_result?.detections ?? const <Detection>[]).take(50).toList();
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Detections', style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 8),
            if (dets.isEmpty)
              const Text('Nothing above the threshold')
            else
              for (final d in dets)
                ListTile(
                  dense: true,
                  contentPadding: EdgeInsets.zero,
                  leading: CircleAvatar(radius: 6, backgroundColor: d.color),
                  title: Text(d.label),
                  trailing: Text('${(d.score * 100).toStringAsFixed(1)}%'),
                ),
          ],
        ),
      ),
    );
  }
}

class _MetricTile extends StatelessWidget {
  const _MetricTile({required this.label, required this.value});

  final String label;
  final String value;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Container(
      width: 140,
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: scheme.surfaceContainerHighest,
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(label, style: Theme.of(context).textTheme.labelMedium),
          const SizedBox(height: 4),
          Text(value, style: Theme.of(context).textTheme.titleLarge),
        ],
      ),
    );
  }
}

/// Draws the source image, the mask overlay and the boxes/labels.
class ResultPainter extends CustomPainter {
  ResultPainter({
    required this.image,
    required this.result,
    required this.letterbox,
    required this.showMasks,
    required this.showBoxes,
  });

  final ui.Image image;
  final SegResult? result;
  final Letterbox? letterbox;
  final bool showMasks;
  final bool showBoxes;

  @override
  void paint(Canvas canvas, Size size) {
    final fit = math.min(size.width / image.width, size.height / image.height);
    final dw = image.width * fit;
    final dh = image.height * fit;
    final dst =
        Rect.fromLTWH((size.width - dw) / 2, (size.height - dh) / 2, dw, dh);

    canvas.drawImageRect(
      image,
      Rect.fromLTWH(0, 0, image.width.toDouble(), image.height.toDouble()),
      dst,
      Paint()..filterQuality = FilterQuality.medium,
    );

    final res = result;
    final lb = letterbox;
    if (res == null || lb == null) return;

    // The mask overlay lives in letterboxed space; crop away the padding.
    final mask = res.maskImage;
    if (showMasks && mask != null) {
      final rx = mask.width / kInputSize;
      final ry = mask.height / kInputSize;
      canvas.drawImageRect(
        mask,
        Rect.fromLTWH(lb.padX * rx, lb.padY * ry, lb.newW * rx, lb.newH * ry),
        dst,
        Paint()..filterQuality = FilterQuality.low,
      );
    }

    if (!showBoxes) return;
    for (final d in res.detections) {
      final r = Rect.fromLTRB(
        dst.left + d.box.left * fit,
        dst.top + d.box.top * fit,
        dst.left + d.box.right * fit,
        dst.top + d.box.bottom * fit,
      );
      canvas.drawRect(
        r,
        Paint()
          ..style = PaintingStyle.stroke
          ..strokeWidth = 2
          ..color = d.color,
      );

      final tp = TextPainter(
        text: TextSpan(
          text: '${d.label} ${(d.score * 100).round()}%',
          style: const TextStyle(
            color: Colors.white,
            fontSize: 12,
            fontWeight: FontWeight.w600,
          ),
        ),
        textDirection: TextDirection.ltr,
      )..layout();

      final labelH = tp.height + 4;
      final top = r.top - labelH >= dst.top ? r.top - labelH : r.top;
      final labelRect = Rect.fromLTWH(r.left, top, tp.width + 8, labelH);
      canvas.drawRect(labelRect, Paint()..color = d.color);
      tp.paint(canvas, Offset(labelRect.left + 4, labelRect.top + 2));
    }
  }

  @override
  bool shouldRepaint(ResultPainter old) =>
      old.image != image ||
      old.result != result ||
      old.letterbox != letterbox ||
      old.showMasks != showMasks ||
      old.showBoxes != showBoxes;
}
