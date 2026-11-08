import 'dart:async';
import 'dart:js_interop';
import 'dart:typed_data';

import 'package:web/web.dart' as web;

/// Opens the browser's file chooser and returns the bytes of the chosen image.
/// Never completes if the user cancels the dialog.
Future<Uint8List?> pickImageBytes() {
  final completer = Completer<Uint8List?>();
  final input = web.document.createElement('input') as web.HTMLInputElement
    ..type = 'file'
    ..accept = 'image/*'
    ..style.display = 'none';

  input.onChange.listen((_) async {
    final file = input.files?.item(0);
    input.remove();
    if (file == null) {
      completer.complete(null);
      return;
    }
    final buffer = await file.arrayBuffer().toDart;
    completer.complete(buffer.toDart.asUint8List());
  });

  web.document.body?.append(input);
  input.click();
  return completer.future;
}
