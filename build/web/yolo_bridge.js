// Thin wrapper around onnxruntime-web so Dart only has to call two functions.
// All pre/post-processing lives in Dart (lib/yolo26_seg.dart).
window.yoloBridge = {
  session: null,

  // Loads the ONNX model. Resolves with a short "inputs -> outputs" description.
  async load(url) {
    ort.env.wasm.wasmPaths =
      'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';
    this.session = await ort.InferenceSession.create(url, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });
    return (
      this.session.inputNames.join(',') + ' -> ' +
      this.session.outputNames.join(',')
    );
  },

  // data: Float32Array of length 3*size*size (CHW, RGB, 0..1).
  // Returns the predictions tensor (rank 3) and the mask prototypes (rank 4).
  async run(data, size) {
    if (!this.session) throw new Error('Model not loaded');
    const input = new ort.Tensor('float32', data, [1, 3, size, size]);
    const feeds = {};
    feeds[this.session.inputNames[0]] = input;
    const results = await this.session.run(feeds);

    let pred = null;
    let proto = null;
    for (const name of this.session.outputNames) {
      const t = results[name];
      if (t.dims.length === 3) pred = t;
      else if (t.dims.length === 4) proto = t;
    }
    if (!pred || !proto) {
      const shapes = this.session.outputNames
        .map((n) => n + '[' + results[n].dims.join('x') + ']')
        .join(', ');
      throw new Error('Unexpected model outputs: ' + shapes);
    }
    return {
      predData: pred.data,
      predDims: Array.from(pred.dims, Number),
      protoData: proto.data,
      protoDims: Array.from(proto.dims, Number),
    };
  },
};
