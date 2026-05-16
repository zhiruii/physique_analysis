/**
 * Web Worker: ONNX Runtime Web 1.14.0 inference — stable WASM-only API.
 * Model input:  'input_layer'  [1, 224, 224, 3]  float32
 * Model output: 'output_0'     [1, 3]            float32
 * Class order (alphabetical = Keras training order):
 *   index 0 = advanced, 1 = beginner, 2 = intermediate
 */
import * as ort from 'onnxruntime-web'

let session = null

self.onmessage = async ({ data: msg }) => {
  if (msg.type === 'init') {
    try {
      self.postMessage({ type: 'log', text: 'Loading ONNX model...' })

      // Point to locally-served WASM files (version-matched, no CDN)
      ort.env.wasm.wasmPaths = '/ort/'
      ort.env.wasm.numThreads = 1

      session = await ort.InferenceSession.create(msg.modelUrl, {
        executionProviders: ['wasm'],
      })

      self.postMessage({ type: 'log', text: 'Model ready' })
      self.postMessage({ type: 'ready' })

    } catch (err) {
      self.postMessage({ type: 'error', message: err.message })
      self.postMessage({ type: 'log', text: `ERR: ${err.message}` })
      console.error('[worker] init error:', err)
    }

  } else if (msg.type === 'predict') {
    if (!session) return

    try {
      const { pixels, width, height } = msg

      // RGBA → RGB float32 normalised to [0,1], shape [1, 224, 224, 3]
      const float32 = new Float32Array(224 * 224 * 3)
      const rgba = new Uint8Array(pixels.buffer)
      for (let i = 0; i < 224 * 224; i++) {
        float32[i * 3]     = rgba[i * 4]     / 255
        float32[i * 3 + 1] = rgba[i * 4 + 1] / 255
        float32[i * 3 + 2] = rgba[i * 4 + 2] / 255
      }

      const tensor = new ort.Tensor('float32', float32, [1, 224, 224, 3])
      const results = await session.run({ input_layer: tensor })
      const probs = Array.from(results['output_0'].data)

      self.postMessage({ type: 'prediction', probs })
    } catch (err) {
      console.error('[worker] predict error:', err)
      self.postMessage({ type: 'predict_error', message: err.message })
    }
  }
}
