/**
 * Web Worker: ONNX Runtime Web inference.
 * Model input:  'input'   [1, 3, 224, 224]  float32  (NCHW, ImageNet normalised)
 * Model output: 'output'  [1, 3]            float32
 * Class order (alphabetical = PyTorch ImageFolder order):
 *   index 0 = advanced, 1 = beginner, 2 = intermediate
 */
import * as ort from 'onnxruntime-web'

let session = null

self.onmessage = async ({ data: msg }) => {
  if (msg.type === 'init') {
    try {
      self.postMessage({ type: 'log', text: 'Loading ONNX model...' })

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
      const { pixels } = msg
      const rgba = new Uint8Array(pixels.buffer)
      const numPixels = 224 * 224

      // RGBA → RGB, ImageNet normalisation, NCHW layout [1, 3, 224, 224]
      const float32 = new Float32Array(3 * numPixels)
      const mean = [0.485, 0.456, 0.406]
      const std  = [0.229, 0.224, 0.225]

      for (let i = 0; i < numPixels; i++) {
        float32[0 * numPixels + i] = (rgba[i * 4]     / 255 - mean[0]) / std[0]
        float32[1 * numPixels + i] = (rgba[i * 4 + 1] / 255 - mean[1]) / std[1]
        float32[2 * numPixels + i] = (rgba[i * 4 + 2] / 255 - mean[2]) / std[2]
      }

      const tensor = new ort.Tensor('float32', float32, [1, 3, 224, 224])
      const results = await session.run({ input: tensor })
      const logits = Array.from(results['output'].data)

      // model outputs raw logits — apply softmax to get probabilities
      const maxLogit = Math.max(...logits)
      const exps = logits.map(l => Math.exp(l - maxLogit))
      const expSum = exps.reduce((a, b) => a + b, 0)
      const probs = exps.map(e => e / expSum)

      self.postMessage({ type: 'prediction', probs })
    } catch (err) {
      console.error('[worker] predict error:', err)
      self.postMessage({ type: 'predict_error', message: err.message })
    }
  }
}
