import { useRef, useEffect, useState, useCallback } from 'react'
import { PoseLandmarker, FilesetResolver } from '@mediapipe/tasks-vision'
import './App.css'

const CLASS_LABELS    = ['advanced', 'beginner', 'intermediate']
const CLASS_MIDPOINTS = { advanced: 95, beginner: 24, intermediate: 71 }

function computeScore(probs) {
  return Math.round(
    CLASS_LABELS.reduce((sum, cls, i) => sum + probs[i] * CLASS_MIDPOINTS[cls], 0)
  )
}

const BODY_CONNECTIONS = [
  [11,12], [11,13],[13,15], [12,14],[14,16],
  [11,23],[12,24],[23,24],
  [23,25],[25,27],[27,29],[27,31],[29,31],
  [24,26],[26,28],[28,30],[28,32],[30,32],
]
const BODY_IDS = [11,12,13,14,15,16,23,24,25,26,27,28,29,30,31,32]

// Landmarks used for upper body crop — matches normalize_dataset.py
const CROP_LM_IDS = [11, 12, 13, 14, 15, 16, 23, 24]
const PAD_TOP     = 0.15
const PAD_BOTTOM  = 0.05
const PAD_SIDES   = 0.15

function drawSkeleton(ctx, landmarks, w, h) {
  if (!landmarks?.length) return
  const lms = landmarks[0]
  ctx.shadowBlur = 6
  ctx.strokeStyle = '#00ff41'; ctx.lineWidth = 2; ctx.shadowColor = '#00ff41'
  for (const [a,b] of BODY_CONNECTIONS) {
    if ((lms[a]?.visibility??0) > 0.5 && (lms[b]?.visibility??0) > 0.5) {
      ctx.beginPath()
      ctx.moveTo(lms[a].x*w, lms[a].y*h)
      ctx.lineTo(lms[b].x*w, lms[b].y*h)
      ctx.stroke()
    }
  }
  ctx.fillStyle = '#ffff00'; ctx.shadowColor = '#ffff00'
  for (const i of BODY_IDS) {
    if ((lms[i]?.visibility??0) > 0.5) {
      ctx.beginPath()
      ctx.arc(lms[i].x*w, lms[i].y*h, 4, 0, Math.PI*2)
      ctx.fill()
    }
  }
  ctx.shadowBlur = 0
}

export default function App() {
  const videoRef    = useRef(null)
  const canvasRef   = useRef(null)
  const captureRef  = useRef(null)   // hidden 224×224 canvas for worker frames
  const landmarkerRef = useRef(null)
  const workerRef      = useRef(null)
  const rafRef         = useRef(null)
  const pendingRef     = useRef(false)   // true while worker is processing a frame
  const modelReadyRef  = useRef(false)   // set true when worker posts 'ready'

  const [log, setLog]             = useState(['Initialising...'])
  const [score, setScore]         = useState(null)
  const [label, setLabel]         = useState(null)
  const [poseVisible, setPoseVisible] = useState(false)
  const [cameraReady, setCameraReady] = useState(false)
  const [modelReady, setModelReady]   = useState(false)
  const [initError, setInitError]     = useState(null)

  const addLog = useCallback((msg) => {
    setLog(prev => [...prev.slice(-3), msg])
  }, [])

  useEffect(() => {
    let cancelled = false

    // ── Worker ─────────────────────────────────────────────
    const worker = new Worker(new URL('./tfWorker.js', import.meta.url), { type: 'module' })
    workerRef.current = worker

    worker.onmessage = ({ data: msg }) => {
      if (cancelled) return
      if (msg.type === 'ready') {
        setModelReady(true)
        modelReadyRef.current = true
        addLog('Model ready — analysing')
      } else if (msg.type === 'log') {
        addLog(msg.text)
      } else if (msg.type === 'error') {
        console.error('Worker model error:', msg.message)
        setInitError(`Model: ${msg.message}`)
        addLog('Model load failed')
      } else if (msg.type === 'predict_error') {
        console.error('Worker predict error:', msg.message)
        pendingRef.current = false
      } else if (msg.type === 'prediction') {
        pendingRef.current = false
        const arr = msg.probs
        setScore(computeScore(arr))
        setLabel(CLASS_LABELS[arr.indexOf(Math.max(...arr))])
      }
    }

    // ── Init sequence ───────────────────────────────────────
    async function init() {
      // 1. Camera
      addLog('Requesting camera...')
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
          audio: false,
        })
        if (cancelled) { stream.getTracks().forEach(t => t.stop()); return }
        const video = videoRef.current
        video.srcObject = stream
        await new Promise((res, rej) => { video.onloadedmetadata = res; video.onerror = rej })
        await video.play()
        setCameraReady(true)
        addLog('Camera active')
      } catch (err) {
        console.error('Camera:', err)
        setInitError('Camera permission denied.')
        addLog('Camera denied')
        return
      }

      // 2. Pose detector
      addLog('Loading pose detector...')
      try {
        const vision = await FilesetResolver.forVisionTasks(
          'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
        )
        if (cancelled) return
        const lm = await PoseLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
            delegate: 'GPU',
          },
          runningMode: 'VIDEO',
          numPoses: 1,
          minPoseDetectionConfidence: 0.5,
          minPosePresenceConfidence: 0.5,
          minTrackingConfidence: 0.5,
        })
        if (cancelled) return
        landmarkerRef.current = lm
        addLog('Pose detector ready')
      } catch (err) {
        console.error('MediaPipe:', err)
        addLog('Pose detector failed')
        return
      }

      // 3. Kick off model loading in the worker (non-blocking — worker runs on its own thread)
      addLog('Loading model...')
      const modelUrl = window.location.origin + '/model/model.onnx'
      worker.postMessage({ type: 'init', modelUrl })

      // 4. Render loop
      let lastVideoTime = -1

      function loop() {
        if (cancelled) return
        const video   = videoRef.current
        const canvas  = canvasRef.current
        const capture = captureRef.current
        if (!video || !canvas || !capture || video.readyState < 2 || video.paused) {
          rafRef.current = requestAnimationFrame(loop)
          return
        }

        const w = video.videoWidth, h = video.videoHeight
        if (canvas.width !== w)  canvas.width  = w
        if (canvas.height !== h) canvas.height = h

        const ctx = canvas.getContext('2d')
        ctx.clearRect(0, 0, w, h)

        if (video.currentTime === lastVideoTime) {
          rafRef.current = requestAnimationFrame(loop)
          return
        }
        lastVideoTime = video.currentTime

        // Pose detection (main thread, WebGL)
        const result = landmarkerRef.current.detectForVideo(video, performance.now())
        drawSkeleton(ctx, result.landmarks, w, h)

        const upperBody =
          result.landmarks.length > 0 &&
          (result.landmarks[0][11]?.visibility ?? 0) > 0.6 &&
          (result.landmarks[0][12]?.visibility ?? 0) > 0.6

        setPoseVisible(upperBody)

        // Send cropped upper body frame to worker for inference
        if (upperBody && modelReadyRef.current && !pendingRef.current) {
          pendingRef.current = true

          const lms = result.landmarks[0]
          const visible = CROP_LM_IDS.filter(i => (lms[i]?.visibility ?? 0) >= 0.3)
          const xs = visible.map(i => lms[i].x)
          const ys = visible.map(i => lms[i].y)

          let xMin = Math.min(...xs), xMax = Math.max(...xs)
          let yMin = Math.min(...ys), yMax = Math.max(...ys)
          const cropW = xMax - xMin
          const cropH = yMax - yMin

          xMin -= cropW * PAD_SIDES
          xMax += cropW * PAD_SIDES
          yMin -= cropH * PAD_TOP
          yMax += cropH * PAD_BOTTOM

          // Force square crop — matches normalize_dataset.py
          const cx   = (xMin + xMax) / 2
          const cy   = (yMin + yMax) / 2
          const half = Math.max(xMax - xMin, yMax - yMin) / 2
          xMin = Math.max(0, cx - half)
          xMax = Math.min(1, cx + half)
          yMin = Math.max(0, cy - half)
          yMax = Math.min(1, cy + half)

          const sx = xMin * w, sy = yMin * h
          const sw = (xMax - xMin) * w, sh = (yMax - yMin) * h

          const capCtx = capture.getContext('2d')
          capCtx.drawImage(video, sx, sy, sw, sh, 0, 0, 224, 224)
          const imageData = capCtx.getImageData(0, 0, 224, 224)
          worker.postMessage(
            { type: 'predict', pixels: imageData.data, width: 224, height: 224 },
            [imageData.data.buffer]
          )
        }

        rafRef.current = requestAnimationFrame(loop)
      }

      rafRef.current = requestAnimationFrame(loop)
    }

    init()

    return () => {
      cancelled = true
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
      const video = videoRef.current
      if (video?.srcObject) video.srcObject.getTracks().forEach(t => t.stop())
      worker.terminate()
    }
  }, [addLog])


  const scorePercent  = score ?? 0

  function getScoreLabel(s) {
    if (s === null) return 'Awaiting'
    if (s >= 90) return 'Pro Competitor'
    if (s >= 76) return 'Sponsored Athlete'
    if (s >= 55) return 'Competitive Amateur'
    if (s >= 35) return 'Consistent'
    return 'Beginner'
  }

  const labelDisplay = getScoreLabel(score)

  return (
    <div className="pheno-app">
      <header className="pheno-header">
        <div className="logo">PHENO</div>
        <a
          className="feedback-btn"
          href="https://docs.google.com/forms/d/e/1FAIpQLSc-0Pz0elrQJJFe4RhK8ZZ5NoiWikT8lnJzQ9tvlqt6VyYhww/viewform?usp=dialog"
          target="_blank"
          rel="noopener noreferrer"
        >
          We would love your feedback
        </a>
      </header>

      <div className="privacy-banner">
        &#x25B8; YOUR CAMERA FEED IS PROCESSED ENTIRELY ON THIS DEVICE — NO DATA IS EVER UPLOADED OR STORED
      </div>

      <div className="pheno-layout">
        <div className="hero-col">
          <h1 className="hero-title">PHENO<br />Physique<br />Analysis.</h1>
          <p className="hero-sub">Real-time physique classification.<br />All processing local. Nothing transmitted.</p>

          <div className="system-log">
            <div className="log-title">SYSTEM LOG</div>
            <ul>
              {log.map((entry, i) => (
                <li key={i}><span className="log-bullet">&#x25B6;</span> {entry}</li>
              ))}
            </ul>
          </div>
        </div>

        <div className="content-col">
          <div className="camera-score-row">
            <section className="analysis-window">
              <div className="window-header">
                Video Analysis Window &mdash; <span className="live-tag">[Live]</span>
              </div>
              <div className="viewport">
                <div className="grid-overlay" />
                <video ref={videoRef} className="camera-feed" playsInline muted />
                <canvas ref={canvasRef} className="skeleton-canvas" />
                <canvas ref={captureRef} width={224} height={224} style={{ display: 'none' }} />

                {initError && <div className="error-overlay">{initError}</div>}

                <div className="frame-hint">
                  {!cameraReady
                    ? 'Waiting for camera permission...'
                    : !landmarkerRef.current
                    ? 'Loading pose detector...'
                    : poseVisible
                    ? 'Upper body locked — analysing'
                    : 'Position upper body within frame'}
                </div>
              </div>
            </section>

            <div className="score-panel">
              <div className="score-card">
                <div className="card-label">SCORE</div>
                <div className={`main-score ${score !== null ? 'scored' : ''}`}>
                  {score !== null ? score : '--'}
                </div>
                <div className="score-label">{labelDisplay}</div>
                <p className="score-footer">Real-time physique classification</p>
              </div>

              <div className="ref-scale">
                <div className="ref-title">SCORE REFERENCE</div>
                <ul className="ref-list">
                  <li><span className="ref-range">90+</span><span className="ref-desc">Pro competitor</span></li>
                  <li><span className="ref-range">76–90</span><span className="ref-desc">Sponsored athlete</span></li>
                  <li><span className="ref-range">55–75</span><span className="ref-desc">Competitive amateur</span></li>
                  <li><span className="ref-range">35–54</span><span className="ref-desc">Consistent</span></li>
                  <li><span className="ref-range">20–34</span><span className="ref-desc">Beginner</span></li>
                </ul>
              </div>

              <div className="metrics-bar">
                <div className="stat-item">
                  <span className="stat-label">Overall Score</span>
                  <div className="bar-track">
                    <div className="bar-fill" style={{ width: `${scorePercent}%` }} />
                  </div>
                  <span className="stat-value">{score ?? '--'}</span>
                </div>
                <div className="stat-item">
                  <span className="stat-label">Classification</span>
                  <div className="bar-track">
                    <div className="bar-fill" style={{ width: score === null ? '0%' : `${score}%` }} />
                  </div>
                  <span className="stat-value">{labelDisplay}</span>
                </div>
                <div className="stat-item">
                  <span className="stat-label">Body Detected</span>
                  <div className="bar-track">
                    <div className="bar-fill" style={{ width: poseVisible ? '100%' : '0%' }} />
                  </div>
                  <span className="stat-value">{poseVisible ? 'YES' : 'NO'}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
