import {
  FaceLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const overlay = document.getElementById("overlay");
const hud = document.getElementById("hud");
const statusEl = document.getElementById("status");
const startBtn = document.getElementById("start-button");
const ctx = canvas.getContext("2d");

const WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

const MAX_FACES = 4;

// Physics constants, expressed in units of the eyeball radius R so the
// behavior is identical at any face size or camera distance.
const GRAVITY = 16; // R / s², pulling the pupil toward the "low" side of the head
const INERTIA_GAIN = 0.22; // how strongly head acceleration shoves the pupil
const INERTIA_MAX = 55; // R / s², cap on the inertial force
const LINEAR_DAMPING = 1.8; // 1/s, air drag on the pupil
const RESTITUTION = 0.38; // bounce energy kept when hitting the socket wall
const PUPIL_RATIO = 0.56; // pupil radius / socket radius
const MAX_ACCEL = 9000; // px / s², clamp on estimated head acceleration

// Face landmarks (478-point model): eye corners, lids, iris centers.
const LEFT_EYE = { outer: 33, inner: 133, top: 159, bottom: 145, iris: 468 };
const RIGHT_EYE = { outer: 263, inner: 362, top: 386, bottom: 374, iris: 473 };
const ROLL_FROM = 33; // left eye outer corner
const ROLL_TO = 263; // right eye outer corner

let landmarker = null;
let running = false;
let lastVideoTime = -1;
let lastT = 0;

// Per-face physics state, keyed by detection index (0..MAX_FACES-1).
// Each eye: pupil offset/velocity, smoothed socket geometry, head-motion estimators.
const faceStates = new Map();

function makeEye() {
  return {
    px: 0,
    py: 0, // pupil offset from socket center, px (relative frame)
    vx: 0,
    vy: 0, // pupil velocity, px/s
    cx: null, // smoothed socket center, css px
    cy: null,
    r: 0, // smoothed socket radius, css px
    roll: 0, // smoothed head roll, rad
    prevCx: null, // raw center last step, for velocity estimation
    prevCy: null,
    evx: 0, // smoothed socket velocity
    evy: 0,
    prevEvx: 0,
    prevEvy: 0,
  };
}

function setStatus(message) {
  statusEl.textContent = message || "";
}

function ensureFaceState(i) {
  let s = faceStates.get(i);
  if (!s) {
    s = { left: makeEye(), right: makeEye() };
    faceStates.set(i, s);
  }
  return s;
}

// MediaPipe gives normalized coords relative to the video frame; the video is
// drawn with object-fit: cover and mirrored, so map into display space here.
function mapPoint(nx, ny) {
  const vw = video.videoWidth || 1;
  const vh = video.videoHeight || 1;
  const dw = canvas.clientWidth;
  const dh = canvas.clientHeight;
  const scale = Math.max(dw / vw, dh / vh);
  const ox = (dw - vw * scale) / 2;
  const oy = (dh - vh * scale) / 2;
  return { x: ox + (1 - nx) * vw * scale, y: oy + ny * vh * scale };
}

function dist2(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.hypot(dx, dy);
}

// Socket geometry (center, radius) for one eye, in display space.
function eyeGeometry(lm, spec) {
  const outer = mapPoint(lm[spec.outer].x, lm[spec.outer].y);
  const inner = mapPoint(lm[spec.inner].x, lm[spec.inner].y);
  const top = mapPoint(lm[spec.top].x, lm[spec.top].y);
  const bottom = mapPoint(lm[spec.bottom].x, lm[spec.bottom].y);
  const iris = mapPoint(lm[spec.iris].x, lm[spec.iris].y);
  const midX = (outer.x + inner.x) / 2;
  const midY = (outer.y + inner.y) / 2;
  return {
    cx: midX + (iris.x - midX) * 0.5,
    cy: midY + (iris.y - midY) * 0.5,
    r: Math.max(dist2(outer, inner) / 2 * 1.25, dist2(top, bottom) / 2 * 1.15),
  };
}

// One physics step for a pupil inside its socket. Pupil position is tracked in
// a socket-relative frame; the socket carries it around as the face moves.
function stepEye(eye, geo, roll, dt) {
  // Smooth the socket so detection jitter doesn't teleport the eyeball.
  if (eye.cx === null) {
    eye.cx = geo.cx;
    eye.cy = geo.cy;
    eye.r = geo.r;
    eye.roll = roll;
    eye.prevCx = geo.cx;
    eye.prevCy = geo.cy;
  } else {
    const k = 0.5;
    eye.cx += (geo.cx - eye.cx) * k;
    eye.cy += (geo.cy - eye.cy) * k;
    eye.r += (geo.r - eye.r) * k;
    let d = roll - eye.roll;
    d = Math.atan2(Math.sin(d), Math.cos(d));
    eye.roll += d * k;
  }

  const R = Math.max(eye.r, 2);

  // Head motion -> inertial force on the pupil (things that lag behind).
  if (eye.prevCx !== null && dt > 0) {
    const ivx = (geo.cx - eye.prevCx) / dt;
    const ivy = (geo.cy - eye.prevCy) / dt;
    eye.evx += (ivx - eye.evx) * 0.5;
    eye.evy += (ivy - eye.evy) * 0.5;
  }
  eye.prevCx = geo.cx;
  eye.prevCy = geo.cy;

  let fx = 0;
  let fy = 0;
  if (dt > 0) {
    const ax = clamp((eye.evx - eye.prevEvx) / dt, MAX_ACCEL);
    const ay = clamp((eye.evy - eye.prevEvy) / dt, MAX_ACCEL);
    fx = clamp(-ax * INERTIA_GAIN, INERTIA_MAX * R);
    fy = clamp(-ay * INERTIA_GAIN, INERTIA_MAX * R);
  }
  eye.prevEvx = eye.evx;
  eye.prevEvy = eye.evy;

  // Gravity in the head's local frame, so tilting the head rolls the "down"
  // direction: the pupil settles toward the low eye, like real googly eyes.
  const g = GRAVITY * R;
  fx += Math.sin(eye.roll) * g;
  fy += Math.cos(eye.roll) * g;

  // Integrate with drag.
  const damp = Math.exp(-LINEAR_DAMPING * dt);
  eye.vx = eye.vx * damp + fx * dt;
  eye.vy = eye.vy * damp + fy * dt;
  eye.px += eye.vx * dt;
  eye.py += eye.vy * dt;

  // Constrain the pupil inside the socket; bounce off the wall.
  const maxLen = Math.max(R * (1 - PUPIL_RATIO), 0.5);
  const len = Math.hypot(eye.px, eye.py);
  if (len > maxLen) {
    const nx = eye.px / len;
    const ny = eye.py / len;
    eye.px = nx * maxLen;
    eye.py = ny * maxLen;
    const vn = eye.vx * nx + eye.vy * ny;
    if (vn > 0) {
      eye.vx -= (1 + RESTITUTION) * vn * nx;
      eye.vy -= (1 + RESTITUTION) * vn * ny;
      eye.vx *= 0.94; // tangential friction
      eye.vy *= 0.94;
    }
  }

  // Let it come to rest instead of micro-jittering at the bottom.
  if (Math.hypot(eye.vx, eye.vy) < R * 0.02) {
    eye.vx = 0;
    eye.vy = 0;
  }
}

function clamp(v, limit) {
  return Math.max(-limit, Math.min(limit, v));
}

function processFaces(result, dt) {
  const faces = result && result.faceLandmarks ? result.faceLandmarks : [];
  hud.hidden = faces.length > 0;
  if (faces.length > 0) {
    hud.textContent =
      faces.length === 1 ? "1 face found" : `${faces.length} faces found`;
  }

  for (let i = 0; i < Math.min(faces.length, MAX_FACES); i++) {
    const lm = faces[i];
    const state = ensureFaceState(i);
    const left = eyeGeometry(lm, LEFT_EYE);
    const right = eyeGeometry(lm, RIGHT_EYE);
    // Googly eyes come in pairs: give both eyes the average radius.
    const pairR = (left.r + right.r) / 2;
    left.r = pairR;
    right.r = pairR;
    const a = mapPoint(lm[ROLL_FROM].x, lm[ROLL_FROM].y);
    const b = mapPoint(lm[ROLL_TO].x, lm[ROLL_TO].y);
    const roll = Math.atan2(b.y - a.y, b.x - a.x);
    stepEye(state.left, left, roll, dt);
    stepEye(state.right, right, roll, dt);
  }
}

function draw() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.round(canvas.clientWidth * dpr);
  const h = Math.round(canvas.clientHeight * dpr);
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);

  for (const state of faceStates.values()) {
    drawEye(state.left);
    drawEye(state.right);
  }
}

function drawEye(eye) {
  if (eye.cx === null || eye.r <= 0) return;
  const R = eye.r;
  const px = eye.cx + eye.px;
  const py = eye.cy + eye.py;
  const pupilR = R * PUPIL_RATIO;

  // Sclera.
  const socket = ctx.createRadialGradient(
    eye.cx - R * 0.25,
    eye.cy - R * 0.3,
    R * 0.1,
    eye.cx,
    eye.cy,
    R
  );
  socket.addColorStop(0, "#ffffff");
  socket.addColorStop(1, "#dfe1e8");
  ctx.beginPath();
  ctx.arc(eye.cx, eye.cy, R, 0, Math.PI * 2);
  ctx.fillStyle = socket;
  ctx.fill();
  ctx.lineWidth = Math.max(R * 0.07, 1.5);
  ctx.strokeStyle = "rgba(15, 17, 23, 0.55)";
  ctx.stroke();

  // Pupil.
  ctx.beginPath();
  ctx.arc(px, py, pupilR, 0, Math.PI * 2);
  const grad = ctx.createRadialGradient(
    px - pupilR * 0.3,
    py - pupilR * 0.35,
    pupilR * 0.1,
    px,
    py,
    pupilR
  );
  grad.addColorStop(0, "#3c3f47");
  grad.addColorStop(0.6, "#111318");
  grad.addColorStop(1, "#000000");
  ctx.fillStyle = grad;
  ctx.fill();

  // Glint.
  ctx.beginPath();
  ctx.arc(
    px - pupilR * 0.32,
    py - pupilR * 0.36,
    pupilR * 0.2,
    0,
    Math.PI * 2
  );
  ctx.fillStyle = "rgba(255, 255, 255, 0.85)";
  ctx.fill();
}

function frame(now) {
  if (!running) return;
  const dt = Math.min(Math.max((now - lastT) / 1000, 0.001), 0.05);
  lastT = now;

  if (landmarker && video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    try {
      const result = landmarker.detectForVideo(video, now);
      processFaces(result, dt);
    } catch (err) {
      console.error("Detection failed:", err);
    }
  }
  draw();
  requestAnimationFrame(frame);
}

// The camera prompt needs a user gesture, but loading the model does not —
// so fetch + compile it the moment the page opens and reuse it on start.
let modelPromise = null;

function loadModel() {
  if (!modelPromise) {
    modelPromise = (async () => {
      const fileset = await FilesetResolver.forVisionTasks(WASM_URL);
      const createWith = (delegate) =>
        FaceLandmarker.createFromOptions(fileset, {
          baseOptions: { modelAssetPath: MODEL_URL, delegate },
          runningMode: "VIDEO",
          numFaces: MAX_FACES,
        });
      // GPU is faster, but some environments have WebGL disabled — fall back
      // to the slower CPU delegate rather than failing outright.
      return createWith("GPU").catch(() => createWith("CPU"));
    })().catch((err) => {
      modelPromise = null; // allow a retry on the next start attempt
      throw err;
    });
  }
  return modelPromise;
}

function preloadModel() {
  setStatus("Loading face detection model…");
  loadModel().then(
    () => setStatus("Model ready — tap Start when you are"),
    (err) => {
      console.error("Model preload failed:", err);
      setStatus("Couldn't preload the model; it will retry when you start");
    }
  );
}

async function start() {
  startBtn.disabled = true;
  setStatus("Requesting camera…");

  // Track each half so the status always shows which stage is pending — if
  // something stalls (ignored prompt, slow download) it's visible.
  let cameraReady = false;
  let modelReady = false;
  const updateStage = () => {
    if (cameraReady && modelReady) setStatus("Starting…");
    else if (cameraReady) setStatus("Camera ready — finishing model load…");
    else if (modelReady) setStatus("Model ready — waiting for camera…");
  };

  const cameraPromise = navigator.mediaDevices
    .getUserMedia({
      video: {
        facingMode: "user",
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
      audio: false,
    })
    .then((stream) => {
      cameraReady = true;
      updateStage();
      return stream;
    });

  const modelWhenReady = loadModel().then((landmarker) => {
    modelReady = true;
    updateStage();
    return landmarker;
  });

  // Some environments never settle the camera request (in-app browsers that
  // can't show a permission prompt, or a camera held by another app) — time
  // out with a useful message instead of hanging forever. Generous, because
  // the user's decision time on the permission prompt counts here too.
  const CAMERA_TIMEOUT_MS = 60000;
  const cameraWithTimeout = Promise.race([
    cameraPromise,
    new Promise((_, reject) =>
      setTimeout(() => {
        const err = new Error("camera request timed out");
        err.name = "CameraTimeoutError";
        reject(err);
      }, CAMERA_TIMEOUT_MS)
    ),
  ]);

  const modelSlowTimer = setTimeout(() => {
    if (!modelReady) {
      setStatus(
        "Still loading the face detection model — big download. If it never finishes, reload the page."
      );
    }
  }, 20000);

  try {
    const [stream, landmarkerResult] = await Promise.all([
      cameraWithTimeout,
      modelWhenReady,
    ]);
    clearTimeout(modelSlowTimer);

    landmarker = landmarkerResult;
    video.srcObject = stream;

    // Un-hide before play(): iOS won't render a display:none video, and don't
    // block on play() — some browsers never resolve it, and the frame loop
    // already waits for frames to arrive on its own.
    video.hidden = false;
    canvas.hidden = false;
    overlay.hidden = true;
    video.play().catch((err) => console.error("video.play() failed:", err));

    hud.hidden = false;
    hud.textContent = "Looking for faces…";
    running = true;
    lastVideoTime = -1;
    lastT = performance.now();
    requestAnimationFrame(frame);
  } catch (err) {
    clearTimeout(modelSlowTimer);
    console.error(err);
    startBtn.disabled = false;
    if (err && err.name === "CameraTimeoutError") {
      setStatus(
        "Timed out waiting for the camera. In-app/embedded browsers often can't show the permission prompt — try opening this page in Safari or Chrome. (Is another app using the camera?)"
      );
    } else if (
      err &&
      (err.name === "NotAllowedError" || err.name === "PermissionDeniedError")
    ) {
      setStatus(
        "Camera access was denied. Allow it in your browser settings and try again."
      );
    } else if (err && err.name === "NotFoundError") {
      setStatus("No camera was found on this device.");
    } else {
      setStatus(`Could not start: ${err && err.message ? err.message : err}`);
    }
  }
}

if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
  setStatus(
    "Camera access needs a secure (HTTPS or localhost) page. Open this site over HTTPS."
  );
  startBtn.disabled = true;
} else {
  preloadModel();
  startBtn.addEventListener("click", start);
}
