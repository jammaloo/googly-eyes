import {
  FaceLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

const video = document.getElementById("video");
const videoBg = document.getElementById("video-bg");
const canvas = document.getElementById("canvas");
const overlay = document.getElementById("overlay");
const hud = document.getElementById("hud");
const statusEl = document.getElementById("status");
const startBtn = document.getElementById("start-button");
const settingsBtn = document.getElementById("settings-button");
const settingsPanel = document.getElementById("settings-panel");
const settingsRows = document.getElementById("settings-rows");
const settingsReset = document.getElementById("settings-reset");
const ctx = canvas.getContext("2d");

const WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

const MAX_FACES = 4;

// Googly-eye physics: each pupil is a ball simulated in screen space inside a
// circular socket that rides on the detected eye. Gravity (scaled by eye size,
// so every eye wobbles at the same tempo — a real ~4 Hz googly swing) pulls
// the pupil to the bottom, and the rattle comes from the pupil bouncing off
// the MOVING socket wall as the face moves. No acceleration estimation.
// The knobs live in P (mutable, tweakable live from the settings panel).
const PHYS_DEFAULTS = {
  GRAVITY: 260, // px/s² per px of socket radius
  PUPIL_RATIO: 0.56, // pupil radius / socket radius
  RESTITUTION: 0.42, // energy kept when the pupil bounces off the wall
  BOUNCE_MIN: 3, // R/s; contacts slower than this rest instead of bounce
  WALL_FRICTION: 0.985, // tangential velocity kept while touching
  AIR_DAMPING: 0.4, // 1/s velocity decay in flight
  SOCKET_TRACK: 45, // 1/s rate the socket chases the detected eye
};
const P = { ...PHYS_DEFAULTS };
const SUBSTEP = 1 / 120; // fixed physics step, runs inside every frame
const MAX_SUBSTEPS = 8;

// Which knobs appear in the settings panel, and their slider ranges.
const SETTINGS_SCHEMA = [
  {
    key: "GRAVITY",
    label: "Gravity",
    hint: "How hard pupils are pulled to the bottom",
    min: 50,
    max: 600,
    step: 5,
    fmt: (v) => `${Math.round(v)}`,
  },
  {
    key: "RESTITUTION",
    label: "Bounciness",
    hint: "Energy kept when the pupil hits the wall",
    min: 0,
    max: 0.9,
    step: 0.01,
    fmt: (v) => v.toFixed(2),
  },
  {
    key: "AIR_DAMPING",
    label: "Swing decay",
    hint: "Higher stops the swinging sooner",
    min: 0,
    max: 2,
    step: 0.05,
    fmt: (v) => v.toFixed(2),
  },
  {
    key: "SOCKET_TRACK",
    label: "Motion response",
    hint: "How sharply face movement flings the pupil",
    min: 10,
    max: 90,
    step: 1,
    fmt: (v) => `${Math.round(v)}`,
  },
  {
    key: "WALL_FRICTION",
    label: "Wall friction",
    hint: "Grip between pupil and socket wall",
    min: 0.9,
    max: 1,
    step: 0.001,
    fmt: (v) => v.toFixed(3),
  },
  {
    key: "PUPIL_RATIO",
    label: "Pupil size",
    hint: "Pupil size relative to the eye",
    min: 0.35,
    max: 0.7,
    step: 0.01,
    fmt: (v) => v.toFixed(2),
  },
];

const SETTINGS_STORAGE_KEY = "googly-physics";

function loadPhysicsSettings() {
  try {
    const saved = JSON.parse(localStorage.getItem(SETTINGS_STORAGE_KEY));
    if (saved && typeof saved === "object") {
      for (const s of SETTINGS_SCHEMA) {
        const v = saved[s.key];
        if (Number.isFinite(v)) {
          P[s.key] = Math.min(s.max, Math.max(s.min, v));
        }
      }
    }
  } catch {
    // ignore corrupt saved settings and keep defaults
  }
}

function savePhysicsSettings() {
  try {
    localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(P));
  } catch {
    // private mode etc. — settings just won't persist
  }
}

// Face landmarks (478-point model): eye corners, lids, iris centers.
const LEFT_EYE = { outer: 33, inner: 133, top: 159, bottom: 145, iris: 468 };
const RIGHT_EYE = { outer: 263, inner: 362, top: 386, bottom: 374, iris: 473 };

let landmarker = null;
let running = false;
let lastVideoTime = -1;
let lastT = 0;
let physAcc = 0; // leftover time for the fixed-step physics loop
let nextFaceId = 1;

// Per-face state: a pair of eye simulations plus matching/expiry bookkeeping.
// Keyed by an arbitrary id (not detection order) so people can cross paths
// without their physics state teleporting.
const faceStates = new Map();
const FACE_EXPIRY_MS = 700; // drop state for faces unseen this long

function makeEye() {
  return {
    init: false,
    cx: 0,
    cy: 0,
    r: 20, // simulated socket center/radius, chases the detection target
    tx: 0,
    ty: 0,
    tr: 20, // target geometry from the latest detection
    svx: 0,
    svy: 0, // socket velocity estimate, felt by the pupil on wall contact
    px: 0,
    py: 0,
    vx: 0,
    vy: 0, // pupil absolute screen position + velocity
  };
}

function setStatus(message) {
  statusEl.textContent = message || "";
}

// Build the settings panel rows and wire them to P live.
function initSettings() {
  const syncRow = (s, input, val) => {
    input.value = P[s.key];
    val.textContent = s.fmt(P[s.key]);
  };

  for (const s of SETTINGS_SCHEMA) {
    const row = document.createElement("div");
    row.className = "setting";

    const top = document.createElement("div");
    top.className = "setting-top";
    const label = document.createElement("label");
    label.textContent = s.label;
    const val = document.createElement("span");
    val.className = "setting-val";
    top.append(label, val);

    const input = document.createElement("input");
    input.type = "range";
    input.min = s.min;
    input.max = s.max;
    input.step = s.step;
    input.setAttribute("aria-label", s.label);
    input.addEventListener("input", () => {
      P[s.key] = Number(input.value);
      val.textContent = s.fmt(P[s.key]);
      savePhysicsSettings();
    });

    const hint = document.createElement("p");
    hint.className = "setting-hint";
    hint.textContent = s.hint;

    row.append(top, input, hint);
    syncRow(s, input, val);
    row._sync = () => syncRow(s, input, val);
    settingsRows.append(row);
  }

  const syncAll = () => {
    for (const row of settingsRows.children) row._sync();
  };

  settingsReset.addEventListener("click", () => {
    Object.assign(P, PHYS_DEFAULTS);
    savePhysicsSettings();
    syncAll();
  });

  const setPanelOpen = (open) => {
    settingsPanel.hidden = !open;
    settingsBtn.setAttribute("aria-expanded", String(open));
  };

  settingsBtn.addEventListener("click", () => {
    setPanelOpen(settingsPanel.hidden);
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !settingsPanel.hidden) setPanelOpen(false);
  });

  document.addEventListener("pointerdown", (e) => {
    if (settingsPanel.hidden) return;
    if (settingsPanel.contains(e.target) || settingsBtn.contains(e.target)) return;
    setPanelOpen(false);
  });
}

// MediaPipe gives normalized coords relative to the video frame; the video is
// drawn with object-fit: contain (mirrored), so map into display space here.
function mapPoint(nx, ny) {
  const vw = video.videoWidth || 1;
  const vh = video.videoHeight || 1;
  const dw = canvas.clientWidth;
  const dh = canvas.clientHeight;
  const scale = Math.min(dw / vw, dh / vh);
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

// One fixed physics step for a pupil. The pupil is a ball in screen space;
// the socket is a circle that chases the detected eye position. When the
// face moves, the socket wall catches up to the pupil and flings it — the
// bounce comes from the collision with the moving wall, not from any
// explicitly estimated head acceleration.
function stepEyePhysics(e, h) {
  if (!e.init) {
    e.init = true;
    e.cx = e.tx;
    e.cy = e.ty;
    e.r = Math.max(e.tr, 2);
    e.px = e.cx;
    e.py = e.cy + e.r * (1 - P.PUPIL_RATIO); // start resting at the bottom
    e.vx = (Math.random() - 0.5) * 6 * e.r; // little wobble on appearance
    e.vy = 0;
  }

  // Socket chases the detected geometry; how fast it moves here is the
  // "throw" the pupil feels on contact.
  const k = 1 - Math.exp(-P.SOCKET_TRACK * h);
  const nx = e.cx + (e.tx - e.cx) * k;
  const ny = e.cy + (e.ty - e.cy) * k;
  e.r += (Math.max(e.tr, 2) - e.r) * k;
  e.svx = (nx - e.cx) / h;
  e.svy = (ny - e.cy) / h;
  e.cx = nx;
  e.cy = ny;

  // Gravity and light air drag, then integrate.
  e.vy += P.GRAVITY * e.r * h;
  const damp = Math.exp(-P.AIR_DAMPING * h);
  e.vx *= damp;
  e.vy *= damp;
  e.px += e.vx * h;
  e.py += e.vy * h;

  // Constrain the pupil inside the socket; bounce off the wall using the
  // velocity RELATIVE to the wall, so a moving socket transfers its motion.
  const maxOff = Math.max(e.r * (1 - P.PUPIL_RATIO), 0.5);
  const dx = e.px - e.cx;
  const dy = e.py - e.cy;
  const dist = Math.hypot(dx, dy);
  if (dist > maxOff) {
    const wnx = dx / dist;
    const wny = dy / dist;
    e.px = e.cx + wnx * maxOff;
    e.py = e.cy + wny * maxOff;
    const rvx = e.vx - e.svx;
    const rvy = e.vy - e.svy;
    const vn = rvx * wnx + rvy * wny;
    if (vn > 0) {
      const bounce = vn > P.BOUNCE_MIN * e.r ? P.RESTITUTION : 0;
      const nvx = rvx - (1 + bounce) * vn * wnx;
      const nvy = rvy - (1 + bounce) * vn * wny;
      e.vx = nvx * P.WALL_FRICTION + e.svx;
      e.vy = nvy * P.WALL_FRICTION + e.svy;
    }
  }
}

function processFaces(result, now) {
  const faces = result && result.faceLandmarks ? result.faceLandmarks : [];
  hud.hidden = faces.length > 0;
  if (faces.length > 0) {
    hud.textContent =
      faces.length === 1 ? "1 face found" : `${faces.length} faces found`;
  }

  const dets = [];
  for (const lm of faces.slice(0, MAX_FACES)) {
    const left = eyeGeometry(lm, LEFT_EYE);
    const right = eyeGeometry(lm, RIGHT_EYE);
    const pairR = (left.r + right.r) / 2; // googly pairs share a size
    left.r = pairR;
    right.r = pairR;
    dets.push({
      left,
      right,
      cx: (left.cx + right.cx) / 2,
      cy: (left.cy + right.cy) / 2,
    });
  }

  // Match detections to existing states by proximity so a face keeps its
  // physics as it moves (and when people cross); new faces get new state.
  const threshold = Math.max(canvas.clientWidth, canvas.clientHeight) * 0.25;
  const unused = new Set(faceStates.keys());
  for (const det of dets) {
    let bestId = null;
    let bestDist = threshold;
    for (const id of unused) {
      const s = faceStates.get(id);
      const d = Math.hypot(s.targetCx - det.cx, s.targetCy - det.cy);
      if (d < bestDist) {
        bestDist = d;
        bestId = id;
      }
    }
    let state;
    if (bestId !== null) {
      unused.delete(bestId);
      state = faceStates.get(bestId);
    } else {
      state = {
        left: makeEye(),
        right: makeEye(),
        targetCx: det.cx,
        targetCy: det.cy,
        lastSeen: now,
      };
      faceStates.set(nextFaceId, state);
      nextFaceId++;
    }
    state.left.tx = det.left.cx;
    state.left.ty = det.left.cy;
    state.left.tr = det.left.r;
    state.right.tx = det.right.cx;
    state.right.ty = det.right.cy;
    state.right.tr = det.right.r;
    state.targetCx = det.cx;
    state.targetCy = det.cy;
    state.lastSeen = now;
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
  if (!eye.init || eye.r <= 0) return;
  const R = eye.r;
  const px = eye.px;
  const py = eye.py;
  const pupilR = R * P.PUPIL_RATIO;

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

// Fixed-timestep physics for all eyes, called once per rendered frame. Runs
// at a constant 120 Hz regardless of camera or display frame rate.
function stepPhysics(dt, now) {
  physAcc = Math.min(physAcc + dt, SUBSTEP * MAX_SUBSTEPS);
  while (physAcc >= SUBSTEP) {
    physAcc -= SUBSTEP;
    for (const s of faceStates.values()) {
      stepEyePhysics(s.left, SUBSTEP);
      stepEyePhysics(s.right, SUBSTEP);
    }
  }
  for (const [id, s] of faceStates) {
    if (now - s.lastSeen > FACE_EXPIRY_MS) faceStates.delete(id);
  }
}

function frame(now) {
  if (!running) return;
  const dt = Math.min(Math.max((now - lastT) / 1000, 0.001), 0.05);
  lastT = now;

  if (landmarker && video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    try {
      const result = landmarker.detectForVideo(video, now);
      processFaces(result, now);
    } catch (err) {
      console.error("Detection failed:", err);
    }
  }
  stepPhysics(dt, now);
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
      // Width only: asking for a 16:9 shape makes 4:3 sensors crop (digital
      // zoom). Leaving the height unset returns the camera's native framing.
      video: { facingMode: "user", width: { ideal: 1280 } },
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
    videoBg.srcObject = stream;

    // Un-hide before play(): iOS won't render a display:none video, and don't
    // block on play() — some browsers never resolve it, and the frame loop
    // already waits for frames to arrive on its own.
    video.hidden = false;
    videoBg.hidden = false;
    canvas.hidden = false;
    overlay.hidden = true;
    video.play().catch((err) => console.error("video.play() failed:", err));
    videoBg.play().catch((err) => console.error("videoBg.play() failed:", err));

    hud.hidden = false;
    hud.textContent = "Looking for faces…";
    settingsBtn.hidden = false;
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
  loadPhysicsSettings();
  initSettings();
  preloadModel();
  startBtn.addEventListener("click", start);
}
