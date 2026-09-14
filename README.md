# Googly Eyes 👀

Point your webcam at a face (or several!) and their eyes become physics-based
googly eyes: the pupils wobble, swing, and bounce around as people move, and
they settle toward the low side when you tilt your head — just like the craft
 googlies stuck on your monitor.

**Try it:** https://jammaloo.github.io/googly-eyes/ (after GitHub Pages finishes
 its first deploy)

## How it works

- **Face detection** — [MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)
  (via `@mediapipe/tasks-vision`, loaded from a CDN) finds up to **4 faces** per
  frame and returns 478 landmarks, including the iris centers. Everything runs
  **on-device** (WASM/WebGL) — no video ever leaves the browser.
- **Googly eyes** — each eye gets a white socket circle sized from its landmark
  geometry (paired eyes share a radius, like real googly eyes), drawn on a
  canvas overlay above the mirrored selfie video.
- **Pupil physics** — each pupil is a little point mass inside its socket:
  - gravity always pulls it "down" *in the head's local frame*, so tilting your
    head makes the pupil roll to the new low side;
  - head movement applies an inertial force opposite to the acceleration, so
    the pupils lag, swing, and rattle as you move;
  - collisions with the socket wall bounce with restitution + friction, plus
    linear drag — tuned in units of eyeball radius so it behaves the same at
    any distance from the camera.

## Run locally

Any static server works (camera access requires `localhost` or HTTPS):

```sh
python3 -m http.server 8000
# then open http://localhost:8000
```

## Browser support

Works on modern desktop and mobile browsers with webcam access (HTTPS
required): Chrome, Edge, Firefox, Safari 16.4+ (including iOS Safari). On
phones it uses the front camera in a mirrored selfie view.

## Privacy

Camera frames are processed entirely in your browser using WebAssembly. Nothing
is uploaded, recorded, or stored.

## License

[MIT](LICENSE)
