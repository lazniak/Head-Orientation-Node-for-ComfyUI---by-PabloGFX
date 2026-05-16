# Head Orientation Node for ComfyUI

Version **1.1.0** — by **PabloGFX**

A custom ComfyUI node that detects faces, estimates head pose (pitch / yaw /
roll), and sorts a batch of input images so that the i-th input best matches
the head orientation of the i-th reference image.

## What's new in 1.1.0

- **Compatibility with modern MediaPipe (>=0.10).** Earlier versions of this
  node relied on `mediapipe.solutions.face_mesh`, which is no longer present in
  the slimmed-down `mediapipe` builds shipped with recent Python embeds (e.g.
  the ComfyUI Windows portable). The node now uses the new
  `mediapipe.tasks.python.vision.FaceLandmarker` API and falls back to the
  legacy `solutions.face_mesh` API if it is available — so the same code works
  on old and new MediaPipe installations.
- **Automatic model download.** On first use the node downloads the official
  Google-hosted `face_landmarker.task` model (~3.7 MB) into
  `head-orientation-node/models/` and reuses it afterwards. No manual setup is
  required.
- **More robust image handling.** Correct RGB handling (the legacy code
  unintentionally swapped R and B channels via an extra `cvtColor` call),
  graceful handling of grayscale and RGBA inputs, and safer batch sorting when
  no input images are provided.
- **Better diagnostics.** The node prints the MediaPipe backend it is using
  (`legacy` vs. `tasks`) and reports model-download progress.

## Data output format

The node returns two outputs:

1. `sorted_images` — the input batch reordered to best match the reference
   batch.
2. `data` — a multi-line string where each line is the head orientation of one
   sorted output image in the form `[x,y,z]`, where:
   - `x` — rotation around the X-axis (pitch, nodding up/down)
   - `y` — rotation around the Y-axis (yaw, turning left/right)
   - `z` — rotation around the Z-axis (roll, tilting side to side)

All values are in degrees, rounded to two decimal places, one orientation per
line.

## Features

- Detects facial landmarks with MediaPipe (new Tasks API or legacy face_mesh).
- Estimates head orientation (pitch, yaw, roll) for every image in a batch.
- Sorts the input batch by similarity to a reference batch (greedy 3D-angle
  match).
- Handles batches of arbitrary size, with safe fallbacks when there are fewer
  inputs than references.

## Installation

1. Clone this repository into your ComfyUI `custom_nodes` directory:
   ```bash
   git clone https://github.com/lazniak/Head-Orientation-Node-for-ComfyUI---by-PabloGFX.git
   ```
2. Install the required dependencies (use the Python that runs ComfyUI — for
   the Windows portable that is `python_embeded\python.exe`):
   ```bash
   pip install -r requirements.txt
   ```
3. Restart ComfyUI. The first run will download
   `models/face_landmarker.task` automatically.

## Requirements

- numpy >= 1.19.3
- opencv-python >= 4.5.5.64
- mediapipe >= 0.10.0
- Pillow >= 8.3.1
- torch >= 1.9.0

## Usage

1. In ComfyUI, locate the node in the browser as **Head Orientation Node - by
   PabloGFX** (category `image/PabloGFX`).
2. Connect a batch of images to `image`.
3. Connect a reference batch to `reference_images`.
4. The node outputs the input batch reordered to best match the orientations of
   the reference batch, together with the per-image orientation string.

## How it works

1. For each image the node converts the ComfyUI tensor to an `RGB uint8` image
   and runs MediaPipe face landmark detection.
2. Six canonical landmarks (eyes, nose, mouth corners, chin) are fed into
   OpenCV's `solvePnP` together with a simple pinhole camera model to recover
   the head rotation, which is decomposed into pitch / yaw / roll.
3. Each reference orientation is greedily matched to the closest unused input
   orientation by Euclidean distance in (pitch, yaw, roll) space.
4. If there are fewer inputs than references, the last matched input is
   repeated to keep the batch sizes aligned.

## Notes about the MediaPipe model

The first run downloads the official Google-hosted face landmarker model from:

```
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

It is cached at `head-orientation-node/models/face_landmarker.task`. If your
ComfyUI installation cannot reach the internet, drop the file there manually
before using the node.

## License

This project is licensed under the Apache License 2.0. This license is
compatible with the licenses of the major dependencies used in this project:

- MediaPipe: Apache License 2.0
- OpenCV: Apache License 2.0
- NumPy: BSD 3-Clause License
- PyTorch: BSD 3-Clause License
- Pillow: HPND License

The Apache License 2.0 allows you to use, modify, distribute, and sublicense
the code, while also providing an express grant of patent rights from
contributors to users. It requires preservation of copyright and license
notices. For the full license text see the `LICENSE` file or
https://www.apache.org/licenses/LICENSE-2.0.

## Contributing

Contributions are welcome. By contributing to this project you agree to
license your contributions under the Apache License 2.0.

## Acknowledgements

Thanks to the developers of MediaPipe, OpenCV, NumPy, PyTorch and Pillow.
