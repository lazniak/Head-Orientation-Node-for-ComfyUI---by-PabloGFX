# Head Orientation Node for ComfyUI

Version **1.3.0** — by **PabloGFX**

A custom ComfyUI node that detects faces, estimates head pose (pitch / yaw /
roll), sorts a batch of images by similarity to a query / reference batch, and
emits a configurable, well-formatted string describing the orientations.

---

## What's new in 1.3.0

### New `sort_source` parameter — fixes the "query + candidate pool" use case

In v1.0.x – v1.2.0 the node always sorted the `image` input batch using
`reference_images` as the target poses, and put the sorted inputs into
`sorted_images`. That semantic confused users who had one main image and a
*pool of candidate references* they wanted ranked — the most common case in
ComfyUI workflows.

v1.3.0 adds `sort_source` with two modes:

- **`sort_source = "references"` (new default)** — treats `image` as the query
  / target pose and `reference_images` as the candidate pool.
  `sorted_images` = candidate references reordered by similarity to the query.
  Use this when you have one main image and want the most-similar candidates
  from a pool.
- **`sort_source = "inputs"` (legacy v1.x semantic)** — sorts the `image`
  batch by similarity to `reference_images`. `sorted_images` = reordered
  inputs.

In the user-reported bug case (1 image + 2 references), v1.3.0 with default
settings now returns 2 sorted references in `sorted_images` and 2 matching
lines in `data` — instead of 1 input + N duplicate data lines.

### Lower default detection confidence + new parameter

`min_detection_confidence` is now exposed (`0.05`–`1.0`, default `0.3`,
previously hard-coded to `0.5`). Lower values are more permissive and detect
faces in dim portraits, faces under hats, low-contrast frames etc. The
detector is rebuilt automatically when the value changes.

### `data_content` default changed to `outputs`

With the new `sort_source` default, `outputs` is the most useful default —
each row in `data` describes the corresponding sorted output image (aligned
with what you see in `sorted_images`). Other values:

- `references` — orientations of every `reference_images` item in input order.
- `inputs` — orientations of every `image` item in input order.
- `paired` — sorted outputs paired with the closest match from the OPPOSITE
  batch.

### Per-row distance is now distance to the OPPOSITE batch

`include_distance = true` always reports the angular distance from this row's
item to the closest item in the opposite batch (works for `outputs`,
`inputs`, `references`, `paired`). Previously distance was always
"to references"; now it follows the data content.

---

## What's new in 1.2.0

### Bug fix

- **The `data` output no longer repeats the same row when the input batch is
  smaller than the reference batch.** In v1.1.0 (and earlier), `data` would
  emit duplicate lines because it reflected the sorted output image's
  orientation — and when the legacy sorter ran out of unique inputs it just
  repeated the last one. The new default `data_content = "references"` emits
  one line per reference image instead, so every line carries distinct,
  meaningful information.

### New sort modes

- `similarity_to_refs` *(new default)* — sort all input images by similarity to
  the reference batch, closest first. Output count = input count.
- `reverse_similarity` — same metric, farthest first.
- `match_references` — legacy v1.1.0 / v1.0.x behaviour: greedy 1-to-1 match;
  output count = reference count; the last assigned input is repeated to fill.
- `as_is` — pass inputs through in their original order (still computes
  orientations).

A `similarity_metric` parameter chooses how each input is scored against the
reference batch:

- `min_to_any_ref` *(default)* — distance to the nearest reference.
- `mean_to_refs` — average distance to all references.
- `first_ref_only` — distance to the first reference only.

### New `output_count` override

- `output_count = 0` *(default)* uses the mode's natural size (input count for
  similarity / as_is, reference count for match).
- Set any positive integer to force a fixed batch size. Excess items are
  truncated; missing items are filled by repeating the last.

### New configurable `data` string

| Parameter           | Default       | Effect                                                                       |
| ------------------- | ------------- | ---------------------------------------------------------------------------- |
| `data_content`      | `references`  | What rows the string contains: `references`, `outputs`, `inputs`, `paired`   |
| `data_format`       | `compact`     | `compact`, `labeled`, `csv`, `json`, `verbose`                               |
| `decimals`          | `2`           | Decimal places for angles and distance                                       |
| `angle_unit`        | `degrees`     | `degrees` or `radians`                                                       |
| `include_index`     | `false`       | Prepend the 1-based row index to each line                                   |
| `include_distance`  | `false`       | Append the angular distance to the closest reference (outputs / inputs / paired) |
| `include_header`    | `false`       | csv only — prepend a header row with column names                            |

### Data content modes

- `references` *(default)* — one line per reference image. Use this when you
  want to see what target poses you supplied.
- `outputs` — one line per sorted output image.
- `inputs` — one line per ORIGINAL input image (before sorting).
- `paired` — one line per sorted output, paired with its closest reference and
  distance. Best when you want a side-by-side comparison.

### Data format examples

`compact` (default, same shape as v1.1.0):

```
[pitch,yaw,roll]
[pitch,yaw,roll]
```

`labeled`:

```
pitch=-0.72 yaw=-2.11 roll=-0.13
pitch=-0.81 yaw=1.72 roll=0.12
```

`csv` with header + index + distance + `paired`:

```
index,pitch,yaw,roll,ref_pitch,ref_yaw,ref_roll,distance
1,-0.72,-2.11,-0.13,-0.72,-2.11,-0.13,0.00
2,-0.81,1.72,0.12,-0.81,1.72,0.12,0.00
```

`json`:

```json
[
  {
    "label": "output",
    "pitch": -0.72,
    "yaw": -2.11,
    "roll": -0.13,
    "reference": { "pitch": -0.72, "yaw": -2.11, "roll": -0.13 },
    "distance": 0.0
  }
]
```

`verbose`:

```
Output 1:
  Orientation: pitch=-0.72, yaw=-2.11, roll=-0.13
  Reference:   pitch=-0.72, yaw=-2.11, roll=-0.13
  Distance:    0.00
```

---

## What's new in 1.1.0

- **Compatibility with modern MediaPipe (>=0.10).** The previous code relied on
  `mediapipe.solutions.face_mesh`, which is missing from the slimmed-down
  mediapipe builds shipped with recent Python embeds (e.g. the ComfyUI Windows
  portable). The node now uses the new
  `mediapipe.tasks.python.vision.FaceLandmarker` API and falls back to the
  legacy `solutions.face_mesh` API if it is available.
- **Automatic model download** of the official Google-hosted
  `face_landmarker.task` (~3.7 MB) into `models/`.
- **Correct RGB handling** (the legacy code unintentionally swapped R and B
  via an extra `cv2.cvtColor` call), graceful handling of grayscale / RGBA
  inputs, and safer batch sorting when no input images are provided.

---

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
4. (Optional) Tweak `sort_mode`, `data_content`, `data_format`, etc.
5. The node outputs the input batch sorted/matched and the per-row data
   string.

## How it works

1. Each image is converted to an RGB `uint8` array and run through MediaPipe
   face landmark detection (Tasks API by default; legacy `solutions` API as a
   fallback).
2. Six canonical landmarks (eyes, nose, mouth corners, chin) are fed into
   OpenCV's `solvePnP` together with a simple pinhole camera model to recover
   the head rotation, then decomposed into pitch / yaw / roll.
3. Inputs are sorted (or matched 1-to-1) against the references using the
   selected mode and metric.
4. The data string is built according to `data_content`, `data_format`, and
   the formatting flags.

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
