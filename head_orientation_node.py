"""Head Orientation Node for ComfyUI - by PabloGFX.

Detects faces in a batch of images, estimates head pose (pitch / yaw / roll)
with MediaPipe + OpenCV PnP, sorts the input batch by similarity to a reference
batch (or matches one-to-one), and emits a configurable data string describing
the orientations.

Supports both the modern MediaPipe Tasks API
(`mediapipe.tasks.python.vision.FaceLandmarker`) and the legacy
`mediapipe.solutions.face_mesh` API for maximum compatibility.
"""

import json
import math
import os
import urllib.request
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

import mediapipe as mp


# ANSI colors for log output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


Orientation = Tuple[float, float, float]


# ---------------------------------------------------------------------------
# MediaPipe model wiring
# ---------------------------------------------------------------------------

# Landmark indices used for PnP head-pose estimation (canonical face mesh):
#   33  - right eye outer corner
#   263 - left eye outer corner
#   1   - nose tip
#   61  - right mouth corner
#   291 - left mouth corner
#   199 - chin
PNP_LANDMARK_INDICES: Tuple[int, ...] = (33, 263, 1, 61, 291, 199)

FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
FACE_LANDMARKER_MODEL_FILENAME = "face_landmarker.task"


def _plugin_model_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "models")


def _ensure_face_landmarker_model() -> str:
    model_dir = _plugin_model_dir()
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, FACE_LANDMARKER_MODEL_FILENAME)

    if os.path.isfile(model_path) and os.path.getsize(model_path) > 0:
        return model_path

    print(
        f"{Colors.YELLOW}[HeadOrientationNode] Downloading MediaPipe face "
        f"landmarker model to {model_path} ...{Colors.ENDC}"
    )
    try:
        urllib.request.urlretrieve(FACE_LANDMARKER_MODEL_URL, model_path)
    except Exception as exc:
        if os.path.isfile(model_path):
            try:
                os.remove(model_path)
            except OSError:
                pass
        raise RuntimeError(
            f"Failed to download MediaPipe face landmarker model from "
            f"{FACE_LANDMARKER_MODEL_URL}: {exc}"
        ) from exc

    print(
        f"{Colors.GREEN}[HeadOrientationNode] Model downloaded "
        f"({os.path.getsize(model_path)/1024:.1f} KB).{Colors.ENDC}"
    )
    return model_path


def _has_legacy_solutions() -> bool:
    return hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh")


def _has_tasks_api() -> bool:
    try:
        from mediapipe.tasks.python import vision as _vision  # noqa: F401
        return hasattr(_vision, "FaceLandmarker")
    except Exception:
        return False


class _FaceDetector:
    """Adapter hiding the differences between the legacy
    `solutions.face_mesh` API and the new
    `tasks.python.vision.FaceLandmarker` API.
    """

    def __init__(self, max_num_faces: int = 1, min_detection_confidence: float = 0.5):
        self._mode: str
        self._impl = None
        self._max_num_faces = max_num_faces
        self._min_detection_confidence = min_detection_confidence

        if _has_legacy_solutions():
            self._mode = "legacy"
            face_mesh_module = mp.solutions.face_mesh
            self._impl = face_mesh_module.FaceMesh(
                static_image_mode=True,
                max_num_faces=max_num_faces,
                min_detection_confidence=min_detection_confidence,
            )
            print(
                f"{Colors.HEADER}[HeadOrientationNode] Using legacy "
                f"mediapipe.solutions.face_mesh (v{mp.__version__}){Colors.ENDC}"
            )
        elif _has_tasks_api():
            self._mode = "tasks"
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            model_path = _ensure_face_landmarker_model()
            base_options = mp_python.BaseOptions(model_asset_path=model_path)
            options = mp_vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=mp_vision.RunningMode.IMAGE,
                num_faces=max_num_faces,
                min_face_detection_confidence=min_detection_confidence,
                min_face_presence_confidence=min_detection_confidence,
                min_tracking_confidence=min_detection_confidence,
            )
            self._impl = mp_vision.FaceLandmarker.create_from_options(options)
            print(
                f"{Colors.HEADER}[HeadOrientationNode] Using mediapipe Tasks "
                f"FaceLandmarker (v{mp.__version__}){Colors.ENDC}"
            )
        else:
            raise RuntimeError(
                "Installed mediapipe (v{ver}) exposes neither the legacy "
                "`solutions.face_mesh` API nor the new "
                "`tasks.python.vision.FaceLandmarker` API. Please install "
                "mediapipe>=0.10 with `pip install -U mediapipe`.".format(
                    ver=getattr(mp, "__version__", "?")
                )
            )

    @property
    def mode(self) -> str:
        return self._mode

    def process(self, image_rgb_uint8: np.ndarray) -> Optional[List[Tuple[float, float, float]]]:
        if self._mode == "legacy":
            results = self._impl.process(image_rgb_uint8)
            if not getattr(results, "multi_face_landmarks", None):
                return None
            face = results.multi_face_landmarks[0]
            return [(lm.x, lm.y, lm.z) for lm in face.landmark]

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb_uint8)
        result = self._impl.detect(mp_image)
        if not getattr(result, "face_landmarks", None):
            return None
        face = result.face_landmarks[0]
        return [(lm.x, lm.y, lm.z) for lm in face]

    def close(self) -> None:
        try:
            if self._impl is not None and hasattr(self._impl, "close"):
                self._impl.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Sort + scoring helpers
# ---------------------------------------------------------------------------

SORT_MODES = ["similarity_to_refs", "reverse_similarity", "match_references", "as_is"]
SIMILARITY_METRICS = ["min_to_any_ref", "mean_to_refs", "first_ref_only"]
DATA_CONTENTS = ["references", "outputs", "inputs", "paired"]
DATA_FORMATS = ["compact", "labeled", "csv", "json", "verbose"]
ANGLE_UNITS = ["degrees", "radians"]


def _angular_distance(a: Orientation, b: Orientation) -> float:
    return float(math.sqrt(
        (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2
    ))


def _closest_ref_idx(orientation: Orientation, refs: Sequence[Orientation]) -> Tuple[int, float]:
    if not refs:
        return -1, float('inf')
    best_idx = 0
    best_dist = float('inf')
    for i, r in enumerate(refs):
        d = _angular_distance(orientation, r)
        if d < best_dist:
            best_dist = d
            best_idx = i
    return best_idx, best_dist


def _score(orientation: Orientation, refs: Sequence[Orientation], metric: str) -> float:
    if not refs:
        return 0.0
    distances = [_angular_distance(orientation, r) for r in refs]
    if metric == "min_to_any_ref":
        return min(distances)
    if metric == "mean_to_refs":
        return sum(distances) / len(distances)
    if metric == "first_ref_only":
        return distances[0]
    return min(distances)


def _sort_similarity(inputs: Sequence[Orientation], refs: Sequence[Orientation],
                     metric: str, descending: bool = False) -> List[int]:
    scored = [(i, _score(o, refs, metric)) for i, o in enumerate(inputs)]
    scored.sort(key=lambda t: t[1], reverse=descending)
    return [t[0] for t in scored]


def _sort_match_references(inputs: Sequence[Orientation],
                           references: Sequence[Orientation]) -> List[int]:
    """Legacy greedy 1-to-1 match. Last assigned input repeats to fill."""
    sorted_indices: List[int] = []
    used: set = set()
    for ref in references:
        best_diff = float('inf')
        best_idx = -1
        for i, o in enumerate(inputs):
            if i in used:
                continue
            d = _angular_distance(ref, o)
            if d < best_diff:
                best_diff = d
                best_idx = i
        if best_idx != -1:
            sorted_indices.append(best_idx)
            used.add(best_idx)
        else:
            for i in range(len(inputs)):
                if i not in used:
                    sorted_indices.append(i)
                    used.add(i)
                    break
    while len(sorted_indices) < len(references) and sorted_indices:
        sorted_indices.append(sorted_indices[-1])
    return sorted_indices


def _apply_count(indices: List[int], count: int) -> List[int]:
    if count <= 0 or not indices:
        return indices
    if len(indices) >= count:
        return indices[:count]
    out = list(indices)
    while len(out) < count:
        out.append(out[-1])
    return out


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

_DEG2RAD = math.pi / 180.0


def _to_unit(value: Optional[float], angle_unit: str) -> Optional[float]:
    if value is None:
        return None
    if angle_unit == "radians":
        return value * _DEG2RAD
    return value


def _orientation_to_unit(o: Optional[Orientation], angle_unit: str) -> Optional[Orientation]:
    if o is None:
        return None
    if angle_unit == "radians":
        return (o[0] * _DEG2RAD, o[1] * _DEG2RAD, o[2] * _DEG2RAD)
    return tuple(o)  # type: ignore[return-value]


class _Row:
    """Single row in the data output."""

    __slots__ = ("index", "label", "orientation", "reference", "distance")

    def __init__(self,
                 index: int,
                 label: str,
                 orientation: Optional[Orientation],
                 reference: Optional[Orientation],
                 distance: Optional[float]):
        self.index = index
        self.label = label
        self.orientation = orientation
        self.reference = reference
        self.distance = distance


def _build_rows(data_content: str,
                input_orientations: Sequence[Orientation],
                reference_orientations: Sequence[Orientation],
                sorted_indices: Sequence[int],
                angle_unit: str) -> List[_Row]:
    rows: List[_Row] = []

    if data_content == "references":
        for i, ref in enumerate(reference_orientations):
            rows.append(_Row(
                index=i + 1,
                label="reference",
                orientation=_orientation_to_unit(ref, angle_unit),
                reference=None,
                distance=None,
            ))
        return rows

    if data_content == "inputs":
        for i, o in enumerate(input_orientations):
            closest_idx, closest_dist = _closest_ref_idx(o, reference_orientations)
            ref = reference_orientations[closest_idx] if closest_idx >= 0 else None
            rows.append(_Row(
                index=i + 1,
                label="input",
                orientation=_orientation_to_unit(o, angle_unit),
                reference=_orientation_to_unit(ref, angle_unit),
                distance=_to_unit(closest_dist if ref is not None else None, angle_unit),
            ))
        return rows

    # outputs / paired share the same iteration over sorted outputs
    for i, src_idx in enumerate(sorted_indices):
        o = input_orientations[src_idx] if 0 <= src_idx < len(input_orientations) else (0.0, 0.0, 0.0)
        closest_idx, closest_dist = _closest_ref_idx(o, reference_orientations)
        ref = reference_orientations[closest_idx] if closest_idx >= 0 else None
        rows.append(_Row(
            index=i + 1,
            label="output",
            orientation=_orientation_to_unit(o, angle_unit),
            reference=_orientation_to_unit(ref, angle_unit) if data_content == "paired" else None,
            distance=_to_unit(closest_dist if ref is not None else None, angle_unit),
        ))
    return rows


def _fmt_num(v: Optional[float], decimals: int) -> str:
    if v is None:
        return ""
    return f"{v:.{decimals}f}"


def _fmt_xyz(o: Optional[Orientation], decimals: int) -> str:
    if o is None:
        return ""
    return f"{_fmt_num(o[0], decimals)},{_fmt_num(o[1], decimals)},{_fmt_num(o[2], decimals)}"


def _format_data(rows: Sequence[_Row], fmt: str, decimals: int,
                 include_index: bool, include_distance: bool,
                 include_header: bool) -> str:
    if not rows:
        return ""

    has_reference = any(r.reference is not None for r in rows)

    if fmt == "compact":
        lines: List[str] = []
        for r in rows:
            parts: List[str] = []
            if include_index:
                parts.append(f"[{r.index}]")
            parts.append(f"[{_fmt_xyz(r.orientation, decimals)}]")
            if has_reference and r.reference is not None:
                parts.append(f"target=[{_fmt_xyz(r.reference, decimals)}]")
            if include_distance and r.distance is not None:
                parts.append(f"dist={_fmt_num(r.distance, decimals)}")
            lines.append(" ".join(parts))
        return "\n".join(lines) + "\n"

    if fmt == "labeled":
        lines = []
        for r in rows:
            parts = []
            if include_index:
                parts.append(f"#{r.index}")
            o = r.orientation if r.orientation is not None else (0.0, 0.0, 0.0)
            parts.append(
                f"pitch={_fmt_num(o[0], decimals)} "
                f"yaw={_fmt_num(o[1], decimals)} "
                f"roll={_fmt_num(o[2], decimals)}"
            )
            if has_reference and r.reference is not None:
                ref = r.reference
                parts.append(
                    f"| target pitch={_fmt_num(ref[0], decimals)} "
                    f"yaw={_fmt_num(ref[1], decimals)} "
                    f"roll={_fmt_num(ref[2], decimals)}"
                )
            if include_distance and r.distance is not None:
                parts.append(f"| dist={_fmt_num(r.distance, decimals)}")
            lines.append(" ".join(parts))
        return "\n".join(lines) + "\n"

    if fmt == "csv":
        lines = []
        if include_header:
            headers: List[str] = []
            if include_index:
                headers.append("index")
            headers.extend(["pitch", "yaw", "roll"])
            if has_reference:
                headers.extend(["ref_pitch", "ref_yaw", "ref_roll"])
            if include_distance:
                headers.append("distance")
            lines.append(",".join(headers))
        for r in rows:
            row: List[str] = []
            if include_index:
                row.append(str(r.index))
            o = r.orientation if r.orientation is not None else (0.0, 0.0, 0.0)
            row.extend([
                _fmt_num(o[0], decimals),
                _fmt_num(o[1], decimals),
                _fmt_num(o[2], decimals),
            ])
            if has_reference:
                ref = r.reference if r.reference is not None else (None, None, None)
                row.extend([
                    _fmt_num(ref[0], decimals),
                    _fmt_num(ref[1], decimals),
                    _fmt_num(ref[2], decimals),
                ])
            if include_distance:
                row.append(_fmt_num(r.distance, decimals))
            lines.append(",".join(row))
        return "\n".join(lines) + "\n"

    if fmt == "json":
        out: List[dict] = []
        for r in rows:
            obj: dict = {}
            if include_index:
                obj["index"] = r.index
            obj["label"] = r.label
            if r.orientation is not None:
                obj["pitch"] = round(r.orientation[0], decimals)
                obj["yaw"] = round(r.orientation[1], decimals)
                obj["roll"] = round(r.orientation[2], decimals)
            if has_reference and r.reference is not None:
                obj["reference"] = {
                    "pitch": round(r.reference[0], decimals),
                    "yaw": round(r.reference[1], decimals),
                    "roll": round(r.reference[2], decimals),
                }
            if include_distance and r.distance is not None:
                obj["distance"] = round(r.distance, decimals)
            out.append(obj)
        return json.dumps(out, indent=2)

    if fmt == "verbose":
        parts: List[str] = []
        for r in rows:
            parts.append(f"{r.label.capitalize()} {r.index}:")
            if r.orientation is not None:
                parts.append(
                    f"  Orientation: pitch={_fmt_num(r.orientation[0], decimals)}, "
                    f"yaw={_fmt_num(r.orientation[1], decimals)}, "
                    f"roll={_fmt_num(r.orientation[2], decimals)}"
                )
            if has_reference and r.reference is not None:
                parts.append(
                    f"  Reference:   pitch={_fmt_num(r.reference[0], decimals)}, "
                    f"yaw={_fmt_num(r.reference[1], decimals)}, "
                    f"roll={_fmt_num(r.reference[2], decimals)}"
                )
            if include_distance and r.distance is not None:
                parts.append(f"  Distance:    {_fmt_num(r.distance, decimals)}")
            parts.append("")
        return "\n".join(parts)

    return ""


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class HeadOrientationNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "reference_images": ("IMAGE",),
            },
            "optional": {
                "sort_mode": (
                    SORT_MODES,
                    {
                        "default": "similarity_to_refs",
                        "tooltip": (
                            "similarity_to_refs (default): sort inputs by similarity to references, "
                            "closest first; output count = input count.\n"
                            "reverse_similarity: same metric, farthest first.\n"
                            "match_references: legacy v1.1.0 1-to-1 greedy match; "
                            "output count = reference count (repeats inputs if needed).\n"
                            "as_is: pass inputs through unchanged."
                        ),
                    },
                ),
                "similarity_metric": (
                    SIMILARITY_METRICS,
                    {
                        "default": "min_to_any_ref",
                        "tooltip": (
                            "Distance metric used for similarity sorting.\n"
                            "min_to_any_ref: distance to the nearest reference.\n"
                            "mean_to_refs: average distance to all references.\n"
                            "first_ref_only: distance to the first reference only."
                        ),
                    },
                ),
                "output_count": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 4096,
                        "step": 1,
                        "tooltip": (
                            "Force the number of output images. 0 = auto: input count for "
                            "similarity/reverse/as_is modes, reference count for match_references. "
                            "Excess items are truncated; missing items repeat the last."
                        ),
                    },
                ),
                "data_content": (
                    DATA_CONTENTS,
                    {
                        "default": "references",
                        "tooltip": (
                            "What goes into the `data` string.\n"
                            "references (default): one line per reference image (target poses).\n"
                            "outputs: one line per sorted output image.\n"
                            "inputs: one line per ORIGINAL input image (pre-sort).\n"
                            "paired: one line per sorted output with its closest reference."
                        ),
                    },
                ),
                "data_format": (
                    DATA_FORMATS,
                    {
                        "default": "compact",
                        "tooltip": (
                            "compact (default, legacy v1.1.0): [pitch,yaw,roll] per line.\n"
                            "labeled: 'pitch=X yaw=Y roll=Z' per line.\n"
                            "csv: comma-separated values (optional header).\n"
                            "json: JSON array of objects.\n"
                            "verbose: multi-line human-readable per item."
                        ),
                    },
                ),
                "decimals": (
                    "INT",
                    {
                        "default": 2,
                        "min": 0,
                        "max": 6,
                        "step": 1,
                        "tooltip": "Decimal places for angles and distance.",
                    },
                ),
                "angle_unit": (
                    ANGLE_UNITS,
                    {
                        "default": "degrees",
                        "tooltip": "Output unit for angles and distance.",
                    },
                ),
                "include_index": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Prepend the 1-based row index to each line.",
                    },
                ),
                "include_distance": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "For outputs/inputs/paired content: append the angular distance "
                            "from this row's orientation to the closest reference."
                        ),
                    },
                ),
                "include_header": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "csv format only: prepend a header row with column names.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("sorted_images", "data")
    FUNCTION = "process_images"
    CATEGORY = "image/PabloGFX"

    def __init__(self):
        self._detector = _FaceDetector(max_num_faces=1, min_detection_confidence=0.5)
        print(
            f"{Colors.HEADER}HeadOrientationNode initialized "
            f"(mode={self._detector.mode}){Colors.ENDC}"
        )

    def __del__(self):
        try:
            if getattr(self, "_detector", None) is not None:
                self._detector.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Pose analysis
    # ------------------------------------------------------------------

    def analyze_orientations(self, images: torch.Tensor) -> List[Orientation]:
        print(f"{Colors.BLUE}[ANALYZE] Starting analysis of {images.shape[0]} images{Colors.ENDC}")
        orientations: List[Orientation] = []

        for idx in range(images.shape[0]):
            img = images[idx].detach().cpu().numpy()

            if img.ndim == 3 and img.shape[0] in (1, 3, 4) and img.shape[-1] not in (1, 3, 4):
                img = np.transpose(img, (1, 2, 0))

            if img.dtype != np.uint8:
                img = np.clip(img, 0.0, 1.0)
                img = (img * 255.0 + 0.5).astype(np.uint8)

            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 1:
                img = cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2RGB)
            elif img.shape[2] >= 4:
                img = img[:, :, :3]

            img = np.ascontiguousarray(img)
            h, w = img.shape[:2]

            landmarks = self._detector.process(img)
            if landmarks is None:
                print(f"{Colors.RED}[ANALYZE] No face detected in image {idx+1}{Colors.ENDC}")
                orientations.append((0.0, 0.0, 0.0))
                continue

            face_2d: List[List[float]] = []
            face_3d: List[List[float]] = []
            for lmk_idx in PNP_LANDMARK_INDICES:
                if lmk_idx >= len(landmarks):
                    continue
                lx, ly, lz = landmarks[lmk_idx]
                x_px, y_px = float(lx * w), float(ly * h)
                face_2d.append([x_px, y_px])
                face_3d.append([x_px, y_px, float(lz)])

            if len(face_2d) < 4:
                print(f"{Colors.RED}[ANALYZE] Not enough landmarks for PnP in image {idx+1}{Colors.ENDC}")
                orientations.append((0.0, 0.0, 0.0))
                continue

            face_2d_np = np.asarray(face_2d, dtype=np.float64)
            face_3d_np = np.asarray(face_3d, dtype=np.float64)

            focal_length = float(w)
            cam_matrix = np.array(
                [
                    [focal_length, 0.0, w / 2.0],
                    [0.0, focal_length, h / 2.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            success, rot_vec, _ = cv2.solvePnP(face_3d_np, face_2d_np, cam_matrix, dist_matrix)
            if not success:
                print(f"{Colors.RED}[ANALYZE] solvePnP failed for image {idx+1}{Colors.ENDC}")
                orientations.append((0.0, 0.0, 0.0))
                continue

            rmat, _ = cv2.Rodrigues(rot_vec)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
            x_deg, y_deg, z_deg = angles[0] * 360.0, angles[1] * 360.0, angles[2] * 360.0
            orientations.append((x_deg, y_deg, z_deg))
            print(
                f"{Colors.GREEN}[ANALYZE] Image {idx+1}: "
                f"pitch={x_deg:.2f}, yaw={y_deg:.2f}, roll={z_deg:.2f}{Colors.ENDC}"
            )

        print(f"{Colors.GREEN}[ANALYZE] Analysis completed for {len(orientations)} images{Colors.ENDC}")
        return orientations

    # ------------------------------------------------------------------
    # Sorting
    # ------------------------------------------------------------------

    def _sort(self, input_orientations: List[Orientation],
              reference_orientations: List[Orientation],
              sort_mode: str, similarity_metric: str,
              output_count: int) -> List[int]:
        if not input_orientations:
            return []

        if sort_mode == "as_is":
            indices = list(range(len(input_orientations)))
            target_count = output_count or len(input_orientations)
        elif sort_mode == "match_references":
            indices = _sort_match_references(input_orientations, reference_orientations)
            target_count = output_count or len(reference_orientations) or len(input_orientations)
        elif sort_mode == "reverse_similarity":
            indices = _sort_similarity(input_orientations, reference_orientations,
                                       similarity_metric, descending=True)
            target_count = output_count or len(input_orientations)
        else:  # similarity_to_refs (default)
            indices = _sort_similarity(input_orientations, reference_orientations,
                                       similarity_metric, descending=False)
            target_count = output_count or len(input_orientations)

        return _apply_count(indices, target_count)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process_images(self, image, reference_images,
                       sort_mode: str = "similarity_to_refs",
                       similarity_metric: str = "min_to_any_ref",
                       output_count: int = 0,
                       data_content: str = "references",
                       data_format: str = "compact",
                       decimals: int = 2,
                       angle_unit: str = "degrees",
                       include_index: bool = False,
                       include_distance: bool = False,
                       include_header: bool = False):
        print(
            f"{Colors.BLUE}[PROCESS] image batch: {tuple(image.shape)}, "
            f"reference batch: {tuple(reference_images.shape)}{Colors.ENDC}"
        )
        print(
            f"{Colors.BLUE}[PROCESS] sort={sort_mode}, metric={similarity_metric}, "
            f"count={output_count or 'auto'}, data={data_content}, fmt={data_format}, "
            f"dec={decimals}, unit={angle_unit}{Colors.ENDC}"
        )

        input_orientations = self.analyze_orientations(image)
        reference_orientations = self.analyze_orientations(reference_images)

        sorted_indices = self._sort(
            input_orientations,
            reference_orientations,
            sort_mode,
            similarity_metric,
            output_count,
        )

        if sorted_indices:
            sorted_images = image[sorted_indices]
        else:
            sorted_images = image

        rows = _build_rows(
            data_content,
            input_orientations,
            reference_orientations,
            sorted_indices,
            angle_unit,
        )
        data_output = _format_data(
            rows,
            data_format,
            decimals,
            include_index,
            include_distance,
            include_header,
        )

        print(
            f"{Colors.GREEN}[PROCESS] sorted_images: {tuple(sorted_images.shape)}, "
            f"data rows: {len(rows)}{Colors.ENDC}"
        )
        if data_output:
            preview = data_output if len(data_output) < 400 else data_output[:400] + "...(truncated)"
            print(f"{Colors.GREEN}[PROCESS] data preview:\n{preview}{Colors.ENDC}")

        return (sorted_images, data_output)


NODE_CLASS_MAPPINGS = {
    "HeadOrientationNode": HeadOrientationNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HeadOrientationNode": "Head Orientation Node - by PabloGFX",
}
