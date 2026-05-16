"""Head Orientation Node for ComfyUI - by PabloGFX.

Detects faces in a batch of images, estimates head pose (pitch/yaw/roll) and
sorts the input batch so the i-th input best matches the orientation of the
i-th reference image.

This module supports both the new MediaPipe Tasks API (>=0.10) and the
legacy `mediapipe.solutions.face_mesh` API for maximum compatibility.
"""

import os
import urllib.request
from typing import List, Optional, Tuple

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


# Landmark indices used for PnP head-pose estimation (canonical face mesh):
#   33  - right eye outer corner
#   263 - left eye outer corner
#   1   - nose tip
#   61  - right mouth corner
#   291 - left mouth corner
#   199 - chin
PNP_LANDMARK_INDICES = (33, 263, 1, 61, 291, 199)

# Official Google-hosted MediaPipe face landmarker model (float16, ~3.7 MB).
FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
FACE_LANDMARKER_MODEL_FILENAME = "face_landmarker.task"


def _plugin_model_dir() -> str:
    """Returns the directory where the face landmarker model is cached."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "models")


def _ensure_face_landmarker_model() -> str:
    """Downloads the MediaPipe face landmarker model on first use.

    Returns the absolute path to the .task model file.
    """
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
    """Returns True if the installed mediapipe exposes the legacy solutions API."""
    return hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh")


def _has_tasks_api() -> bool:
    """Returns True if mediapipe.tasks vision FaceLandmarker is importable."""
    try:
        from mediapipe.tasks.python import vision as _vision  # noqa: F401
        return hasattr(_vision, "FaceLandmarker")
    except Exception:
        return False


class _FaceDetector:
    """Thin adapter that hides the differences between the legacy
    `solutions.face_mesh` API and the new `tasks.python.vision.FaceLandmarker`
    API. Exposes a single `process(rgb_uint8)` -> Optional[list of (x, y, z)
    normalized landmarks] method.
    """

    def __init__(self, max_num_faces: int = 1, min_detection_confidence: float = 0.5):
        self._mode: str
        self._impl = None
        self._max_num_faces = max_num_faces
        self._min_detection_confidence = min_detection_confidence

        # Prefer legacy when available (it keeps behavior identical to older
        # versions of this node). Fall back to the new Tasks API otherwise.
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
        """Runs face detection on a single HxWx3 uint8 RGB image.

        Returns a list of (x_norm, y_norm, z_norm) tuples for the first
        detected face, or None if no face was found.
        """
        if self._mode == "legacy":
            results = self._impl.process(image_rgb_uint8)
            if not getattr(results, "multi_face_landmarks", None):
                return None
            face = results.multi_face_landmarks[0]
            return [(lm.x, lm.y, lm.z) for lm in face.landmark]

        # Tasks API
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


class HeadOrientationNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "reference_images": ("IMAGE",),
            }
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

    def process_images(self, image, reference_images):
        print(f"{Colors.BLUE}[PROCESS] Input image shape: {image.shape}, dtype: {image.dtype}{Colors.ENDC}")
        print(f"{Colors.BLUE}[PROCESS] Reference images shape: {reference_images.shape}, dtype: {reference_images.dtype}{Colors.ENDC}")

        print(f"{Colors.YELLOW}[PROCESS] Analyzing input images...{Colors.ENDC}")
        input_orientations = self.analyze_orientations(image)
        print(f"{Colors.YELLOW}[PROCESS] Analyzing reference images...{Colors.ENDC}")
        reference_orientations = self.analyze_orientations(reference_images)

        print(f"{Colors.YELLOW}[PROCESS] Sorting images...{Colors.ENDC}")
        sorted_images, sorted_orientations = self.sort_images(image, input_orientations, reference_orientations)

        data_output = self.format_orientation_data(sorted_orientations)

        print(f"{Colors.GREEN}[PROCESS] Output images shape: {sorted_images.shape}, dtype: {sorted_images.dtype}{Colors.ENDC}")
        print(f"{Colors.GREEN}[PROCESS] Output data: \n{data_output}{Colors.ENDC}")
        return (sorted_images, data_output)

    def analyze_orientations(self, images: torch.Tensor) -> List[Tuple[float, float, float]]:
        print(f"{Colors.BLUE}[ANALYZE] Starting analysis of {images.shape[0]} images{Colors.ENDC}")
        orientations: List[Tuple[float, float, float]] = []

        for idx in range(images.shape[0]):
            print(f"{Colors.YELLOW}[ANALYZE] Processing image {idx+1}/{images.shape[0]}{Colors.ENDC}")
            img = images[idx].detach().cpu().numpy()
            print(f"{Colors.BLUE}[ANALYZE] Image {idx+1} raw shape: {img.shape}, dtype: {img.dtype}{Colors.ENDC}")

            # Normalize layout to HWC.
            if img.ndim == 3 and img.shape[0] in (1, 3, 4) and img.shape[-1] not in (1, 3, 4):
                print(f"{Colors.YELLOW}[ANALYZE] Image {idx+1} is CHW, transposing to HWC{Colors.ENDC}")
                img = np.transpose(img, (1, 2, 0))

            # ComfyUI IMAGE tensors are float32 in [0, 1]. Convert to uint8 RGB.
            if img.dtype != np.uint8:
                img = np.clip(img, 0.0, 1.0)
                img = (img * 255.0 + 0.5).astype(np.uint8)

            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 1:
                img = cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2RGB)
            elif img.shape[2] >= 4:
                print(f"{Colors.YELLOW}[ANALYZE] Image {idx+1} has {img.shape[2]} channels, dropping alpha{Colors.ENDC}")
                img = img[:, :, :3]

            # MediaPipe expects RGB uint8 - ComfyUI tensors are already RGB.
            img = np.ascontiguousarray(img)
            h, w = img.shape[:2]

            print(f"{Colors.YELLOW}[ANALYZE] Detecting face in image {idx+1} ({w}x{h}){Colors.ENDC}")
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

            print(f"{Colors.YELLOW}[ANALYZE] Calculating head pose for image {idx+1}{Colors.ENDC}")
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

    def sort_images(self, images, input_orientations, reference_orientations):
        print(f"{Colors.BLUE}[SORT] Sorting {len(input_orientations)} input images based on {len(reference_orientations)} reference orientations{Colors.ENDC}")
        sorted_indices: List[int] = []
        sorted_orientations: List[Tuple[float, float, float]] = []
        used_indices = set()

        for ref_idx, ref_orientation in enumerate(reference_orientations):
            print(f"{Colors.YELLOW}[SORT] Finding best match for reference orientation {ref_idx+1}{Colors.ENDC}")
            best_diff = float('inf')
            best_index = -1

            for i, orientation in enumerate(input_orientations):
                if i in used_indices:
                    continue
                diff = sum((a - b) ** 2 for a, b in zip(ref_orientation, orientation)) ** 0.5
                if diff < best_diff:
                    best_diff = diff
                    best_index = i

            if best_index != -1:
                sorted_indices.append(best_index)
                sorted_orientations.append(input_orientations[best_index])
                used_indices.add(best_index)
                print(f"{Colors.GREEN}[SORT] Best match for reference {ref_idx+1} is input image {best_index+1}, difference: {best_diff:.2f}{Colors.ENDC}")
            else:
                for i in range(len(input_orientations)):
                    if i not in used_indices:
                        sorted_indices.append(i)
                        sorted_orientations.append(input_orientations[i])
                        used_indices.add(i)
                        print(f"{Colors.YELLOW}[SORT] No match found for reference {ref_idx+1}, using input image {i+1}{Colors.ENDC}")
                        break

        while len(sorted_indices) < len(reference_orientations) and sorted_indices:
            sorted_indices.append(sorted_indices[-1])
            sorted_orientations.append(sorted_orientations[-1])
            print(f"{Colors.YELLOW}[SORT] Not enough input images, repeating last image (index {sorted_indices[-1]+1}){Colors.ENDC}")

        if not sorted_indices:
            print(f"{Colors.RED}[SORT] No input images to sort, returning original batch{Colors.ENDC}")
            return images, []

        print(f"{Colors.BLUE}[SORT] Final sorted indices: {sorted_indices}{Colors.ENDC}")
        sorted_images = images[sorted_indices]
        print(f"{Colors.GREEN}[SORT] Sorted images shape: {sorted_images.shape}{Colors.ENDC}")
        return sorted_images, sorted_orientations

    def format_orientation_data(self, orientations):
        return "".join(f"[{x:.2f},{y:.2f},{z:.2f}]\n" for x, y, z in orientations)


# Mappings used by ComfyUI to discover and load custom nodes
NODE_CLASS_MAPPINGS = {
    "HeadOrientationNode": HeadOrientationNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HeadOrientationNode": "Head Orientation Node - by PabloGFX",
}
