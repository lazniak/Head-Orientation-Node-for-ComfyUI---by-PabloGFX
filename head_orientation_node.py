import torch
import numpy as np
import cv2
import mediapipe as mp

# Definicje kolorów ANSI
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class HeadOrientationNode:
    @classmethod
    def INPUT_TYPES(s):
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
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=True, min_detection_confidence=0.5, max_num_faces=1)
        print(f"{Colors.HEADER}HeadOrientationNode initialized{Colors.ENDC}")

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

    def analyze_orientations(self, images):
        print(f"{Colors.BLUE}[ANALYZE] Starting analysis of {images.shape[0]} images{Colors.ENDC}")
        orientations = []
        for idx in range(images.shape[0]):
            print(f"{Colors.YELLOW}[ANALYZE] Processing image {idx+1}/{images.shape[0]}{Colors.ENDC}")
            img = images[idx].cpu().numpy()
            print(f"{Colors.BLUE}[ANALYZE] Image {idx+1} shape: {img.shape}, dtype: {img.dtype}{Colors.ENDC}")
            
            if img.shape[0] == 3:
                print(f"{Colors.YELLOW}[ANALYZE] Image {idx+1} is in CHW format, transposing to HWC{Colors.ENDC}")
                img = np.transpose(img, (1, 2, 0))
            
            img = (img * 255).astype(np.uint8)
            print(f"{Colors.BLUE}[ANALYZE] Image {idx+1} converted to uint8, shape: {img.shape}, dtype: {img.dtype}{Colors.ENDC}")
            
            if img.shape[2] > 3:
                print(f"{Colors.YELLOW}[ANALYZE] Image {idx+1} has {img.shape[2]} channels, using only first 3{Colors.ENDC}")
                img = img[:, :, :3]
            
            print(f"{Colors.YELLOW}[ANALYZE] Converting image {idx+1} to RGB{Colors.ENDC}")
            image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            print(f"{Colors.YELLOW}[ANALYZE] Processing image {idx+1} with MediaPipe{Colors.ENDC}")
            results = self.face_mesh.process(image_rgb)

            if results.multi_face_landmarks:
                print(f"{Colors.GREEN}[ANALYZE] Face detected in image {idx+1}{Colors.ENDC}")
                face_landmarks = results.multi_face_landmarks[0]
                face_3d = []
                face_2d = []
                for lmk_idx, lm in enumerate(face_landmarks.landmark):
                    if lmk_idx in [33, 263, 1, 61, 291, 199]:
                        x, y = int(lm.x * img.shape[1]), int(lm.y * img.shape[0])
                        face_3d.append([x, y, lm.z])
                        face_2d.append([x, y])

                face_3d = np.array(face_3d, dtype=np.float64)
                face_2d = np.array(face_2d, dtype=np.float64)

                focal_length = 1 * img.shape[1]
                cam_matrix = np.array([[focal_length, 0, img.shape[0] / 2],
                                       [0, focal_length, img.shape[1] / 2],
                                       [0, 0, 1]])
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                print(f"{Colors.YELLOW}[ANALYZE] Calculating head pose for image {idx+1}{Colors.ENDC}")
                success, rot_vec, _ = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
                rmat, _ = cv2.Rodrigues(rot_vec)
                angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
                x, y, z = angles[0] * 360, angles[1] * 360, angles[2] * 360
                orientations.append((x, y, z))
                print(f"{Colors.GREEN}[ANALYZE] Image {idx+1}: Orientation calculated - (x: {x:.2f}, y: {y:.2f}, z: {z:.2f}){Colors.ENDC}")
            else:
                print(f"{Colors.RED}[ANALYZE] No face detected in image {idx+1}{Colors.ENDC}")
                orientations.append((0, 0, 0))

        print(f"{Colors.GREEN}[ANALYZE] Analysis completed for {len(orientations)} images{Colors.ENDC}")
        return orientations

    def sort_images(self, images, input_orientations, reference_orientations):
        print(f"{Colors.BLUE}[SORT] Sorting {len(input_orientations)} input images based on {len(reference_orientations)} reference orientations{Colors.ENDC}")
        sorted_indices = []
        sorted_orientations = []
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

        while len(sorted_indices) < len(reference_orientations):
            sorted_indices.append(sorted_indices[-1])
            sorted_orientations.append(sorted_orientations[-1])
            print(f"{Colors.YELLOW}[SORT] Not enough input images, repeating last image (index {sorted_indices[-1]+1}){Colors.ENDC}")

        print(f"{Colors.BLUE}[SORT] Final sorted indices: {sorted_indices}{Colors.ENDC}")
        sorted_images = images[sorted_indices]
        print(f"{Colors.GREEN}[SORT] Sorted images shape: {sorted_images.shape}{Colors.ENDC}")
        return sorted_images, sorted_orientations

    def format_orientation_data(self, orientations):
        data_output = ""
        for x, y, z in orientations:
            data_output += f"[{x:.2f},{y:.2f},{z:.2f}]\n"
        return data_output

# This line is necessary for ComfyUI to recognize and load your custom node
NODE_CLASS_MAPPINGS = {
    "HeadOrientationNode": HeadOrientationNode
}
