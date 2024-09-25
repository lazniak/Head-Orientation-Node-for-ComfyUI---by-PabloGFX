## Head Orientation Node Update version = "1.0.11"

### New Data Output Format

The Head Orientation Node now provides a string output named "data" alongside the sorted images. This output contains information about the head orientation for each processed image.

#### Format Details:
- Each line in the output represents one image's head orientation.
- The format for each line is `[x,y,z]`, where:
  - `x`: Rotation around the X-axis (nodding up and down)
  - `y`: Rotation around the Y-axis (turning left and right)
  - `z`: Rotation around the Z-axis (tilting head side to side)
- All values are in degrees and rounded to two decimal places.
- Each orientation is on a new line.

#### Example Output:

# Head Orientation Node for ComfyUI

## Description

The Head Orientation Node is a custom node for ComfyUI that analyzes and sorts images based on head orientation. It uses the MediaPipe library to detect facial landmarks and calculate head pose, allowing for intelligent image sorting and matching.

Developed by PabloGFX, this node enhances ComfyUI's capabilities in facial analysis and image processing tasks.

## Features

- Detects facial landmarks using MediaPipe
- Calculates head orientation (pitch, yaw, roll) for input images
- Sorts input images based on similarity to reference images' head orientations
- Supports batch processing of multiple images

## Installation

1. Ensure you have ComfyUI installed and set up.
2. Clone this repository into your ComfyUI custom nodes directory:
   ```
   git clone https://github.com/YourGitHubUsername/head-orientation-node.git
   ```
3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Requirements

- numpy>=1.19.3
- opencv-python>=4.5.5.64
- mediapipe>=0.8.9.1
- Pillow>=8.3.1
- torch>=1.9.0

## Usage

1. In ComfyUI, you'll find the node listed as "Head Orientation Node - by PabloGFX" in the node browser.
2. Connect an image or batch of images to the "image" input.
3. Connect a set of reference images to the "reference_images" input.
4. The node will output a sorted batch of images based on head orientation similarity to the reference images.

## How it Works

1. The node analyzes both input and reference images using MediaPipe's face mesh detection.
2. It calculates the head orientation (pitch, yaw, roll) for each detected face.
3. Input images are then sorted to best match the orientations of the reference images.
4. If there are fewer input images than reference images, the last input image is repeated to match the count.

## License

This project is licensed under the Apache License 2.0. This license is compatible with the licenses of the major dependencies used in this project:

- MediaPipe: Apache License 2.0
- OpenCV: Apache License 2.0
- NumPy: BSD 3-Clause License
- PyTorch: BSD 3-Clause License
- Pillow: HPND License

The Apache License 2.0 allows you to use, modify, distribute, and sublicense the code, while also providing an express grant of patent rights from contributors to users. It requires preservation of copyright and license notices.

For the full license text, please see the LICENSE file in the project repository or visit: https://www.apache.org/licenses/LICENSE-2.0

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. By contributing to this project, you agree to license your contributions under the Apache License 2.0.

## Acknowledgements

Special thanks to the developers of MediaPipe, OpenCV, and other open-source libraries that made this project possible.

