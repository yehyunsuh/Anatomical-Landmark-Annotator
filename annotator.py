"""
annotator.py

Interactive image annotation tool with visual feedback.

This script allows users to click on images to annotate a fixed number of landmarks.
Annotated results are saved to CSV files, and annotated images are saved with visual legends.

Author: Yehyun Suh
Date: 2025-04-14
Copyright: (c) 2025 Yehyun Suh

Example:
    python annotator.py \
        --input input_images \
        --output output_images \
        --output_coordinates output_annotations \
        --n_clicks 5
"""

import os
import cv2
import csv
import argparse
import numpy as np

from datetime import datetime
from typing import Tuple, List, Optional

# Global variables to store state across image navigation
clicked_points: List[Tuple[int, int]] = []  # Points clicked on current image
current_index = 0  # Index of current image
image_paths: List[str] = []  # All input image paths
annotation_data: List[List] = []  # Collected annotations for CSV
visualization_dir: Optional[str] = None  # Output directory for visualized images


def generate_colors(n: int) -> List[Tuple[int, int, int]]:
    """
    Generate `n` distinct colors using HSV to BGR conversion.

    Args:
        n (int): Number of distinct colors to generate.

    Returns:
        List[Tuple[int, int, int]]: List of BGR color tuples.
    """
    # Spread hues evenly in HSV space, convert to BGR
    colors = []
    for i in range(n):
        hue = int(179.0 * i / n)
        color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(int(c) for c in color))

    return colors


def compute_circle_radius(image_shape: Tuple[int, int]) -> int:
    """
    Calculate annotation circle radius based on image diagonal size.

    Args:
        image_shape (Tuple[int, int]): Shape of the image as (height, width).

    Returns:
        int: Calculated circle radius.
    """
    h, w = image_shape[:2]
    diag = np.sqrt(h ** 2 + w ** 2)

    return max(2, int(diag * 0.005))


def resize_if_needed(image: np.ndarray, max_size: int = 1000) -> Tuple[np.ndarray, float]:
    """
    Resize image if either dimension exceeds max_size, preserving aspect ratio.

    Args:
        image (np.ndarray): Input image.
        max_size (int): Maximum allowed dimension.

    Returns:
        Tuple[np.ndarray, float]: Resized image and scale factor.
    """
    h, w = image.shape[:2]
    if h <= max_size and w <= max_size:
        return image, 1.0
    
    scale = max_size / max(h, w)
    resized = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    return resized, scale


def click_event(event, x, y, flags, params) -> None:
    """
    Handle mouse click events. Add point if under max_clicks, then redraw image.

    Args:
        event: OpenCV mouse event type.
        x (int): X-coordinate of click.
        y (int): Y-coordinate of click.
        flags: Flags provided by OpenCV.
        params (dict): Parameters containing image context.
    """
    global clicked_points

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_points) < params['max_clicks']:
            clicked_points.append((x, y))
            # print(f"Point clicked: ({x}, {y})")
            redraw_image(params)
        else:
            print("Maximum number of clicks reached. Further clicks are ignored.")


def draw_annotation(image: np.ndarray, points: List[Tuple[int, int]], colors: List[Tuple[int, int, int]], radius: int) -> None:
    """
    Draw colored annotation circles sorted by y-position on the image.

    Args:
        image (np.ndarray): Image to annotate.
        points (List[Tuple[int, int]]): List of clicked points.
        colors (List[Tuple[int, int, int]]): Colors for each point.
        radius (int): Radius of the circle to draw.
    """
    sorted_points = sorted(points, key=lambda pt: pt[1])  # Sort vertically for consistent color assignment
    for i, pt in enumerate(sorted_points):
        color = colors[i % len(colors)]
        cv2.circle(image, pt, radius, color, -1)


def draw_legend(image: np.ndarray, width: int, colors: List[Tuple[int, int, int]], font_scale: float) -> None:
    """
    Draw a colored legend with labels for each landmark on the right side of image.

    Args:
        image (np.ndarray): Image to draw on.
        width (int): Width of original image (used for positioning).
        colors (List[Tuple[int, int, int]]): Color palette.
        font_scale (float): Scale for text and markers.
    """
    # Control layout of legend items
    line_spacing = int(45 * font_scale)
    top_padding = int(50 * font_scale)
    legend_radius = int(8 * font_scale)

    for i, color in enumerate(colors):
        y = top_padding + i * line_spacing
        x_text = width + 20
        label = f'Landmark {i+1}'
        
        cv2.putText(image, label, (x_text + 25, y + 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
        cv2.circle(image, (x_text, y - 10), legend_radius, color, -1)


def redraw_image(params: dict) -> None:
    """
    Redraw the image including all annotations and updated legend.

    Args:
        params (dict): Parameters including image, radius, colors, and base_image.
    """
    base_image = params['base_image'].copy()
    h, w = base_image.shape[:2]

    # Scale fonts and padding based on image size
    font_scale = max(0.5, min(w, h) / 1500)
    pad_width = int(300 * font_scale)
    padded_image = cv2.copyMakeBorder(base_image, 0, 0, 0, pad_width, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    draw_legend(padded_image, w, params['colors'], font_scale)
    draw_annotation(padded_image, clicked_points, params['colors'], params['radius'])

    params['image'][:] = padded_image
    cv2.imshow("Image", params['image'])


def show_image(image_path: str, max_clicks: int, colors: List[Tuple[int, int, int]], vis_resize: int) -> Tuple[str, Optional[int], Optional[int], Optional[int]]:
    """
    Show image with live annotation capability. Waits for user input.

    Args:
        image_path (str): Path to image.
        max_clicks (int): Number of points to annotate.
        colors (List[Tuple[int, int, int]]): Landmark colors.
        vis_resize (int): Resize limit for saved image.

    Returns:
        Tuple[str, Optional[int], Optional[int], Optional[int]]:
            Action (str): 'next', 'prev', or 'quit'.
            Image width, height, and radius if action was 'next'.
    """
    global clicked_points

    clicked_points = []

    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read image: {image_path}")
        return 'next', None, None, None

    h, w = image.shape[:2]

    radius = compute_circle_radius(image.shape)
    font_scale = max(0.5, min(w, h) / 1500)
    pad_width = int(300 * font_scale)

    # Prepare padded image for drawing
    base_image = image.copy()
    padded_image = cv2.copyMakeBorder(base_image, 0, 0, 0, pad_width, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    draw_legend(padded_image, w, colors, font_scale)
    display_image = padded_image.copy()

    cv2.imshow("Image", display_image)

    # Store display parameters for callbacks
    params = {
        'image': display_image,
        'base_image': base_image,
        'max_clicks': max_clicks,
        'radius': radius,
        'colors': colors
    }
    cv2.setMouseCallback("Image", click_event, param=params)

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('n'):
            save_annotation(image_path, max_clicks, w, h, radius, vis_resize, colors)
            return 'next', w, h, radius
        elif key == ord('p'):
            return 'prev', None, None, None
        elif key == ord('q'):
            print("Exiting annotation session.")
            return 'quit', None, None, None
        elif key == ord('b'):
            if clicked_points:
                removed = clicked_points.pop()
                print(f"Removed point: {removed}")
                redraw_image(params)
            else:
                print("No points to remove.")
        else:
            print("Invalid key. Use only: n (next), p (prev), b (undo), q (quit).")


def save_annotation(image_path: str, max_clicks: int, width: int, height: int, radius: int, vis_resize: int, colors: List[Tuple[int, int, int]]) -> None:
    """
    Save clicked coordinates to annotation list and output visualized annotated image.

    Args:
        image_path (str): Original image path.
        max_clicks (int): Expected number of landmarks.
        width (int): Image width.
        height (int): Image height.
        radius (int): Circle radius.
        vis_resize (int): Resize limit for output.
        colors (List[Tuple[int, int, int]]): Color palette.
    """
    global annotation_data, clicked_points, visualization_dir

    base_name = os.path.basename(image_path)

    # Remove previous annotations for this image if any
    annotation_data[:] = [row for row in annotation_data if row[0] != base_name]

    flattened_points = [coord for pt in clicked_points for coord in pt]
    n_landmarks = len(clicked_points)

    # Pad missing landmarks with empty strings
    while len(flattened_points) < max_clicks * 2:
        flattened_points.append("")

    row = [base_name, width, height, n_landmarks] + flattened_points
    annotation_data.append(row)

    # Save visualized image
    image = cv2.imread(image_path)
    original_h, original_w = image.shape[:2]
    font_scale = max(0.5, min(original_w, original_h) / 1500)
    pad_width = int(300 * font_scale)
    padded_image = cv2.copyMakeBorder(image, 0, 0, 0, pad_width, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    draw_legend(padded_image, original_w, colors, font_scale)
    draw_annotation(padded_image, clicked_points, colors, radius)

    padded_image_resized, _ = resize_if_needed(padded_image, max_size=vis_resize)
    output_path = os.path.join(visualization_dir, base_name)
    cv2.imwrite(output_path, padded_image_resized)


def main(args):
    """
    Main function to launch annotation session and save results.

    Args:
        args: Parsed command line arguments.
    """
    global image_paths, current_index, annotation_data, visualization_dir

    # Create unique timestamped filenames for output
    input_dir_name = os.path.basename(os.path.abspath(args.input))
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(args.output_coordinates, f"{input_dir_name}_{timestamp}.csv")
    visualization_dir = os.path.join(args.output, f"{input_dir_name}_{timestamp}")
    os.makedirs(visualization_dir, exist_ok=True)

    # Filter image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    image_paths = [
        os.path.join(args.input, f)
        for f in sorted(os.listdir(args.input))
        if os.path.splitext(f)[1].lower() in image_extensions
    ]
    if not image_paths:
        print("No images found in the input directory.")
        return

    colors = generate_colors(args.n_clicks)

    # Loop through images
    while 0 <= current_index < len(image_paths):
        print(f'Annotating image ({current_index + 1}/{len(image_paths)}): {os.path.basename(image_paths[current_index])}')
        action, w, h, r = show_image(image_paths[current_index], args.n_clicks, colors, args.vis_resize)
        if action == 'next':
            current_index += 1
        elif action == 'prev':
            current_index = max(0, current_index - 1)
        elif action == 'quit':
            break

    cv2.destroyAllWindows()

    # Save annotations to CSV
    if annotation_data:
        header = ['image_name', 'image_width', 'image_height', 'n_landmarks']
        for i in range(args.n_clicks):
            header += [f'landmark_{i+1}_x', f'landmark_{i+1}_y']

        os.makedirs(args.output_coordinates, exist_ok=True)
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(annotation_data)

        print(f"\n✅ All annotations saved to: {output_file}")
        print(f"🖼️ Visualizations saved to: {visualization_dir}")
    else:
        print("\n⚠️ No annotations were saved.")


if __name__ == "__main__":
    # Command-line interface for configuration
    parser = argparse.ArgumentParser(description="Interactive image annotation tool with live legend display.")
    parser.add_argument('--input', type=str, default='input_images', help='Input image directory path')
    parser.add_argument('--output', type=str, default='output_images', help='Output image directory for visualizations')
    parser.add_argument('--output_coordinates', type=str, default='output_annotations', help='Output CSV directory')
    parser.add_argument('--n_clicks', type=int, default=5, help='Number of landmarks per image')
    parser.add_argument('--vis_resize', type=int, default=700, help='Resize maximum for saved images')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ Input directory {args.input} does not exist.")
        exit(1)

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.output_coordinates, exist_ok=True)

    main(args)