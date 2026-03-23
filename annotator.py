"""
annotator.py

Interactive image annotation tool for paired pre/post images with task-specific workflows.

This script groups images into patient-level pre/post pairs using filenames that contain
`pre` or `post`, enforces a fixed annotation stage order, saves one CSV row per completed
task, computes task-specific measurements, and exports task-specific visualization images
with live geometric overlays.

Author: Yehyun Suh
Date: 2026-03-22
Copyright: (c) 2025 Yehyun Suh

Example:
    python annotator.py \
        --input input_images \
        --output output_images \
        --output_coordinates output_annotations
"""

import argparse
import csv
import math
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


WINDOW_NAME = "Image"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SIDEBAR_MIN_WIDTH = 460
OVERLAY_COLOR = (0, 255, 255)
MEASUREMENT_TEXT_COLOR = (255, 255, 0)

# Global state used by the OpenCV mouse callback for the currently active stage.
clicked_points: List[Tuple[int, int]] = []


@dataclass(frozen=True)
class TaskDefinition:
    """Defines the ordered click sequence for one annotation task."""

    image_type: str
    task_name: str
    landmark_names: Tuple[str, ...]

    @property
    def n_landmarks(self) -> int:
        """Return the number of required landmarks for the task."""
        return len(self.landmark_names)


@dataclass(frozen=True)
class WorkflowStage:
    """Represents one required annotation stage for a patient image."""

    patient_id: str
    image_path: str
    image_name: str
    image_type: str
    task_name: str
    landmark_names: Tuple[str, ...]

    @property
    def n_landmarks(self) -> int:
        """Return the number of required landmarks for this stage."""
        return len(self.landmark_names)

    @property
    def stage_key(self) -> Tuple[str, str, str]:
        """Return a stable key for indexing saved annotations by patient/type/task."""
        return (self.patient_id, self.image_type, self.task_name)

    @property
    def checklist_id(self) -> str:
        """Return a human-editable identifier used in the checklist file."""
        return f"{self.patient_id}|{self.image_name}|{self.image_type}|{self.task_name}"


@dataclass
class MeasurementResult:
    """Stores task-specific derived measurements for one annotation stage."""

    pelvic_tilt_ratio: Optional[float] = None
    perpendicular_distance: Optional[float] = None
    trans_teardrop_length: Optional[float] = None
    cup_inclination: Optional[float] = None
    cup_anteversion: Optional[float] = None
    leg_length: Optional[float] = None
    selected_teardrop: str = ""


@dataclass
class AnnotationRecord:
    """Stores the completed coordinates, image metadata, and measurements for one workflow stage."""

    patient_id: str
    image_name: str
    image_type: str
    task_name: str
    image_width: int
    image_height: int
    n_landmarks: int
    points: List[Tuple[int, int]]
    measurements: MeasurementResult = field(default_factory=MeasurementResult)


def get_task_definitions() -> Tuple[TaskDefinition, ...]:
    """
    Return the fixed task definitions for the paired orthopaedic workflow.

    Returns:
        Tuple[TaskDefinition, ...]: Ordered task definitions.
    """
    return (
        TaskDefinition(
            image_type="pre",
            task_name="pelvic_tilt",
            landmark_names=(
                "pubic_symphysis",
                "left_pelvic_teardrop",
                "right_pelvic_teardrop",
            ),
        ),
        TaskDefinition(
            image_type="post",
            task_name="pelvic_tilt_leg_length",
            landmark_names=(
                "pubic_symphysis",
                "left_pelvic_teardrop",
                "right_pelvic_teardrop",
                "lesser_trochanter",
            ),
        ),
        TaskDefinition(
            image_type="post",
            task_name="cup_anteversion_inclination",
            landmark_names=(
                "superior_lateral_end_of_cup",
                "inferior_medial_end_of_cup",
                "posterior_lateral_end_of_cup",
                "left_ischium",
                "right_ischium",
            ),
        ),
    )


def generate_colors(n: int) -> List[Tuple[int, int, int]]:
    """
    Generate `n` distinct colors using HSV to BGR conversion.

    Args:
        n (int): Number of distinct colors to generate.

    Returns:
        List[Tuple[int, int, int]]: List of BGR color tuples.
    """
    colors: List[Tuple[int, int, int]] = []
    for i in range(n):
        hue = int(179.0 * i / max(n, 1))
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
    diagonal = np.sqrt(h ** 2 + w ** 2)
    return max(2, int(diagonal * 0.005))


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


def is_image_file(filename: str) -> bool:
    """
    Check whether a filename has a supported image extension.

    Args:
        filename (str): Filename to inspect.

    Returns:
        bool: True when the file extension is supported.
    """
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS


def parse_patient_and_image_type(filename: str) -> Optional[Tuple[str, str]]:
    """
    Extract patient identifier and image type from a filename.

    Rule used:
        - Find the first case-insensitive occurrence of `pre` or `post` in the filename stem.
        - Use that match as `image_type`.
        - Remove that matched token from the stem and trim surrounding separators (`_`, `-`, space)
          to derive `patient_id`.

    This function intentionally does not assume any additional filename structure.

    Args:
        filename (str): Filename to parse.

    Returns:
        Optional[Tuple[str, str]]: Parsed `(patient_id, image_type)` or None when parsing fails.
    """
    stem = os.path.splitext(os.path.basename(filename))[0]
    match = re.search(r"(pre|post)", stem, flags=re.IGNORECASE)
    if match is None:
        return None

    image_type = match.group(1).lower()
    patient_stem = f"{stem[:match.start()]}{stem[match.end():]}"
    patient_id = re.sub(r"[_\-\s]+", " ", patient_stem).strip()
    if not patient_id:
        return None

    return patient_id, image_type


def collect_patient_images(input_dir: str) -> Dict[str, Dict[str, str]]:
    """
    Group images into patient-level pre/post pairs.

    Args:
        input_dir (str): Directory containing input images.

    Returns:
        Dict[str, Dict[str, str]]: Mapping of patient_id to available `pre` and `post` image paths.
    """
    patient_images: Dict[str, Dict[str, str]] = {}

    for filename in sorted(os.listdir(input_dir)):
        if not is_image_file(filename):
            continue

        parsed = parse_patient_and_image_type(filename)
        if parsed is None:
            print(f"Skipping {filename}: filename must contain `pre` or `post` and a recoverable patient identifier.")
            continue

        patient_id, image_type = parsed
        patient_entry = patient_images.setdefault(patient_id, {})
        image_path = os.path.join(input_dir, filename)

        if image_type in patient_entry:
            print(
                f"Skipping {filename}: duplicate `{image_type}` image for patient `{patient_id}`. "
                f"Existing file is {os.path.basename(patient_entry[image_type])}."
            )
            continue

        patient_entry[image_type] = image_path

    return patient_images


def build_workflow_stages(patient_images: Dict[str, Dict[str, str]]) -> List[WorkflowStage]:
    """
    Build the full ordered annotation workflow across all complete patient pairs.

    Args:
        patient_images (Dict[str, Dict[str, str]]): Grouped patient image mapping.

    Returns:
        List[WorkflowStage]: Ordered workflow stages.
    """
    task_definitions = get_task_definitions()
    stages: List[WorkflowStage] = []

    for patient_id in sorted(patient_images):
        image_map = patient_images[patient_id]
        if "pre" not in image_map or "post" not in image_map:
            print(
                f"Skipping patient `{patient_id}`: complete pair required. "
                f"Found image types: {sorted(image_map.keys())}."
            )
            continue

        for task_definition in task_definitions:
            image_path = image_map[task_definition.image_type]
            stages.append(
                WorkflowStage(
                    patient_id=patient_id,
                    image_path=image_path,
                    image_name=os.path.basename(image_path),
                    image_type=task_definition.image_type,
                    task_name=task_definition.task_name,
                    landmark_names=task_definition.landmark_names,
                )
            )

    return stages


def print_workflow_summary(stages: Sequence[WorkflowStage]) -> None:
    """
    Print the planned annotation sequence for each patient at startup.

    Args:
        stages (Sequence[WorkflowStage]): Ordered workflow stages.
    """
    if not stages:
        return

    stages_by_patient: Dict[str, List[WorkflowStage]] = {}
    for stage in stages:
        stages_by_patient.setdefault(stage.patient_id, []).append(stage)

    print("Planned annotation sequence:")
    for patient_id in sorted(stages_by_patient):
        print(f"  Patient: {patient_id}")
        for stage in stages_by_patient[patient_id]:
            print(f"    {stage.image_type} -> {stage.task_name}")


def get_next_click_instruction(stage: WorkflowStage, points: Sequence[Tuple[int, int]]) -> str:
    """
    Return the next required landmark instruction for the active stage.

    Args:
        stage (WorkflowStage): Active workflow stage.
        points (Sequence[Tuple[int, int]]): Current clicked points.

    Returns:
        str: Human-readable next-click instruction.
    """
    if len(points) >= stage.n_landmarks:
        return "All required landmarks captured. Press `n` to continue."

    next_index = len(points)
    return f"Next landmark ({next_index + 1}/{stage.n_landmarks}): {stage.landmark_names[next_index]}"


def as_point_array(point: Tuple[int, int]) -> np.ndarray:
    """
    Convert an integer point tuple to a float NumPy vector.

    Args:
        point (Tuple[int, int]): Point coordinates.

    Returns:
        np.ndarray: Float vector representation of the point.
    """
    return np.array(point, dtype=float)


def point_from_array(point: np.ndarray) -> Tuple[int, int]:
    """
    Convert a float NumPy vector to an integer point tuple.

    Args:
        point (np.ndarray): Float vector point.

    Returns:
        Tuple[int, int]: Rounded integer point.
    """
    return int(round(float(point[0]))), int(round(float(point[1])))


def euclidean_distance(point_a: Tuple[int, int], point_b: Tuple[int, int]) -> float:
    """
    Compute Euclidean distance between two points.

    Args:
        point_a (Tuple[int, int]): First point.
        point_b (Tuple[int, int]): Second point.

    Returns:
        float: Euclidean distance.
    """
    return float(np.linalg.norm(as_point_array(point_a) - as_point_array(point_b)))


def perpendicular_projection(
    point: Tuple[int, int],
    line_start: Tuple[int, int],
    line_end: Tuple[int, int],
) -> Optional[Tuple[int, int]]:
    """
    Project a point perpendicularly onto a line defined by two points.

    Args:
        point (Tuple[int, int]): Point to project.
        line_start (Tuple[int, int]): First point on the line.
        line_end (Tuple[int, int]): Second point on the line.

    Returns:
        Optional[Tuple[int, int]]: Projected point on the infinite line, or None for a degenerate line.
    """
    point_vec = as_point_array(point)
    start_vec = as_point_array(line_start)
    end_vec = as_point_array(line_end)
    line_vec = end_vec - start_vec
    denominator = float(np.dot(line_vec, line_vec))
    if denominator == 0.0:
        return None

    scale = float(np.dot(point_vec - start_vec, line_vec) / denominator)
    projection = start_vec + scale * line_vec
    return point_from_array(projection)


def distance_point_to_line(
    point: Tuple[int, int],
    line_start: Tuple[int, int],
    line_end: Tuple[int, int],
) -> Optional[float]:
    """
    Compute perpendicular distance from a point to the infinite line through two points.

    Args:
        point (Tuple[int, int]): Point to measure from.
        line_start (Tuple[int, int]): First line point.
        line_end (Tuple[int, int]): Second line point.

    Returns:
        Optional[float]: Perpendicular distance, or None for a degenerate line.
    """
    projection = perpendicular_projection(point, line_start, line_end)
    if projection is None:
        return None
    return euclidean_distance(point, projection)


def angle_between_lines(
    line_a_start: Tuple[int, int],
    line_a_end: Tuple[int, int],
    line_b_start: Tuple[int, int],
    line_b_end: Tuple[int, int],
) -> Optional[float]:
    """
    Compute the angle in degrees between two lines.

    Args:
        line_a_start (Tuple[int, int]): First point on line A.
        line_a_end (Tuple[int, int]): Second point on line A.
        line_b_start (Tuple[int, int]): First point on line B.
        line_b_end (Tuple[int, int]): Second point on line B.

    Returns:
        Optional[float]: Angle in degrees, or None for a degenerate line.
    """
    vector_a = as_point_array(line_a_end) - as_point_array(line_a_start)
    vector_b = as_point_array(line_b_end) - as_point_array(line_b_start)
    magnitude_a = float(np.linalg.norm(vector_a))
    magnitude_b = float(np.linalg.norm(vector_b))
    if magnitude_a == 0.0 or magnitude_b == 0.0:
        return None

    cosine = float(np.dot(vector_a, vector_b) / (magnitude_a * magnitude_b))
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def angle_at_vertex(
    vertex: Tuple[int, int],
    point_a: Tuple[int, int],
    point_b: Tuple[int, int],
) -> Optional[float]:
    """
    Compute the angle in degrees formed at a vertex by two segments.

    Args:
        vertex (Tuple[int, int]): Shared vertex point.
        point_a (Tuple[int, int]): Endpoint of the first segment.
        point_b (Tuple[int, int]): Endpoint of the second segment.

    Returns:
        Optional[float]: Angle in degrees, or None for a degenerate segment.
    """
    return angle_between_lines(vertex, point_a, vertex, point_b)


def line_extension_to_line_intersection(
    through_start: Tuple[int, int],
    through_end: Tuple[int, int],
    target_start: Tuple[int, int],
    target_end: Tuple[int, int],
) -> Optional[Tuple[int, int]]:
    """
    Extend one line through two points until it intersects another infinite line.

    Args:
        through_start (Tuple[int, int]): Start point of the line being extended.
        through_end (Tuple[int, int]): Second point defining the direction of extension.
        target_start (Tuple[int, int]): First point of the target line.
        target_end (Tuple[int, int]): Second point of the target line.

    Returns:
        Optional[Tuple[int, int]]: Intersection point, or None when the lines are parallel.
    """
    p = as_point_array(through_start)
    r = as_point_array(through_end) - p
    q = as_point_array(target_start)
    s = as_point_array(target_end) - q

    denominator = float(r[0] * s[1] - r[1] * s[0])
    if abs(denominator) < 1e-9:
        return None

    q_minus_p = q - p
    t = float((q_minus_p[0] * s[1] - q_minus_p[1] * s[0]) / denominator)
    intersection = p + t * r
    return point_from_array(intersection)


def select_nearest_teardrop(
    lesser_trochanter: Tuple[int, int],
    left_teardrop: Tuple[int, int],
    right_teardrop: Tuple[int, int],
) -> Tuple[str, Tuple[int, int]]:
    """
    Select the teardrop nearest to the clicked lesser trochanter.

    Args:
        lesser_trochanter (Tuple[int, int]): Lesser trochanter point.
        left_teardrop (Tuple[int, int]): Left teardrop point.
        right_teardrop (Tuple[int, int]): Right teardrop point.

    Returns:
        Tuple[str, Tuple[int, int]]: Label and coordinates of the nearer teardrop.
    """
    left_distance = euclidean_distance(lesser_trochanter, left_teardrop)
    right_distance = euclidean_distance(lesser_trochanter, right_teardrop)
    if left_distance <= right_distance:
        return "left_pelvic_teardrop", left_teardrop
    return "right_pelvic_teardrop", right_teardrop


def compute_pelvic_tilt_measurement(
    pubic_symphysis: Tuple[int, int],
    left_teardrop: Tuple[int, int],
    right_teardrop: Tuple[int, int],
) -> MeasurementResult:
    """
    Compute pelvic tilt measurements from the pubic symphysis and teardrop landmarks.

    Args:
        pubic_symphysis (Tuple[int, int]): Pubic symphysis point.
        left_teardrop (Tuple[int, int]): Left pelvic teardrop point.
        right_teardrop (Tuple[int, int]): Right pelvic teardrop point.

    Returns:
        MeasurementResult: Pelvic tilt length, perpendicular distance, and ratio.
    """
    trans_teardrop_length = euclidean_distance(left_teardrop, right_teardrop)
    perpendicular_distance = distance_point_to_line(pubic_symphysis, left_teardrop, right_teardrop)

    ratio: Optional[float] = None
    if perpendicular_distance is not None and trans_teardrop_length > 0.0:
        ratio = perpendicular_distance / trans_teardrop_length

    return MeasurementResult(
        pelvic_tilt_ratio=ratio,
        perpendicular_distance=perpendicular_distance,
        trans_teardrop_length=trans_teardrop_length,
    )


def compute_cup_measurements(
    superior_lateral_end_of_cup: Tuple[int, int],
    inferior_medial_end_of_cup: Tuple[int, int],
    posterior_lateral_end_of_cup: Tuple[int, int],
    left_ischium: Tuple[int, int],
    right_ischium: Tuple[int, int],
) -> MeasurementResult:
    """
    Compute cup inclination and cup anteversion from the five post-operative cup landmarks.

    Args:
        superior_lateral_end_of_cup (Tuple[int, int]): Superior lateral cup point.
        inferior_medial_end_of_cup (Tuple[int, int]): Inferior medial cup point.
        posterior_lateral_end_of_cup (Tuple[int, int]): Posterior lateral cup point.
        left_ischium (Tuple[int, int]): Left ischium point.
        right_ischium (Tuple[int, int]): Right ischium point.

    Returns:
        MeasurementResult: Cup inclination and cup anteversion.
    """
    inclination = angle_between_lines(
        superior_lateral_end_of_cup,
        inferior_medial_end_of_cup,
        left_ischium,
        right_ischium,
    )
    if inclination is not None and inclination > 90.0:
        inclination = 180.0 - inclination

    beta_degrees = angle_at_vertex(
        inferior_medial_end_of_cup,
        superior_lateral_end_of_cup,
        posterior_lateral_end_of_cup,
    )

    anteversion: Optional[float] = None
    if beta_degrees is not None:
        beta_radians = math.radians(beta_degrees)
        anteversion = math.degrees(math.asin(float(np.clip(math.tan(beta_radians), -1.0, 1.0))))

    return MeasurementResult(
        cup_inclination=inclination,
        cup_anteversion=anteversion,
    )


def compute_pelvic_tilt_leg_length_measurement(
    pubic_symphysis: Tuple[int, int],
    left_teardrop: Tuple[int, int],
    right_teardrop: Tuple[int, int],
    lesser_trochanter: Tuple[int, int],
) -> MeasurementResult:
    """
    Compute pelvic tilt ratio plus leg length from the combined post-operative task.

    Args:
        pubic_symphysis (Tuple[int, int]): Pubic symphysis point.
        left_teardrop (Tuple[int, int]): Left pelvic teardrop point.
        right_teardrop (Tuple[int, int]): Right pelvic teardrop point.
        lesser_trochanter (Tuple[int, int]): Lesser trochanter point.

    Returns:
        MeasurementResult: Pelvic tilt measurements plus leg length and selected teardrop.
    """
    result = compute_pelvic_tilt_measurement(pubic_symphysis, left_teardrop, right_teardrop)
    selected_teardrop, teardrop_point = select_nearest_teardrop(lesser_trochanter, left_teardrop, right_teardrop)
    result.leg_length = float(abs(teardrop_point[1] - lesser_trochanter[1]))
    result.selected_teardrop = selected_teardrop
    return result


def compute_stage_measurements(stage: WorkflowStage, points: Sequence[Tuple[int, int]]) -> MeasurementResult:
    """
    Compute task-specific measurements for a completed annotation stage.

    Args:
        stage (WorkflowStage): Completed workflow stage.
        points (Sequence[Tuple[int, int]]): Clicked points in task order.

    Returns:
        MeasurementResult: Derived measurements for the task.
    """
    if stage.task_name == "pelvic_tilt":
        return compute_pelvic_tilt_measurement(points[0], points[1], points[2])
    if stage.task_name == "cup_anteversion_inclination":
        return compute_cup_measurements(points[0], points[1], points[2], points[3], points[4])
    if stage.task_name == "pelvic_tilt_leg_length":
        return compute_pelvic_tilt_leg_length_measurement(points[0], points[1], points[2], points[3])
    return MeasurementResult()


def format_measurement_value(value: Optional[float]) -> str:
    """
    Format a measurement value for display and CSV output.

    Args:
        value (Optional[float]): Value to format.

    Returns:
        str: Formatted text or empty string.
    """
    if value is None:
        return ""
    return f"{value:.4f}"


def compute_final_pelvic_tilt_value(pelvic_tilt_ratio: Optional[float]) -> Optional[float]:
    """
    Convert the pelvic tilt ratio to a final tilt angle in degrees for display.

    Args:
        pelvic_tilt_ratio (Optional[float]): Pelvic tilt ratio.

    Returns:
        Optional[float]: Final pelvic tilt angle in degrees, or None when unavailable.
    """
    if pelvic_tilt_ratio is None:
        return None
    return math.degrees(math.atan(pelvic_tilt_ratio))


def build_measurement_display_lines(measurements: MeasurementResult) -> List[str]:
    """
    Build human-readable measurement summary lines for the sidebar and saved visualization.

    Args:
        measurements (MeasurementResult): Task-specific measurement values.

    Returns:
        List[str]: Renderable measurement lines.
    """
    lines: List[str] = []
    if measurements.pelvic_tilt_ratio is not None:
        lines.append(f"Pelvic Tilt Ratio: {measurements.pelvic_tilt_ratio:.4f}")
        final_tilt_value = compute_final_pelvic_tilt_value(measurements.pelvic_tilt_ratio)
        if final_tilt_value is not None:
            lines.append(f"Pelvic Tilt: {final_tilt_value:.2f} deg")
    if measurements.cup_inclination is not None:
        lines.append(f"Cup Inclination: {measurements.cup_inclination:.2f} deg")
    if measurements.cup_anteversion is not None:
        lines.append(f"Cup Anteversion: {measurements.cup_anteversion:.2f} deg")
    if measurements.leg_length is not None:
        lines.append(f"Leg Length: {measurements.leg_length:.2f}")
    return lines


def draw_segment(image: np.ndarray, start: Tuple[int, int], end: Tuple[int, int], color: Tuple[int, int, int], thickness: int) -> None:
    """
    Draw a single line segment.

    Args:
        image (np.ndarray): Target image.
        start (Tuple[int, int]): Start point.
        end (Tuple[int, int]): End point.
        color (Tuple[int, int, int]): BGR line color.
        thickness (int): Line thickness.
    """
    cv2.line(image, start, end, color, thickness)


def draw_pelvic_tilt_overlay(image: np.ndarray, points: Sequence[Tuple[int, int]], thickness: int) -> None:
    """
    Draw the live overlay for the `pelvic_tilt` task.

    Args:
        image (np.ndarray): Target image.
        points (Sequence[Tuple[int, int]]): Current clicked points.
        thickness (int): Line thickness.
    """
    if len(points) < 3:
        return

    pubic_symphysis, left_teardrop, right_teardrop = points[:3]
    projection = perpendicular_projection(pubic_symphysis, left_teardrop, right_teardrop)
    draw_segment(image, left_teardrop, right_teardrop, OVERLAY_COLOR, thickness)
    if projection is not None:
        draw_segment(image, pubic_symphysis, projection, OVERLAY_COLOR, thickness)


def draw_cup_anteversion_inclination_overlay(image: np.ndarray, points: Sequence[Tuple[int, int]], thickness: int) -> None:
    """
    Draw the live overlay for the `cup_anteversion_inclination` task.

    Args:
        image (np.ndarray): Target image.
        points (Sequence[Tuple[int, int]]): Current clicked points.
        thickness (int): Line thickness.
    """
    if len(points) >= 2:
        draw_segment(image, points[0], points[1], OVERLAY_COLOR, thickness)
    if len(points) >= 3:
        draw_segment(image, points[1], points[2], OVERLAY_COLOR, thickness)
    if len(points) >= 5:
        superior_lateral_end = points[0]
        inferior_medial_end = points[1]
        left_ischium = points[3]
        right_ischium = points[4]

        draw_segment(image, left_ischium, right_ischium, OVERLAY_COLOR, thickness)
        intersection = line_extension_to_line_intersection(
            through_start=superior_lateral_end,
            through_end=inferior_medial_end,
            target_start=left_ischium,
            target_end=right_ischium,
        )
        if intersection is not None:
            draw_segment(image, superior_lateral_end, intersection, OVERLAY_COLOR, thickness)


def draw_pelvic_tilt_leg_length_overlay(image: np.ndarray, points: Sequence[Tuple[int, int]], thickness: int) -> None:
    """
    Draw the live overlay for the `pelvic_tilt_leg_length` task.

    Args:
        image (np.ndarray): Target image.
        points (Sequence[Tuple[int, int]]): Current clicked points.
        thickness (int): Line thickness.
    """
    if len(points) >= 3:
        pubic_symphysis, left_teardrop, right_teardrop = points[:3]
        projection = perpendicular_projection(pubic_symphysis, left_teardrop, right_teardrop)
        draw_segment(image, left_teardrop, right_teardrop, OVERLAY_COLOR, thickness)
        if projection is not None:
            draw_segment(image, pubic_symphysis, projection, OVERLAY_COLOR, thickness)

    if len(points) >= 4:
        left_teardrop = points[1]
        right_teardrop = points[2]
        lesser_trochanter = points[3]
        _, nearest_teardrop = select_nearest_teardrop(lesser_trochanter, left_teardrop, right_teardrop)

        teardrop_horizontal_end = (lesser_trochanter[0], nearest_teardrop[1])
        trochanter_horizontal_end = (nearest_teardrop[0], lesser_trochanter[1])
        midpoint_x = int(round((nearest_teardrop[0] + lesser_trochanter[0]) / 2))
        teardrop_midpoint = (midpoint_x, nearest_teardrop[1])
        trochanter_midpoint = (midpoint_x, lesser_trochanter[1])

        draw_segment(image, nearest_teardrop, teardrop_horizontal_end, OVERLAY_COLOR, thickness)
        draw_segment(image, lesser_trochanter, trochanter_horizontal_end, OVERLAY_COLOR, thickness)
        draw_segment(image, teardrop_midpoint, trochanter_midpoint, OVERLAY_COLOR, thickness)


def draw_task_specific_overlays(
    image: np.ndarray,
    stage: WorkflowStage,
    points: Sequence[Tuple[int, int]],
    thickness: int,
) -> None:
    """
    Dispatch task-specific live geometry overlays.

    Args:
        image (np.ndarray): Target image.
        stage (WorkflowStage): Active workflow stage.
        points (Sequence[Tuple[int, int]]): Current clicked points.
        thickness (int): Line thickness.
    """
    if stage.task_name == "pelvic_tilt":
        draw_pelvic_tilt_overlay(image, points, thickness)
    elif stage.task_name == "cup_anteversion_inclination":
        draw_cup_anteversion_inclination_overlay(image, points, thickness)
    elif stage.task_name == "pelvic_tilt_leg_length":
        draw_pelvic_tilt_leg_length_overlay(image, points, thickness)


def click_event(event, x, y, flags, params) -> None:
    """
    Handle mouse click events for the active workflow stage.

    Args:
        event: OpenCV mouse event type.
        x (int): X-coordinate of click.
        y (int): Y-coordinate of click.
        flags: Flags provided by OpenCV.
        params (dict): Parameters containing image context.
    """
    del flags
    global clicked_points

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_points) < params["stage"].n_landmarks:
            clicked_points.append((x, y))
            redraw_image(params)
        else:
            print(f"Maximum number of clicks reached for {params['stage'].task_name}.")


def draw_annotation(
    image: np.ndarray,
    points: Sequence[Tuple[int, int]],
    colors: Sequence[Tuple[int, int, int]],
    landmark_names: Sequence[str],
    radius: int,
) -> None:
    """
    Draw colored landmark points and index labels on the image.

    Args:
        image (np.ndarray): Image to annotate.
        points (Sequence[Tuple[int, int]]): Clicked points.
        colors (Sequence[Tuple[int, int, int]]): Colors for each point.
        landmark_names (Sequence[str]): Task-specific landmark labels.
        radius (int): Point radius.
    """
    for index, point in enumerate(points):
        color = colors[index % len(colors)]
        cv2.circle(image, point, radius, color, -1)
        cv2.putText(
            image,
            str(index + 1),
            (point[0] + radius + 4, point[1] - radius - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
        cv2.putText(
            image,
            landmark_names[index],
            (point[0] + radius + 4, point[1] + radius + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )


def draw_sidebar(
    image: np.ndarray,
    width: int,
    stage: WorkflowStage,
    stage_index: int,
    total_stages: int,
    points: Sequence[Tuple[int, int]],
    colors: Sequence[Tuple[int, int, int]],
    font_scale: float,
    measurements: Optional[MeasurementResult] = None,
) -> None:
    """
    Draw workflow metadata, click instructions, landmark legend, and measurements in the sidebar.

    Args:
        image (np.ndarray): Padded image to draw on.
        width (int): Width of the original image.
        stage (WorkflowStage): Active workflow stage.
        stage_index (int): Zero-based index of the active stage.
        total_stages (int): Total number of stages in the session.
        points (Sequence[Tuple[int, int]]): Current clicked points.
        colors (Sequence[Tuple[int, int, int]]): Color palette for landmarks.
        font_scale (float): Text size scale.
        measurements (Optional[MeasurementResult]): Computed measurements for display.
    """
    x_text = width + 20
    line_height = int(30 * font_scale) + 10
    circle_radius = max(4, int(8 * font_scale))
    points_count = len(points)
    next_instruction = get_next_click_instruction(stage, points)

    info_lines = [
        f"Stage: {stage_index + 1}/{total_stages}",
        f"Patient: {stage.patient_id}",
        f"Image Type: {stage.image_type}",
        f"Task: {stage.task_name}",
        f"Required Landmarks: {stage.n_landmarks}",
        f"Clicked: {points_count}/{stage.n_landmarks}",
        next_instruction,
        "Keys: n next, p previous, b undo, q quit",
    ]

    y = int(35 * font_scale)
    for line in info_lines:
        cv2.putText(
            image,
            line,
            (x_text, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            2,
        )
        y += line_height

    y += int(10 * font_scale)
    cv2.putText(
        image,
        "Click Order",
        (x_text, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        2,
    )
    y += line_height

    for index, landmark_name in enumerate(stage.landmark_names):
        color = colors[index]
        status_prefix = "[x]" if index < points_count else "[ ]"
        circle_center = (x_text + circle_radius, y - circle_radius // 2)
        cv2.circle(image, circle_center, circle_radius, color, -1)
        cv2.putText(
            image,
            f"{status_prefix} {index + 1}. {landmark_name}",
            (x_text + 25, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            2,
        )
        y += line_height

    measurement_lines = build_measurement_display_lines(measurements or MeasurementResult())
    if measurement_lines:
        y += int(10 * font_scale)
        cv2.putText(
            image,
            "Measurements",
            (x_text, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            MEASUREMENT_TEXT_COLOR,
            2,
        )
        y += line_height
        for line in measurement_lines:
            cv2.putText(
                image,
                line,
                (x_text, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                MEASUREMENT_TEXT_COLOR,
                2,
            )
            y += line_height


def build_display_image(
    base_image: np.ndarray,
    stage: WorkflowStage,
    stage_index: int,
    total_stages: int,
    colors: Sequence[Tuple[int, int, int]],
    radius: int,
    points: Sequence[Tuple[int, int]],
    measurements: Optional[MeasurementResult] = None,
) -> np.ndarray:
    """
    Construct the padded display image with task info, instructions, overlays, annotations, and measurements.

    Args:
        base_image (np.ndarray): Original image for the stage.
        stage (WorkflowStage): Active workflow stage.
        stage_index (int): Zero-based stage index.
        total_stages (int): Total number of stages in the session.
        colors (Sequence[Tuple[int, int, int]]): Landmark colors.
        radius (int): Annotation circle radius.
        points (Sequence[Tuple[int, int]]): Current clicked points.
        measurements (Optional[MeasurementResult]): Measurement values derived from `points`.

    Returns:
        np.ndarray: Rendered display image.
    """
    h, w = base_image.shape[:2]
    font_scale = max(0.5, min(w, h) / 1500)
    pad_width = max(SIDEBAR_MIN_WIDTH, int(430 * font_scale))
    line_thickness = max(2, int(radius * 0.75))
    padded_image = cv2.copyMakeBorder(base_image, 0, 0, 0, pad_width, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    draw_task_specific_overlays(padded_image, stage, points, line_thickness)
    draw_annotation(padded_image, points, colors, stage.landmark_names, radius)
    draw_sidebar(
        padded_image,
        w,
        stage,
        stage_index,
        total_stages,
        points,
        colors,
        font_scale,
        measurements=measurements,
    )
    return padded_image


def redraw_image(params: dict) -> None:
    """
    Redraw the active image after annotation changes.

    Args:
        params (dict): Display context for the current workflow stage.
    """
    measurements = None
    if len(clicked_points) == params["stage"].n_landmarks:
        measurements = compute_stage_measurements(params["stage"], clicked_points)

    params["image"] = build_display_image(
        base_image=params["base_image"],
        stage=params["stage"],
        stage_index=params["stage_index"],
        total_stages=params["total_stages"],
        colors=params["colors"],
        radius=params["radius"],
        points=clicked_points,
        measurements=measurements,
    )
    cv2.imshow(WINDOW_NAME, params["image"])


def build_annotation_record(stage: WorkflowStage, width: int, height: int) -> AnnotationRecord:
    """
    Build an annotation record from the current clicked points.

    Args:
        stage (WorkflowStage): Completed workflow stage.
        width (int): Original image width.
        height (int): Original image height.

    Returns:
        AnnotationRecord: Saved annotation metadata, coordinates, and measurements.
    """
    return AnnotationRecord(
        patient_id=stage.patient_id,
        image_name=stage.image_name,
        image_type=stage.image_type,
        task_name=stage.task_name,
        image_width=width,
        image_height=height,
        n_landmarks=len(clicked_points),
        points=list(clicked_points),
        measurements=compute_stage_measurements(stage, clicked_points),
    )


def make_visualization_filename(stage: WorkflowStage) -> str:
    """
    Generate a task-specific visualization filename.

    Args:
        stage (WorkflowStage): Workflow stage being saved.

    Returns:
        str: Visualization filename.
    """
    base_root, extension = os.path.splitext(stage.image_name)
    safe_patient_id = re.sub(r"[^A-Za-z0-9]+", "_", stage.patient_id).strip("_") or "patient"
    safe_task_name = re.sub(r"[^A-Za-z0-9]+", "_", stage.task_name).strip("_")
    return f"{safe_patient_id}__{stage.image_type}__{safe_task_name}__{base_root}{extension}"


def save_visualization(
    stage: WorkflowStage,
    base_image: np.ndarray,
    points: Sequence[Tuple[int, int]],
    colors: Sequence[Tuple[int, int, int]],
    radius: int,
    vis_resize: int,
    visualization_dir: str,
    stage_index: int,
    total_stages: int,
    measurements: MeasurementResult,
) -> str:
    """
    Save a task-specific annotated visualization image.

    Args:
        stage (WorkflowStage): Completed workflow stage.
        base_image (np.ndarray): Original stage image.
        points (Sequence[Tuple[int, int]]): Clicked landmark coordinates.
        colors (Sequence[Tuple[int, int, int]]): Landmark colors.
        radius (int): Circle radius.
        vis_resize (int): Resize limit for output image.
        visualization_dir (str): Output directory for saved visualizations.
        stage_index (int): Zero-based stage index.
        total_stages (int): Total number of stages.
        measurements (MeasurementResult): Computed task-specific measurements.

    Returns:
        str: Path to the saved visualization image.
    """
    rendered_image = build_display_image(
        base_image=base_image,
        stage=stage,
        stage_index=stage_index,
        total_stages=total_stages,
        colors=colors,
        radius=radius,
        points=points,
        measurements=measurements,
    )
    rendered_image_resized, _ = resize_if_needed(rendered_image, max_size=vis_resize)
    output_path = os.path.join(visualization_dir, make_visualization_filename(stage))
    cv2.imwrite(output_path, rendered_image_resized)
    return output_path


def show_stage(
    stage: WorkflowStage,
    stage_index: int,
    total_stages: int,
    prior_points: Optional[List[Tuple[int, int]]],
    vis_resize: int,
    visualization_dir: str,
) -> Tuple[str, Optional[AnnotationRecord]]:
    """
    Show one workflow stage with live annotation and stage-aware navigation.

    Args:
        stage (WorkflowStage): Active workflow stage.
        stage_index (int): Zero-based stage index.
        total_stages (int): Total number of stages.
        prior_points (Optional[List[Tuple[int, int]]]): Previously saved points for this stage.
        vis_resize (int): Resize limit for saved visualizations.
        visualization_dir (str): Visualization output directory.

    Returns:
        Tuple[str, Optional[AnnotationRecord]]: User action and optional saved record.
    """
    global clicked_points

    image = cv2.imread(stage.image_path)
    if image is None:
        print(f"Failed to read image: {stage.image_path}")
        return "next", None

    clicked_points = list(prior_points or [])
    h, w = image.shape[:2]
    radius = compute_circle_radius(image.shape)
    colors = generate_colors(stage.n_landmarks)

    params = {
        "base_image": image.copy(),
        "stage": stage,
        "stage_index": stage_index,
        "total_stages": total_stages,
        "colors": colors,
        "radius": radius,
        "image": None,
    }

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    redraw_image(params)
    cv2.setMouseCallback(WINDOW_NAME, click_event, param=params)

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord("n"):
            if len(clicked_points) != stage.n_landmarks:
                print(
                    f"{stage.task_name} on the {stage.image_type} image requires exactly "
                    f"{stage.n_landmarks} clicks. Current clicks: {len(clicked_points)}. "
                    f"Next required landmark: {stage.landmark_names[len(clicked_points)]}."
                )
                continue

            record = build_annotation_record(stage, w, h)
            save_visualization(
                stage=stage,
                base_image=image,
                points=record.points,
                colors=colors,
                radius=radius,
                vis_resize=vis_resize,
                visualization_dir=visualization_dir,
                stage_index=stage_index,
                total_stages=total_stages,
                measurements=record.measurements,
            )
            return "next", record

        if key == ord("p"):
            return "prev", None

        if key == ord("q"):
            print("Exiting annotation session.")
            return "quit", None

        if key == ord("b"):
            if clicked_points:
                removed = clicked_points.pop()
                print(f"Removed point: {removed}")
                redraw_image(params)
            else:
                print("No points to remove.")
            continue

        print("Invalid key. Use only: n (next), p (prev), b (undo), q (quit).")


def annotation_record_to_row(record: AnnotationRecord, max_landmarks: int) -> List[object]:
    """
    Convert an annotation record to a CSV row with padded coordinates and measurement fields.

    Args:
        record (AnnotationRecord): Completed annotation record.
        max_landmarks (int): Maximum landmark count across all tasks.

    Returns:
        List[object]: CSV row values.
    """
    flattened_points = [coordinate for point in record.points for coordinate in point]
    while len(flattened_points) < max_landmarks * 2:
        flattened_points.append("")

    return [
        record.patient_id,
        record.image_name,
        record.image_type,
        record.task_name,
        record.image_width,
        record.image_height,
        record.n_landmarks,
        *flattened_points,
        format_measurement_value(record.measurements.pelvic_tilt_ratio),
        format_measurement_value(compute_final_pelvic_tilt_value(record.measurements.pelvic_tilt_ratio)),
        format_measurement_value(record.measurements.cup_inclination),
        format_measurement_value(record.measurements.cup_anteversion),
        format_measurement_value(record.measurements.leg_length),
        record.measurements.selected_teardrop,
    ]


def get_csv_header(max_landmarks: int) -> List[str]:
    """
    Build the CSV header for coordinates plus task-specific measurements.

    Args:
        max_landmarks (int): Maximum landmark count across all workflow stages.

    Returns:
        List[str]: Ordered CSV header fields.
    """
    header = [
        "patient_id",
        "image_name",
        "image_type",
        "task_name",
        "image_width",
        "image_height",
        "n_landmarks",
    ]
    for index in range(max_landmarks):
        header.extend([f"landmark_{index + 1}_x", f"landmark_{index + 1}_y"])
    header.extend(
        [
            "pelvic_tilt_ratio",
            "pelvic_tilt",
            "cup_anteversion",
            "cup_inclination",
            "leg_length",
            "selected_teardrop",
        ]
    )
    return header


def parse_optional_float(value: str) -> Optional[float]:
    """
    Parse an optional float field from CSV text.

    Args:
        value (str): CSV text value.

    Returns:
        Optional[float]: Parsed float or None when blank.
    """
    if value == "":
        return None
    return float(value)


def record_from_csv_row(row: Dict[str, str]) -> AnnotationRecord:
    """
    Reconstruct an annotation record from a CSV row.

    Args:
        row (Dict[str, str]): CSV row keyed by header.

    Returns:
        AnnotationRecord: Parsed annotation record.
    """
    n_landmarks = int(row["n_landmarks"])
    points: List[Tuple[int, int]] = []
    for index in range(n_landmarks):
        x_value = row.get(f"landmark_{index + 1}_x", "")
        y_value = row.get(f"landmark_{index + 1}_y", "")
        if x_value == "" or y_value == "":
            continue
        points.append((int(float(x_value)), int(float(y_value))))

    return AnnotationRecord(
        patient_id=row["patient_id"],
        image_name=row["image_name"],
        image_type=row["image_type"],
        task_name=row["task_name"],
        image_width=int(row["image_width"]),
        image_height=int(row["image_height"]),
        n_landmarks=n_landmarks,
        points=points,
        measurements=MeasurementResult(
            pelvic_tilt_ratio=parse_optional_float(row.get("pelvic_tilt_ratio", "")),
            cup_anteversion=parse_optional_float(row.get("cup_anteversion", "")),
            cup_inclination=parse_optional_float(row.get("cup_inclination", "")),
            leg_length=parse_optional_float(row.get("leg_length", "")),
            selected_teardrop=row.get("selected_teardrop", ""),
        ),
    )


def load_existing_records(output_file: str) -> Dict[Tuple[str, str, str], AnnotationRecord]:
    """
    Load existing annotations from a persistent CSV file.

    Args:
        output_file (str): CSV path.

    Returns:
        Dict[Tuple[str, str, str], AnnotationRecord]: Existing records keyed by workflow stage.
    """
    if not os.path.exists(output_file):
        return {}

    existing_records: Dict[Tuple[str, str, str], AnnotationRecord] = {}
    with open(output_file, "r", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            record = record_from_csv_row(row)
            stage_key = (record.patient_id, record.image_type, record.task_name)
            existing_records[stage_key] = record
    return existing_records


def load_checklist_ids(
    checklist_file: str,
    ordered_stages: Sequence[WorkflowStage],
    existing_records: Dict[Tuple[str, str, str], AnnotationRecord],
) -> set[str]:
    """
    Load the checklist of completed stage identifiers.

    If the checklist file does not exist yet, initialize it from the existing CSV rows so
    already-annotated stages are skipped on the next run.

    Args:
        checklist_file (str): Checklist path.
        ordered_stages (Sequence[WorkflowStage]): Full workflow stage order.
        existing_records (Dict[Tuple[str, str, str], AnnotationRecord]): Existing CSV-backed records.

    Returns:
        set[str]: Completed stage identifiers.
    """
    if os.path.exists(checklist_file):
        completed_stage_ids: set[str] = set()
        with open(checklist_file, "r", encoding="utf-8") as checklist_handle:
            for line in checklist_handle:
                checklist_id = line.strip()
                if checklist_id:
                    completed_stage_ids.add(checklist_id)
        return completed_stage_ids

    stage_id_by_key = {stage.stage_key: stage.checklist_id for stage in ordered_stages}
    return {stage_id_by_key[key] for key in existing_records if key in stage_id_by_key}


def write_checklist_file(
    checklist_file: str,
    ordered_stages: Sequence[WorkflowStage],
    completed_stage_ids: set[str],
) -> None:
    """
    Write the checklist file in workflow order.

    Args:
        checklist_file (str): Checklist path.
        ordered_stages (Sequence[WorkflowStage]): Full workflow stage order.
        completed_stage_ids (set[str]): Completed stage identifiers.
    """
    with open(checklist_file, "w", encoding="utf-8") as checklist_handle:
        for stage in ordered_stages:
            if stage.checklist_id in completed_stage_ids:
                checklist_handle.write(f"{stage.checklist_id}\n")


def persist_annotation_state(
    output_file: str,
    checklist_file: str,
    ordered_stages: Sequence[WorkflowStage],
    annotation_records: Dict[Tuple[str, str, str], AnnotationRecord],
    completed_stage_ids: set[str],
) -> None:
    """
    Persist both the CSV and checklist so progress survives partial sessions.

    Args:
        output_file (str): CSV path.
        checklist_file (str): Checklist path.
        ordered_stages (Sequence[WorkflowStage]): Full workflow stage order.
        annotation_records (Dict[Tuple[str, str, str], AnnotationRecord]): Saved records.
        completed_stage_ids (set[str]): Completed stage identifiers.
    """
    write_annotations_csv(output_file, ordered_stages, annotation_records)
    write_checklist_file(checklist_file, ordered_stages, completed_stage_ids)


def write_annotations_csv(
    output_file: str,
    ordered_stages: Sequence[WorkflowStage],
    annotation_records: Dict[Tuple[str, str, str], AnnotationRecord],
) -> None:
    """
    Write completed annotations to a timestamped CSV file in workflow order.

    Args:
        output_file (str): Output CSV path.
        ordered_stages (Sequence[WorkflowStage]): Full ordered workflow stage list.
        annotation_records (Dict[Tuple[str, str, str], AnnotationRecord]): Saved records by stage key.
    """
    max_landmarks = max(stage.n_landmarks for stage in ordered_stages)
    header = get_csv_header(max_landmarks)

    rows: List[List[object]] = []
    for stage in ordered_stages:
        record = annotation_records.get(stage.stage_key)
        if record is not None:
            rows.append(annotation_record_to_row(record, max_landmarks))

    with open(output_file, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        writer.writerows(rows)


def main(args) -> None:
    """
    Launch the patient-paired annotation session and save results.

    Args:
        args: Parsed command line arguments.
    """
    input_dir_name = os.path.basename(os.path.abspath(args.input))
    output_file = os.path.join(args.output_coordinates, f"{input_dir_name}.csv")
    checklist_file = os.path.join(args.output_coordinates, f"{input_dir_name}_checklist.txt")
    visualization_dir = os.path.join(args.output, input_dir_name)

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.output_coordinates, exist_ok=True)
    os.makedirs(visualization_dir, exist_ok=True)

    patient_images = collect_patient_images(args.input)
    workflow_stages = build_workflow_stages(patient_images)

    if not workflow_stages:
        print("No complete pre/post patient pairs were found in the input directory.")
        return

    print_workflow_summary(workflow_stages)

    annotation_records = load_existing_records(output_file)
    completed_stage_ids = load_checklist_ids(checklist_file, workflow_stages, annotation_records)

    pending_stages = [stage for stage in workflow_stages if stage.checklist_id not in completed_stage_ids]
    if not pending_stages:
        print("All workflow stages are already marked complete in the checklist.")
        print(f"Checklist file: {checklist_file}")
        print(f"CSV file: {output_file}")
        return

    current_stage_index = 0

    while 0 <= current_stage_index < len(pending_stages):
        stage = pending_stages[current_stage_index]
        prior_record = annotation_records.get(stage.stage_key)

        print(
            f"Annotating pending stage {current_stage_index + 1}/{len(pending_stages)}: "
            f"patient={stage.patient_id}, image_type={stage.image_type}, task={stage.task_name}, "
            f"required_clicks={stage.n_landmarks}, image={stage.image_name}"
        )

        action, record = show_stage(
            stage=stage,
            stage_index=current_stage_index,
            total_stages=len(pending_stages),
            prior_points=prior_record.points if prior_record is not None else None,
            vis_resize=args.vis_resize,
            visualization_dir=visualization_dir,
        )

        if action == "next":
            if record is not None:
                annotation_records[stage.stage_key] = record
                completed_stage_ids.add(stage.checklist_id)
                persist_annotation_state(
                    output_file=output_file,
                    checklist_file=checklist_file,
                    ordered_stages=workflow_stages,
                    annotation_records=annotation_records,
                    completed_stage_ids=completed_stage_ids,
                )
            current_stage_index += 1
        elif action == "prev":
            current_stage_index = max(0, current_stage_index - 1)
        elif action == "quit":
            break

    cv2.destroyAllWindows()

    if annotation_records:
        persist_annotation_state(
            output_file=output_file,
            checklist_file=checklist_file,
            ordered_stages=workflow_stages,
            annotation_records=annotation_records,
            completed_stage_ids=completed_stage_ids,
        )
        print(f"\nAll annotations saved to: {output_file}")
        print(f"Checklist saved to: {checklist_file}")
        print(f"Visualizations saved to: {visualization_dir}")
    else:
        print("\nNo annotations were saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interactive annotation tool for paired pre/post images with task-specific stages."
    )
    parser.add_argument("--input", type=str, default="input_images", help="Input image directory path")
    parser.add_argument("--output", type=str, default="output_images", help="Output directory for visualizations")
    parser.add_argument("--output_coordinates", type=str, default="output_annotations", help="Output CSV directory")
    parser.add_argument("--vis_resize", type=int, default=700, help="Resize maximum for saved visualization images")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Input directory {args.input} does not exist.")
        raise SystemExit(1)

    main(args)
