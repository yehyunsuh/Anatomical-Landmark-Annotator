# Anatomical-Landmark-Annotator

An interactive Python tool for manually annotating orthopaedic landmarks on paired pre/post-operative images. The tool enforces a fixed multi-stage workflow, shows live geometric overlays while you annotate, computes task-specific measurements directly from the clicked landmarks, and saves both `.csv` results and visualized images.

- For converting dicom files to png files, please visit https://github.com/yehyunsuh/DICOM-to-PNG
- For training, please visit https://github.com/yehyunsuh/Anatomical-Landmark-Detector-Training
- For testing, please visit https://github.com/yehyunsuh/Anatomical-Landmark-Detector-Testing

---

## ⚙️ Environment Setting
Set up the Conda environment and install dependencies:

```bash
git clone https://github.com/yehyunsuh/Anatomical-Landmark-Annotator.git
cd Anatomical-Landmark-Annotator
conda create -n annotator python=3.10 -y
conda activate annotator
pip3 install -r requirements.txt
```

## 📂 Directory Structure
```
Anatomical-Landmark-Annotator/
├── annotator.py                # Main annotation script
├── requirements.txt            # Required Python packages
├── README.md                   # Project documentation
├── input_images/               # Folder of input images to annotate
├── output_images/              # Annotated visual outputs (auto-created)
└── output_annotations/         # CSV files of landmark coordinates (auto-created)
```

## 🚀 Usage
To launch the annotator, run:
```bash
python annotator.py \
    --input input_images \
    --output output_images \
    --output_coordinates output_annotations
```

The annotator groups images by filename using the presence of `pre` or `post` in the filename stem. The matched token becomes the image type, and the remaining filename text is used as the patient identifier.

The tool also keeps a persistent checklist and a persistent CSV per input folder, so completed stages are skipped on later runs unless you intentionally remove them from the checklist.

## 🔁 Workflow
The annotation order is fixed per patient:

1. `pre` image -> `pelvic_tilt`
2. same patient `post` image -> `pelvic_tilt_leg_length`
3. same patient `post` image again -> `cup_anteversion_inclination`

At startup, the tool prints a debug summary showing the planned stage sequence for every detected patient pair.

### Task Click Order

`pre` image, task `pelvic_tilt`

1. `pubic_symphysis`
2. `left_pelvic_teardrop`
3. `right_pelvic_teardrop`

`post` image, task `pelvic_tilt_leg_length`

1. `pubic_symphysis`
2. `left_pelvic_teardrop`
3. `right_pelvic_teardrop`
4. `lesser_trochanter`

`post` image, task `cup_anteversion_inclination`

1. `superior_lateral_end_of_cup`
2. `inferior_medial_end_of_cup`
3. `posterior_lateral_end_of_cup`
4. `left_ischium`
5. `right_ischium`

## 🔧 Command Line Arguments
You can customize the annotation behavior using the following command-line options:

| Argument               | Type    | Description                                                        | Default              |
|------------------------|---------|--------------------------------------------------------------------|----------------------|
| `--input`              | `str`   | Path to the folder containing images to annotate                   | `input_images`       |
| `--output`             | `str`   | Folder to save annotated images with visualized landmarks          | `output_images`      |
| `--output_coordinates` | `str`   | Folder to save the CSV file containing landmark coordinates         | `output_annotations` |
| `--vis_resize`         | `int`   | Maximum pixel size for saved images (preserves aspect ratio)       | `700`                |

## 🖱️ Keyboard & Mouse Controls

Use the following controls during the annotation session:

| Control        | Action                                                |
|----------------|--------------------------------------------------------|
| 🖱️ Left Click  | Add the next required landmark for the active task     |
| ⌨️ `b`         | Undo the most recently placed landmark                 |
| ⌨️ `n`         | Save the current task and proceed to the next stage    |
| ⌨️ `p`         | Go back to the previous annotation stage               |
| ⌨️ `q`         | Quit the annotation session                            |

> ⚠️ Landmarks are saved only when you press `n`. If you quit (`q`) before pressing `n`, current annotations will not be saved.   
> ⚠️ The tool requires the exact number of clicks for the active task before it will advance.  
> ⚠️ When you go back with `p`, that stage can be re-annotated and re-saved.

## ✅ Checklist-Based Progress
The annotator keeps a checklist file for completed stages so you do not need to re-annotate every file on every run.

- Each completed task is recorded as one checklist entry.
- On the next run, stages already listed in the checklist are skipped automatically.
- To re-annotate a stage, remove its line from the checklist file and run the annotator again.
- When a stage is redone, its old CSV row is overwritten in place instead of appending a new row.
- CSV row order always stays aligned with the fixed workflow order.

## 📐 Live Overlays And Measurements
The saved visualizations include the same task geometry used for the measurements.

`pelvic_tilt`
- Trans-teardrop line between left and right teardrops
- Perpendicular from `pubic_symphysis` to the trans-teardrop line
- Saved measurements:
  - `pelvic_tilt_ratio`
  - `pelvic_tilt`

`pelvic_tilt_leg_length`
- Trans-teardrop line between left and right teardrops
- Perpendicular from `pubic_symphysis` to the trans-teardrop line
- Horizontal guide from the selected teardrop to the lesser trochanter x-position
- Horizontal guide from the lesser trochanter to the selected teardrop x-position
- Vertical connector at the midpoint x-value
- Saved measurements:
  - `pelvic_tilt_ratio`
  - `pelvic_tilt`
  - `leg_length`
  - `selected_teardrop`

`cup_anteversion_inclination`
- Cup edge lines from points 1 -> 2 and 2 -> 3
- Ischial reference line from points 4 -> 5
- Extended line from point 1 through point 2 until it intersects the ischial line
- Saved measurements:
  - `cup_anteversion`
  - `cup_inclination`

## 📊 Output
After annotation, two types of output are generated:

📁 Annotated Images

Saved in:
```
output_images/<input_folder_name>/
```
Each image contains:   
- Colored circles marking the landmarks   
- Task-specific click instructions and workflow metadata
- Live geometric overlays aligned with the saved calculations
- A measurement summary once a task is completed

📄 CSV File

Saved in:
```
output_annotations/<input_folder_name>.csv
```
Each row represents one completed task and includes:
```
patient_id, image_name, image_type, task_name, image_width, image_height, n_landmarks,
landmark_1_x, landmark_1_y, ..., landmark_N_x, landmark_N_y,
pelvic_tilt_ratio, pelvic_tilt, cup_anteversion, cup_inclination, leg_length, selected_teardrop
```
Fields that do not apply to a given task are left blank.

📋 Checklist File

Saved in:
```
output_annotations/<input_folder_name>_checklist.txt
```

Each line represents one completed workflow stage:
```
patient_id|image_name|image_type|task_name
```

Delete a line from this file when you want to redo that exact stage. The next run will reopen that task and replace its existing CSV row while keeping the original CSV ordering.

## Citation
If you find this tool helpful, please cite this [paper](https://openreview.net/forum?id=bVC9bi_-t7Y):
```
@inproceedings{
suh2023dilationerosion,
title={Dilation-Erosion Methods for Radiograph Annotation in Total Knee Replacement},
author={Yehyun Suh and Aleksander Mika and J. Ryan Martin and Daniel Moyer},
booktitle={Medical Imaging with Deep Learning, short paper track},
year={2023},
url={https://openreview.net/forum?id=bVC9bi_-t7Y}
}
```
