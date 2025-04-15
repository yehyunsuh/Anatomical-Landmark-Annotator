# Anatomical-Landmark-Annotator

An interactive Python tool for manually annotating anatomical landmarks on medical or scientific images. Click to place a fixed number of landmarks, view real-time colored markers and legends, and save results as both `.csv` and visualized images.

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
    --output_coordinates output_annotations \
    --n_clicks 5
```

## 🔧 Command Line Arguments
You can customize the annotation behavior using the following command-line options:

| Argument               | Type    | Description                                                        | Default              |
|------------------------|---------|--------------------------------------------------------------------|----------------------|
| `--input`              | `str`   | Path to the folder containing images to annotate                   | `input_images`       |
| `--output`             | `str`   | Folder to save annotated images with visualized landmarks          | `output_images`      |
| `--output_coordinates` | `str`   | Folder to save the CSV file containing landmark coordinates         | `output_annotations` |
| `--n_clicks`           | `int`   | Number of landmarks to annotate per image                          | `5`                  |
| `--vis_resize`         | `int`   | Maximum pixel size for saved images (preserves aspect ratio)       | `700`                |

## 🖱️ Keyboard & Mouse Controls

Use the following controls during the annotation session:

| Control        | Action                                                |
|----------------|--------------------------------------------------------|
| 🖱️ Left Click  | Add a landmark at the clicked location (up to `n_clicks`) |
| ⌨️ `b`         | Undo the most recently placed landmark                 |
| ⌨️ `n`         | Save current annotations and proceed to the next image |
| ⌨️ `p`         | Go back to the previous image                          |
| ⌨️ `q`         | Quit the annotation session                            |

> ⚠️ Landmarks are saved only when you press `n`. If you quit (`q`) before pressing `n`, current annotations will not be saved.   
> ⚠️ When you re-annotate an image that has been annotating using `p`, previous annotation will be deleted.

## 📊 Output
After annotation, two types of output are generated:

📁 Annotated Images

Saved in:
```
output_images/<input_folder_name>_<timestamp>/
```
Each image contains:   
- Colored circles marking the landmarks   
- A dynamically scaled legend on the right   

📄 CSV File

Saved in:
```
output_annotations/<input_folder_name>_<timestamp>.csv
```
Each row represents one image and includes:
```
image_name, image_width, image_height, n_landmarks,
landmark_1_x, landmark_1_y, ..., landmark_N_x, landmark_N_y
```
Missing landmarks (if fewer than n_clicks are placed) are left blank.

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