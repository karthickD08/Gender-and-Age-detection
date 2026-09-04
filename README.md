# Age & Gender Detection

AI-based age and gender estimation system using RetinaFace for face detection and facial landmark alignment, combined with an ONNX age-gender model for prediction.

## Features

- Face detection using RetinaFace
- 5-point facial landmark alignment
- Age estimation
- Gender classification
- ONNX Runtime inference
- UTKFace evaluation
- Confusion matrix and performance metrics

## Architecture

```text
Input Image
    ↓
RetinaFace
    ↓
Face Detection + 5 Landmarks
    ↓
Face Alignment
    ↓
224 × 224 Preprocessing
    ↓
Age/Gender ONNX Model
    ↓
Age + Gender Prediction
```

## Tech Stack

- Python
- OpenCV
- ONNX Runtime
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- RetinaFace
- UTKFace Dataset

## Installation

```bash
pip install opencv-python numpy pandas scikit-learn matplotlib onnxruntime
```

## Usage

Run the detection program:

```bash
python age_and_gender_prediction.py
```

For model evaluation:

```bash
python evaluate_model_onnx.py
```

## Final Results

Current best documented experiment:

Dataset                  : UTKFace 800-image evaluation set
Successfully evaluated   : 800/800

Age MAE                  : 3.6443 years
Age RMSE                 : 5.4140 years
Age within ±3 years     : 57.625%
Age within ±5 years     : 74.375%
Age within ±10 years    : 93.000%

Gender accuracy          : 88.12%
Gender precision         : 88.13%
Gender recall            : 88.12%
Gender F1                : 88.12%


## Model

The project uses:

- RetinaFace ONNX for face detection and facial landmarks
- `abhilash88/age-gender-prediction` ONNX model for age and gender prediction

Input size: `224 × 224`

## Results

Evaluated on 800 UTKFace images.

| Metric | Result |
|---|---:|
| Age MAE | 3.64 years |
| Age RMSE | 5.41 years |
| Age ±5 years | 74.38% |
| Gender Accuracy | 88.12% |
| 8-Class Age Accuracy | 78.69% |

## Project Structure

```text
age-and-gender-detection/
│
├── models/
├── retina-face-0.0.18/
├── photo/
├── UTKFace_800/
├── age_and_gender_prediction.py
├── evaluate_model_onnx.py
├── evaluation_results_new_model.csv
├── age_confusion_matrix_new_model.png
├── gender_confusion_matrix_new_model.png
└── README.md
```

## Future Improvements

- Improve age estimation for neighboring age groups
- Optimize ONNX inference speed
- Evaluate on a larger and more diverse dataset
- Improve real-time webcam performance
