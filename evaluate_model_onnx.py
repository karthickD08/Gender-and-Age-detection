# ============================================================
# MODEL EVALUATION
# ============================================================
# UTKFace
#
# Pipeline:
#
# UTKFace image
#       ↓
# RetinaFace ONNX
#       ↓
# 5-point facial landmark alignment
#       ↓
# Age/Gender ONNX
#       ↓
# Continuous Age + Gender
#
# Metrics:
#
# AGE:
#   MAE
#   RMSE
#   Within ±3 years
#   Within ±5 years
#   Within ±10 years
#   8-class accuracy
#   8-class precision
#   8-class recall
#   8-class F1
#   Confusion matrix
#
# GENDER:
#   Accuracy
#   Precision
#   Recall
#   F1
#   Confusion matrix
# ============================================================


import cv2 as cv
import numpy as np

import onnxruntime as ort

import math
import time

from pathlib import Path

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error
)


print(
    "\nRUNNING EVALUATOR:",
    __file__
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(
    __file__
).resolve().parent

MODEL_DIR = BASE_DIR / "models"

DATASET_DIR = BASE_DIR / "UTKFace_800"


# ============================================================
# MODEL PATHS
# ============================================================

RETINAFACE_MODEL = (
    MODEL_DIR /
    "retinaface_mv1_0.25.onnx"
)

AGE_GENDER_MODEL = (
    MODEL_DIR /
    "model.onnx"
)


# ============================================================
# AGE GROUPS
# ============================================================

ageList = [
    "(0-2)",
    "(4-6)",
    "(8-12)",
    "(15-20)",
    "(25-32)",
    "(38-43)",
    "(48-53)",
    "(60-100)"
]


genderList = [
    "Male",
    "Female"
]


age_ranges = [
    (0, 2),
    (4, 6),
    (8, 12),
    (15, 20),
    (25, 32),
    (38, 43),
    (48, 53),
    (60, 100)
]


# ============================================================
# AGE -> CLASS
# ============================================================

def age_to_class(age):

    """
    Convert an actual/predicted age into
    one of the eight traditional UTKFace
    age groups.

    If the age falls inside a defined group,
    return that class.

    If the age falls into a gap such as:

        3
        7
        13
        14
        21-24
        33-37
        44-47
        54-59

    return None.

    This is important because those ages do not
    belong to any of the eight defined classes.
    """


    for index, (
        low,
        high
    ) in enumerate(age_ranges):

        if (
            low <= age <= high
        ):

            return index


    return None


# ============================================================
# RETINAFACE CONFIGURATION
# ============================================================

INPUT_SIZE = 640

CONF_THRESHOLD = 0.5

NMS_THRESHOLD = 0.4


MIN_SIZES = [
    [16, 32],
    [64, 128],
    [256, 512]
]


STEPS = [
    8,
    16,
    32
]


VARIANCE = [
    0.1,
    0.2
]


# ============================================================
# LOAD AGE/GENDER ONNX
# ============================================================

print("\n======================================")
print("LOADING AGE/GENDER MODEL")
print("======================================")


print(
    "Model:",
    AGE_GENDER_MODEL
)


# ------------------------------------------------------------
# ONNX Runtime configuration
# ------------------------------------------------------------

age_gender_options = (
    ort.SessionOptions()
)


age_gender_options.graph_optimization_level = (
    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
)


# User has 4 logical CPUs

age_gender_options.intra_op_num_threads = 4

age_gender_options.inter_op_num_threads = 1


age_gender_session = ort.InferenceSession(

    str(
        AGE_GENDER_MODEL
    ),

    sess_options=(
        age_gender_options
    ),

    providers=[
        "CPUExecutionProvider"
    ]
)


age_gender_input_name = (
    age_gender_session
    .get_inputs()[0]
    .name
)


print(
    "Age/Gender ONNX loaded successfully."
)


print(
    "Input name:",
    age_gender_input_name
)


print(
    "Input shape:",
    age_gender_session
    .get_inputs()[0]
    .shape
)


print(
    "Input type:",
    age_gender_session
    .get_inputs()[0]
    .type
)


print("Outputs:")


for output in (
    age_gender_session
    .get_outputs()
):

    print(
        "  Name:",
        output.name,
        "| Shape:",
        output.shape,
        "| Type:",
        output.type
    )


# ============================================================
# LOAD RETINAFACE
# ============================================================

print("\n======================================")
print("LOADING RETINAFACE")
print("======================================")


retina_session = ort.InferenceSession(

    str(
        RETINAFACE_MODEL
    ),

    providers=[
        "CPUExecutionProvider"
    ]
)


retina_input_name = (
    retina_session
    .get_inputs()[0]
    .name
)


print(
    "RetinaFace ONNX loaded successfully."
)


print(
    "Input:",
    retina_input_name
)


print(
    "Outputs:",
    [
        output.name
        for output in (
            retina_session
            .get_outputs()
        )
    ]
)


# ============================================================
# GENERATE RETINAFACE PRIORS
# ============================================================

def generate_priors(
    image_height,
    image_width
):

    priors = []


    feature_maps = [

        [
            math.ceil(
                image_height /
                step
            ),

            math.ceil(
                image_width /
                step
            )
        ]

        for step in STEPS

    ]


    for k, (
        map_height,
        map_width
    ) in enumerate(
        feature_maps
    ):

        step = STEPS[k]


        for i in range(
            map_height
        ):

            for j in range(
                map_width
            ):

                for min_size in (
                    MIN_SIZES[k]
                ):

                    s_kx = (
                        min_size /
                        image_width
                    )

                    s_ky = (
                        min_size /
                        image_height
                    )


                    cx = (
                        (j + 0.5)
                        *
                        step
                        /
                        image_width
                    )


                    cy = (
                        (i + 0.5)
                        *
                        step
                        /
                        image_height
                    )


                    priors.append([
                        cx,
                        cy,
                        s_kx,
                        s_ky
                    ])


    return np.array(
        priors,
        dtype=np.float32
    )


PRIORS = generate_priors(
    INPUT_SIZE,
    INPUT_SIZE
)


print(
    "\nRetinaFace priors:",
    len(PRIORS)
)


# ============================================================
# NMS
# ============================================================

def nms(
    boxes,
    scores,
    threshold
):

    if len(boxes) == 0:

        return np.array(
            [],
            dtype=np.int32
        )


    x1 = boxes[:, 0]

    y1 = boxes[:, 1]

    x2 = boxes[:, 2]

    y2 = boxes[:, 3]


    areas = (
        x2 - x1
    ) * (
        y2 - y1
    )


    order = (
        scores
        .argsort()[::-1]
    )


    keep = []


    while len(order) > 0:

        i = order[0]

        keep.append(i)


        if len(order) == 1:

            break


        xx1 = np.maximum(
            x1[i],
            x1[order[1:]]
        )


        yy1 = np.maximum(
            y1[i],
            y1[order[1:]]
        )


        xx2 = np.minimum(
            x2[i],
            x2[order[1:]]
        )


        yy2 = np.minimum(
            y2[i],
            y2[order[1:]]
        )


        width = np.maximum(
            0,
            xx2 - xx1
        )


        height = np.maximum(
            0,
            yy2 - yy1
        )


        intersection = (
            width *
            height
        )


        union = (
            areas[i]
            +
            areas[order[1:]]
            -
            intersection
        )


        iou = (
            intersection /
            np.maximum(
                union,
                1e-8
            )
        )


        remaining = np.where(
            iou <= threshold
        )[0]


        order = order[
            remaining + 1
        ]


    return np.array(
        keep,
        dtype=np.int32
    )


# ============================================================
# ALIGN FACE
# ============================================================

def align_face(
    image,
    landmarks,
    output_size=(224, 224)
):

    src_points = np.array(
        landmarks,
        dtype=np.float32
    )


    dst_points = np.array([

        [70.0, 85.0],

        [154.0, 85.0],

        [112.0, 125.0],

        [82.0, 165.0],

        [142.0, 165.0]

    ], dtype=np.float32)


    transformation_matrix, _ = (
        cv.estimateAffinePartial2D(

            src_points,

            dst_points,

            method=cv.LMEDS

        )
    )


    if transformation_matrix is None:

        return None


    aligned = cv.warpAffine(

        image,

        transformation_matrix,

        output_size,

        flags=cv.INTER_CUBIC,

        borderMode=cv.BORDER_REPLICATE

    )


    return aligned


# ============================================================
# RETINAFACE DETECTION
# ============================================================

def get_aligned_face(
    image
):

    if (
        image is None
        or
        image.size == 0
    ):

        return None, None


    original_height, original_width = (
        image.shape[:2]
    )


    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    resized = cv.resize(

        image,

        (
            INPUT_SIZE,
            INPUT_SIZE
        ),

        interpolation=cv.INTER_LINEAR

    )


    # --------------------------------------------------------
    # RetinaFace preprocessing
    # --------------------------------------------------------

    input_image = (
        resized.astype(
            np.float32
        )
    )


    input_image -= np.array(

        [
            104.0,
            117.0,
            123.0
        ],

        dtype=np.float32

    )


    input_image = (
        input_image
        .transpose(
            2,
            0,
            1
        )
    )


    input_image = np.expand_dims(
        input_image,
        axis=0
    )


    # --------------------------------------------------------
    # RetinaFace inference
    # --------------------------------------------------------

    outputs = retina_session.run(

        None,

        {
            retina_input_name:
            input_image
        }

    )


    loc = outputs[0].squeeze(0)

    conf = outputs[1].squeeze(0)

    landm = outputs[2].squeeze(0)


    if len(PRIORS) != len(loc):

        raise RuntimeError(

            "RetinaFace prior mismatch: "

            f"{len(PRIORS)} priors vs "

            f"{len(loc)} predictions"

        )


    # --------------------------------------------------------
    # Decode boxes
    # --------------------------------------------------------

    center = (

        PRIORS[:, 0:2]

        +

        loc[:, 0:2]

        *

        VARIANCE[0]

        *

        PRIORS[:, 2:4]

    )


    size = (

        PRIORS[:, 2:4]

        *

        np.exp(

            loc[:, 2:4]

            *

            VARIANCE[1]

        )

    )


    boxes = np.zeros_like(
        loc
    )


    boxes[:, 0:2] = (
        center -
        size / 2
    )


    boxes[:, 2:4] = (
        center +
        size / 2
    )


    # --------------------------------------------------------
    # Decode landmarks
    # --------------------------------------------------------

    landm = landm.reshape(
        -1,
        5,
        2
    )


    decoded_landmarks = (

        PRIORS[:, None, 0:2]

        +

        landm

        *

        VARIANCE[0]

        *

        PRIORS[:, None, 2:4]

    )


    landmarks = (
        decoded_landmarks
        .reshape(-1, 10)
    )


    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    scores = conf[:, 1]


    keep = (
        scores >=
        CONF_THRESHOLD
    )


    boxes = boxes[keep]

    landmarks = landmarks[keep]

    scores = scores[keep]


    if len(boxes) == 0:

        return None, None


    # --------------------------------------------------------
    # Convert to 640x640
    # --------------------------------------------------------

    boxes[:, [0, 2]] *= INPUT_SIZE

    boxes[:, [1, 3]] *= INPUT_SIZE


    landmarks[:, 0::2] *= INPUT_SIZE

    landmarks[:, 1::2] *= INPUT_SIZE


    # --------------------------------------------------------
    # NMS
    # --------------------------------------------------------

    keep = nms(

        boxes,

        scores,

        NMS_THRESHOLD

    )


    boxes = boxes[keep]

    landmarks = landmarks[keep]

    scores = scores[keep]


    if len(boxes) == 0:

        return None, None


    # --------------------------------------------------------
    # Select highest confidence face
    # --------------------------------------------------------

    best_index = int(
        np.argmax(scores)
    )


    points = (
        landmarks[
            best_index
        ]
        .reshape(5, 2)
    )


    # --------------------------------------------------------
    # Convert landmarks to original image
    # --------------------------------------------------------

    points[:, 0] *= (
        original_width /
        INPUT_SIZE
    )


    points[:, 1] *= (
        original_height /
        INPUT_SIZE
    )


    confidence = float(
        scores[best_index]
    )


    # --------------------------------------------------------
    # Alignment
    # --------------------------------------------------------

    aligned = align_face(

        image,

        points

    )


    if aligned is None:

        return None, None


    return aligned, {

        "confidence":
            confidence,

        "landmarks":
            points

    }


# ============================================================
# AGE/GENDER PREPROCESSING
# ============================================================

def preprocess_age_gender(
    face
):

    # --------------------------------------------------------
    # BGR -> RGB
    # --------------------------------------------------------

    rgb_face = cv.cvtColor(

        face,

        cv.COLOR_BGR2RGB

    )


    # --------------------------------------------------------
    # float32
    # --------------------------------------------------------

    image = (
        rgb_face.astype(
            np.float32
        )
    )


    # --------------------------------------------------------
    # Scale 0-255 -> 0-1
    # --------------------------------------------------------

    image /= 255.0


    # --------------------------------------------------------
    # ImageNet normalization
    # --------------------------------------------------------

    mean = np.array(

        [
            0.485,
            0.456,
            0.406
        ],

        dtype=np.float32

    )


    std = np.array(

        [
            0.229,
            0.224,
            0.225
        ],

        dtype=np.float32

    )


    image = (
        image - mean
    ) / std


    # --------------------------------------------------------
    # HWC -> CHW
    # --------------------------------------------------------

    image = image.transpose(

        2,
        0,
        1

    )


    # --------------------------------------------------------
    # Batch dimension
    # --------------------------------------------------------

    image = np.expand_dims(

        image,

        axis=0

    )


    return image.astype(
        np.float32
    )


# ============================================================
# PREDICT AGE + GENDER
# ============================================================

def predict_image(
    frame
):

    if (
        frame is None
        or
        frame.size == 0
    ):

        return None


    # --------------------------------------------------------
    # RetinaFace + alignment
    # --------------------------------------------------------

    aligned_face, detection = (
        get_aligned_face(
            frame
        )
    )


    if aligned_face is None:

        return None


    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    input_tensor = (
        preprocess_age_gender(
            aligned_face
        )
    )


    # --------------------------------------------------------
    # ONNX inference
    # --------------------------------------------------------

    start = time.time()


    outputs = (
        age_gender_session.run(

            None,

            {
                age_gender_input_name:
                input_tensor
            }

        )
    )


    inference_time = (
        time.time() -
        start
    )


    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    output = outputs[0][0]


    predicted_age = float(
        output[0]
    )


    # --------------------------------------------------------
    # Clamp age
    # --------------------------------------------------------

    predicted_age = float(
        np.clip(
            predicted_age,
            0,
            100
        )
    )


    # --------------------------------------------------------
    # Gender
    #
    # IMPORTANT:
    #
    # output[1] is ALREADY the
    # female probability.
    #
    # DO NOT apply sigmoid.
    # --------------------------------------------------------

    female_probability = float(
        output[1]
    )


    # Safety clamp

    female_probability = float(
        np.clip(
            female_probability,
            0.0,
            1.0
        )
    )


    male_probability = (
        1.0 -
        female_probability
    )


    if (
        female_probability >=
        0.5
    ):

        gender = 1

    else:

        gender = 0


    gender_confidence = max(
        female_probability,
        male_probability
    )


    return {

        "age":
            predicted_age,

        "gender":
            gender,

        "gender_confidence":
            gender_confidence,

        "female_probability":
            female_probability,

        "male_probability":
            male_probability,

        "age_confidence":
            None,

        "face_confidence":
            detection[
                "confidence"
            ],

        "inference_time":
            inference_time

    }


# ============================================================
# DATA STORAGE
# ============================================================

true_age = []

pred_age = []


true_gender = []

pred_gender = []


true_age_class = []

pred_age_class = []


results = []


# ============================================================
# SKIP COUNTERS
# ============================================================

skipped_invalid_filename = 0

skipped_read_error = 0

skipped_no_face = 0

skipped_age_class = 0


# ============================================================
# START EVALUATION
# ============================================================

total_start = time.time()


image_files = sorted(DATASET_DIR.glob("*.jpg"))

print("Images found:", len(image_files))


print(
    "\n======================================"
)

print(
    "STARTING EVALUATION"
)

print(
    "======================================"
)


print(
    "Images found:",
    len(image_files)
)


print(
    "Dataset:",
    DATASET_DIR
)


print(
    "Pipeline:"
)


print(
    "UTKFace"
)


print(
    "   -> RetinaFace"
)


print(
    "   -> 5-point alignment"
)


print(
    "   -> Age/Gender ONNX"
)


print(
    "   -> Continuous age + gender"
)


# ============================================================
# EVALUATION LOOP
# ============================================================

for index, image_path in enumerate(
    image_files
):

    filename = (
        image_path.name
    )


    # --------------------------------------------------------
    # Parse filename
    #
    # Example:
    #
    # 25_0_0_20170120133812304.jpg
    #
    # age = 25
    # gender = 0
    # --------------------------------------------------------

    try:

        parts = (
            filename.split("_")
        )


        actual_age = int(
            parts[0]
        )


        actual_gender = int(
            parts[1]
        )


    except Exception:

        skipped_invalid_filename += 1

        continue


    # --------------------------------------------------------
    # Validate gender
    # --------------------------------------------------------

    if actual_gender not in [
        0,
        1
    ]:

        skipped_invalid_filename += 1

        continue


    # --------------------------------------------------------
    # Age class
    # --------------------------------------------------------

    actual_class = age_to_class(
        actual_age
    )


    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    frame = cv.imread(
        str(image_path)
    )


    if frame is None:

        skipped_read_error += 1

        continue


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = predict_image(
        frame
    )


    if prediction is None:

        skipped_no_face += 1

        continue


    # --------------------------------------------------------
    # Extract prediction
    # --------------------------------------------------------

    predicted_age = (
        prediction["age"]
    )


    predicted_gender = (
        prediction["gender"]
    )


    # ========================================================
    # CONTINUOUS AGE
    # ========================================================

    true_age.append(
        actual_age
    )


    pred_age.append(
        predicted_age
    )


    # ========================================================
    # GENDER
    # ========================================================

    true_gender.append(
        actual_gender
    )


    pred_gender.append(
        predicted_gender
    )


    # ========================================================
    # AGE GROUP
    # ========================================================

    predicted_class = age_to_class(
        predicted_age
    )


    # Only compare age groups if BOTH
    # actual and predicted ages belong
    # to one of the eight groups.

    if (
        actual_class is not None
        and
        predicted_class is not None
    ):

        true_age_class.append(
            actual_class
        )

        pred_age_class.append(
            predicted_class
        )

        age_class_valid = True

    else:

        age_class_valid = False

        skipped_age_class += 1


    # ========================================================
    # STORE RESULT
    # ========================================================

    result = {

        "filename":
            filename,

        "actual_age":
            actual_age,

        "predicted_age":
            predicted_age,

        "absolute_age_error":
            abs(
                actual_age -
                predicted_age
            ),

        "actual_gender":
            genderList[
                actual_gender
            ],

        "predicted_gender":
            genderList[
                predicted_gender
            ],

        "gender_correct":
            (
                actual_gender ==
                predicted_gender
            ),

        "female_probability":
            prediction[
                "female_probability"
            ],

        "male_probability":
            prediction[
                "male_probability"
            ],

        "gender_confidence":
            prediction[
                "gender_confidence"
            ],

        "face_confidence":
            prediction[
                "face_confidence"
            ],

        "age_in_8_classes":
            age_class_valid

    }


    # Add age classes only when valid

    if actual_class is not None:

        result[
            "actual_age_class"
        ] = ageList[
            actual_class
        ]

    else:

        result[
            "actual_age_class"
        ] = "OUT_OF_CLASS_RANGE"


    if predicted_class is not None:

        result[
            "predicted_age_class"
        ] = ageList[
            predicted_class
        ]

    else:

        result[
            "predicted_age_class"
        ] = "OUT_OF_CLASS_RANGE"


    if age_class_valid:

        result[
            "age_class_correct"
        ] = (
            actual_class ==
            predicted_class
        )

    else:

        result[
            "age_class_correct"
        ] = None


    results.append(
        result
    )


    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if (
        index + 1
    ) % 100 == 0:

        elapsed = (
            time.time()
            -
            total_start
        )


        print(

            f"Processed "
            f"{index + 1}/"
            f"{len(image_files)} | "

            f"Evaluated: "
            f"{len(results)} | "

            f"Time: "
            f"{elapsed / 60:.1f} min"

        )


# ============================================================
# TOTAL TIME
# ============================================================

total_time = (
    time.time()
    -
    total_start
)


# ============================================================
# SUMMARY
# ============================================================

print("\n======================================")

print(
    "MODEL EVALUATION RESULTS"
)

print("======================================")


print(
    "\nNumber of images found:",
    len(image_files)
)


print(
    "Number of successfully evaluated:",
    len(results)
)


print(
    "Skipped - invalid filename:",
    skipped_invalid_filename
)


print(
    "Skipped - image read error:",
    skipped_read_error
)


print(
    "Skipped - RetinaFace/alignment:",
    skipped_no_face
)


print(
    "Images not usable for 8-class metrics:",
    skipped_age_class
)


print(
    "Total evaluation time:",
    f"{total_time:.2f} sec"
)


if len(results) == 0:

    raise RuntimeError(
        "No images were successfully evaluated."
    )


# ============================================================
# AGE METRICS
# ============================================================

true_age_array = np.array(
    true_age,
    dtype=np.float32
)


pred_age_array = np.array(
    pred_age,
    dtype=np.float32
)


# ------------------------------------------------------------
# MAE
# ------------------------------------------------------------

age_mae = mean_absolute_error(
    true_age_array,
    pred_age_array
)


# ------------------------------------------------------------
# RMSE
# ------------------------------------------------------------

age_rmse = np.sqrt(
    mean_squared_error(
        true_age_array,
        pred_age_array
    )
)


# ------------------------------------------------------------
# Absolute error
# ------------------------------------------------------------

absolute_errors = np.abs(
    true_age_array -
    pred_age_array
)


within_3 = (
    np.mean(
        absolute_errors <= 3
    )
)


within_5 = (
    np.mean(
        absolute_errors <= 5
    )
)


within_10 = (
    np.mean(
        absolute_errors <= 10
    )
)


# ============================================================
# PRINT CONTINUOUS AGE METRICS
# ============================================================

print("\n---------- CONTINUOUS AGE ----------")


print(
    f"MAE       : {age_mae:.4f} years"
)


print(
    f"RMSE      : {age_rmse:.4f} years"
)


print(
    f"Within ±3 : {within_3:.4%}"
)


print(
    f"Within ±5 : {within_5:.4%}"
)


print(
    f"Within ±10: {within_10:.4%}"
)


# ============================================================
# GENDER METRICS
# ============================================================

gender_accuracy = accuracy_score(

    true_gender,

    pred_gender

)


gender_precision = precision_score(

    true_gender,

    pred_gender,

    average="weighted",

    zero_division=0

)


gender_recall = recall_score(

    true_gender,

    pred_gender,

    average="weighted",

    zero_division=0

)


gender_f1 = f1_score(

    true_gender,

    pred_gender,

    average="weighted",

    zero_division=0

)


print("\n---------- GENDER ----------")


print(
    f"Accuracy  : {gender_accuracy:.4f}"
)


print(
    f"Precision : {gender_precision:.4f}"
)


print(
    f"Recall    : {gender_recall:.4f}"
)


print(
    f"F1-score  : {gender_f1:.4f}"
)


print(
    "\nGender Classification Report:"
)


print(

    classification_report(

        true_gender,

        pred_gender,

        labels=[
            0,
            1
        ],

        target_names=genderList,

        zero_division=0

    )

)


# ============================================================
# GENDER CONFUSION MATRIX
# ============================================================

gender_cm = confusion_matrix(

    true_gender,

    pred_gender,

    labels=[
        0,
        1
    ]

)


print(
    "\nGender Confusion Matrix:"
)


print(
    gender_cm
)


# ============================================================
# 8-CLASS AGE METRICS
# ============================================================

if len(true_age_class) > 0:

    age_class_accuracy = (
        accuracy_score(
            true_age_class,
            pred_age_class
        )
    )


    age_class_precision = (
        precision_score(

            true_age_class,

            pred_age_class,

            average="weighted",

            zero_division=0

        )
    )


    age_class_recall = (
        recall_score(

            true_age_class,

            pred_age_class,

            average="weighted",

            zero_division=0

        )
    )


    age_class_f1 = (
        f1_score(

            true_age_class,

            pred_age_class,

            average="weighted",

            zero_division=0

        )
    )


    print(
        "\n---------- 8-CLASS AGE ----------"
    )


    print(
        "Images used:",
        len(true_age_class)
    )


    print(
        f"Accuracy  : "
        f"{age_class_accuracy:.4f}"
    )


    print(
        f"Precision : "
        f"{age_class_precision:.4f}"
    )


    print(
        f"Recall    : "
        f"{age_class_recall:.4f}"
    )


    print(
        f"F1-score  : "
        f"{age_class_f1:.4f}"
    )


    print(
        "\nAge Classification Report:"
    )


    print(

        classification_report(

            true_age_class,

            pred_age_class,

            labels=list(
                range(
                    len(ageList)
                )
            ),

            target_names=ageList,

            zero_division=0

        )

    )


    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    age_cm = confusion_matrix(

        true_age_class,

        pred_age_class,

        labels=list(
            range(
                len(ageList)
            )
        )

    )


    print(
        "\nAge Confusion Matrix:"
    )


    print(
        age_cm
    )

else:

    age_cm = None

    print(
        "\nNo valid 8-class age samples."
    )


# ============================================================
# SAVE RESULTS
# ============================================================

df = pd.DataFrame(
    results
)


csv_path = (
    BASE_DIR /
    "evaluation_results_new_model.csv"
)


df.to_csv(

    csv_path,

    index=False

)


print(
    "\nDetailed results saved to:"
)


print(
    csv_path
)


# ============================================================
# PLOT GENDER CONFUSION MATRIX
# ============================================================

plt.figure(
    figsize=(6, 5)
)


plt.imshow(
    gender_cm
)


plt.title(
    "Gender Confusion Matrix - New ONNX Model"
)


plt.xlabel(
    "Predicted"
)


plt.ylabel(
    "Actual"
)


plt.xticks(

    [0, 1],

    genderList

)


plt.yticks(

    [0, 1],

    genderList

)


for i in range(2):

    for j in range(2):

        plt.text(

            j,

            i,

            gender_cm[i, j],

            ha="center",

            va="center"

        )


plt.tight_layout()


gender_plot = (

    BASE_DIR /

    "gender_confusion_matrix_new_model.png"

)


plt.savefig(
    gender_plot
)


plt.close()


# ============================================================
# PLOT AGE CONFUSION MATRIX
# ============================================================

if age_cm is not None:

    plt.figure(
        figsize=(10, 8)
    )


    plt.imshow(
        age_cm
    )


    plt.title(
        "Age Group Confusion Matrix - New ONNX Model"
    )


    plt.xlabel(
        "Predicted"
    )


    plt.ylabel(
        "Actual"
    )


    plt.xticks(

        range(
            len(ageList)
        ),

        ageList,

        rotation=45

    )


    plt.yticks(

        range(
            len(ageList)
        ),

        ageList

    )


    for i in range(
        len(ageList)
    ):

        for j in range(
            len(ageList)
        ):

            plt.text(

                j,

                i,

                age_cm[i, j],

                ha="center",

                va="center"

            )


    plt.tight_layout()


    age_plot = (

        BASE_DIR /

        "age_confusion_matrix_new_model.png"

    )


    plt.savefig(
        age_plot
    )


    plt.close()


else:

    age_plot = None


# ============================================================
# FINAL SUMMARY
# ============================================================

print(
    "\n======================================"
)

print(
    "EVALUATION COMPLETE"
)

print(
    "======================================"
)


print(
    "\nContinuous Age:"
)


print(
    f"MAE: {age_mae:.4f} years"
)


print(
    f"RMSE: {age_rmse:.4f} years"
)


print(
    f"Within ±5 years: "
    f"{within_5:.2%}"
)


print(
    "\nGender:"
)


print(
    f"Accuracy: "
    f"{gender_accuracy:.4f}"
)


if age_cm is not None:

    print(
        "\n8-Class Age:"
    )


    print(
        f"Accuracy: "
        f"{age_class_accuracy:.4f}"
    )


print(
    "\nSaved:"
)


print(
    csv_path
)


print(
    gender_plot
)


if age_plot is not None:

    print(
        age_plot
    )