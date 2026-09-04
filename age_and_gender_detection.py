# ============================================================
# AGE AND GENDER DETECTION
# ONNX RetinaFace + 5-Point Alignment
# + Hugging Face Age/Gender ONNX Model
# ============================================================

import cv2 as cv
import numpy as np
import onnxruntime as ort
import math
import time
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

RETINAFACE_MODEL = MODEL_DIR / "retinaface_mv1_0.25.onnx"
AGE_GENDER_MODEL = MODEL_DIR / "model.onnx"


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


# ============================================================
# AGE GROUP CONVERSION
# ============================================================

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



AGE_CLASS_CENTERS = np.array([
        1.0,
        5.0,
        10.0,
        17.5,
        28.5,
        40.5,
        50.5,
        80.0
    ], dtype=np.float32)


def age_to_nearest_class(age):
    """
    Convert continuous predicted age into
    the nearest UTKFace age-group representative.
    """

    distances = np.abs(
        AGE_CLASS_CENTERS - age
    )

    return int(np.argmin(distances))

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

print("Model:", AGE_GENDER_MODEL)


# ------------------------------------------------------------
# ONNX Runtime configuration
# ------------------------------------------------------------

session_options = ort.SessionOptions()

session_options.graph_optimization_level = (
    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
)

# CPU has 4 logical processors
session_options.intra_op_num_threads = 4
session_options.inter_op_num_threads = 1


age_gender_session = ort.InferenceSession(
    str(AGE_GENDER_MODEL),
    sess_options=session_options,
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

for output in age_gender_session.get_outputs():

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
    str(RETINAFACE_MODEL),
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
        for output in retina_session.get_outputs()
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
                image_height / step
            ),

            math.ceil(
                image_width / step
            )
        ]

        for step in STEPS
    ]

    for k, (
        map_height,
        map_width
    ) in enumerate(feature_maps):

        step = STEPS[k]

        for i in range(map_height):

            for j in range(map_width):

                for min_size in MIN_SIZES[k]:

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
                        * step
                        / image_width
                    )

                    cy = (
                        (i + 0.5)
                        * step
                        / image_height
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

    order = scores.argsort()[::-1]

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
            width * height
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

def getFaceBox(image):

    if image is None:

        return None, []


    original = image.copy()

    original_height, original_width = (
        image.shape[:2]
    )


    # --------------------------------------------------------
    # RESIZE
    # --------------------------------------------------------

    resized = cv.resize(
        image,
        (INPUT_SIZE, INPUT_SIZE),
        interpolation=cv.INTER_LINEAR
    )


    # --------------------------------------------------------
    # RETINAFACE PREPROCESSING
    # --------------------------------------------------------

    input_image = resized.astype(
        np.float32
    )

    input_image -= np.array(
        [104.0, 117.0, 123.0],
        dtype=np.float32
    )

    input_image = input_image.transpose(
        2,
        0,
        1
    )

    input_image = np.expand_dims(
        input_image,
        axis=0
    )


    # --------------------------------------------------------
    # RETINAFACE INFERENCE
    # --------------------------------------------------------

    retina_start = time.time()

    outputs = retina_session.run(
        None,
        {
            retina_input_name:
            input_image
        }
    )

    retina_time = (
        time.time()
        -
        retina_start
    )


    # --------------------------------------------------------
    # OUTPUTS
    # --------------------------------------------------------

    loc = outputs[0].squeeze(0)

    conf = outputs[1].squeeze(0)

    landm = outputs[2].squeeze(0)


    if len(PRIORS) != len(loc):

        raise RuntimeError(
            "RetinaFace prior count mismatch: "
            f"{len(PRIORS)} priors vs "
            f"{len(loc)} predictions"
        )


    # --------------------------------------------------------
    # DECODE BOXES
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
        center - size / 2
    )

    boxes[:, 2:4] = (
        center + size / 2
    )


    # --------------------------------------------------------
    # DECODE LANDMARKS
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
    # FACE CONFIDENCE
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

        return original, []


    # --------------------------------------------------------
    # CONVERT TO 640x640
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


    # --------------------------------------------------------
    # SCALE BACK TO ORIGINAL IMAGE
    # --------------------------------------------------------

    scale_x = (
        original_width /
        INPUT_SIZE
    )

    scale_y = (
        original_height /
        INPUT_SIZE
    )

    boxes[:, [0, 2]] *= scale_x

    boxes[:, [1, 3]] *= scale_y

    landmarks[:, 0::2] *= scale_x

    landmarks[:, 1::2] *= scale_y


    # --------------------------------------------------------
    # CREATE FACE DATA
    # --------------------------------------------------------

    faces = []

    for i in range(
        len(boxes)
    ):

        x1, y1, x2, y2 = (
            boxes[i]
        )


        x1 = int(
            max(
                0,
                min(
                    original_width - 1,
                    x1
                )
            )
        )

        y1 = int(
            max(
                0,
                min(
                    original_height - 1,
                    y1
                )
            )
        )

        x2 = int(
            max(
                0,
                min(
                    original_width - 1,
                    x2
                )
            )
        )

        y2 = int(
            max(
                0,
                min(
                    original_height - 1,
                    y2
                )
            )
        )


        points = (
            landmarks[i]
            .reshape(5, 2)
        )


        aligned_face = align_face(
            original,
            points
        )


        if aligned_face is None:

            continue


        faces.append({

            "bbox": (
                x1,
                y1,
                x2,
                y2
            ),

            "landmarks": points,

            "confidence": float(
                scores[i]
            ),

            "aligned_face":
                aligned_face
        })


    # --------------------------------------------------------
    # DRAW DETECTIONS
    # --------------------------------------------------------

    frameFace = original.copy()


    for face_data in faces:

        x1, y1, x2, y2 = (
            face_data["bbox"]
        )

        confidence = (
            face_data["confidence"]
        )

        points = (
            face_data["landmarks"]
        )


        cv.rectangle(
            frameFace,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )


        for point in points:

            px = int(
                point[0]
            )

            py = int(
                point[1]
            )

            cv.circle(
                frameFace,
                (px, py),
                3,
                (0, 0, 255),
                -1
            )


        cv.putText(
            frameFace,
            f"Face {confidence:.3f}",
            (
                x1,
                max(
                    25,
                    y1 - 10
                )
            ),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv.LINE_AA
        )


    print(
        "RetinaFace time: "
        f"{retina_time:.4f} sec"
    )


    return frameFace, faces


# ============================================================
# AGE + GENDER ONNX INFERENCE
# ============================================================

def predict_age_gender(face):

    """
    Input:
        aligned BGR face, 224x224

    Output:
        predicted age
        female probability
        male probability
    """


    # --------------------------------------------------------
    # BGR -> RGB
    # --------------------------------------------------------

    rgb_face = cv.cvtColor(
        face,
        cv.COLOR_BGR2RGB
    )


    # --------------------------------------------------------
    # Convert to float
    # --------------------------------------------------------

    image = rgb_face.astype(
        np.float32
    )


    # --------------------------------------------------------
    # Normalize
    #
    # Model expects pixel_values.
    # Standard image normalization:
    #
    # (pixel / 255 - mean) / std
    # --------------------------------------------------------

    image = image / 255.0

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
    # Add batch dimension
    # --------------------------------------------------------

    input_tensor = np.expand_dims(
        image,
        axis=0
    ).astype(np.float32)


    # --------------------------------------------------------
    # ONNX INFERENCE
    # --------------------------------------------------------

    start = time.time()

    outputs = age_gender_session.run(
        None,
        {
            age_gender_input_name:
            input_tensor
        }
    )

    inference_time = (
        time.time()
        -
        start
    )


    # --------------------------------------------------------
    # LOGITS
    # --------------------------------------------------------

    logits = outputs[0][0]


    # Model output:
    #
    # [age, gender_logit]
    #
    # age = continuous age
    # gender logit converted with sigmoid


    predicted_age = float(
        logits[0]
    )

    female_probability = float(logits[1])
    male_probability = 1.0 - female_probability


    if female_probability >= 0.5:
        gender = "Female"
        gender_confidence = female_probability
    else:
        gender = "Male"
        gender_confidence = male_probability

    return {

        "age": predicted_age,

        "gender": gender,

        "female_probability":
            float(female_probability),

        "male_probability":
            float(male_probability),

        "gender_confidence":
            float(gender_confidence),

        "inference_time":
            inference_time
    }


# ============================================================
# AGE + GENDER DETECTOR
# ============================================================

def age_gender_detector(frame):

    total_start = time.time()


    # --------------------------------------------------------
    # RETINAFACE
    # --------------------------------------------------------

    frameFace, faces = getFaceBox(
        frame
    )


    if frameFace is None:

        return frame


    if len(faces) == 0:

        cv.putText(
            frameFace,
            "No face detected",
            (20, 40),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        return frameFace


    # --------------------------------------------------------
    # PROCESS EACH FACE
    # --------------------------------------------------------

    for face_data in faces:

        bbox = face_data["bbox"]

        face = face_data[
            "aligned_face"
        ]


        x1, y1, x2, y2 = bbox


        # ----------------------------------------------------
        # AGE/GENDER
        # ----------------------------------------------------

        prediction = predict_age_gender(
            face
        )


        predicted_age = (
            prediction["age"]
        )

        gender = (
            prediction["gender"]
        )

        gender_confidence = (
            prediction["gender_confidence"]
        )

        female_probability = (
            prediction["female_probability"]
        )

        male_probability = (
            prediction["male_probability"]
        )

        inference_time = (
            prediction["inference_time"]
        )


        # ----------------------------------------------------
        # AGE GROUP
        # ----------------------------------------------------

        age_class = age_to_nearest_class(
            predicted_age
        )

        age_group = ageList[
            age_class
        ]


        # ----------------------------------------------------
        # PRINT RESULTS
        # ----------------------------------------------------

        print(
            "\nAge : "
            f"{predicted_age:.2f} years"
            f" -> {age_group}"
        )

        print(
            "Gender : "
            f"{gender}, "
            f"female probability = "
            f"{female_probability:.3f}, "
            f"confidence = "
            f"{gender_confidence:.3f}"
        )

        print(
            "Male probability: "
            f"{male_probability:.3f}"
        )

        print(
            "Age/Gender ONNX time: "
            f"{inference_time:.4f} sec"
        )


        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        label = (
            f"{gender}, "
            f"{predicted_age:.0f} years"
        )


        cv.rectangle(
            frameFace,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )


        cv.putText(
            frameFace,
            label,
            (
                x1,
                max(
                    25,
                    y1 - 10
                )
            ),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv.LINE_AA
        )


    # --------------------------------------------------------
    # TOTAL TIME
    # --------------------------------------------------------

    total_time = (
        time.time()
        -
        total_start
    )


    print(
        "\nTOTAL time: "
        f"{total_time:.4f} sec"
    )


    return frameFace


# ============================================================
# OPTIONAL TEST
# ============================================================

if __name__ == "__main__":

    print("\n======================================")
    print("AGE + GENDER TEST")
    print("======================================")


    test_image_path = (
        BASE_DIR /
        "photo" /
        "image1.jpg"
    )


    print(
        "Test image:",
        test_image_path
    )


    image = cv.imread(
        str(test_image_path)
    )


    if image is None:

        raise FileNotFoundError(
            f"Could not load: "
            f"{test_image_path}"
        )


    result = age_gender_detector(
        image
    )


    output_path = (
        BASE_DIR /
        "final_age_gender_result.jpg"
    )


    cv.imwrite(
        str(output_path),
        result
    )


    print(
        "\nResult saved to:"
    )

    print(
        output_path
    )


    print("\n======================================")
    print("TEST COMPLETE")
    print("======================================")