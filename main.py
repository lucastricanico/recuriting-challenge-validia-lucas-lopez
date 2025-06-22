# Lucas Lopez - main.py File
# 06/21/2025

# Import necessary libraries for API, image processing, and facial landmark detection
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import mediapipe as mp

app = FastAPI()

# Initialize Mediapipe Face Mesh solution for detecting facial landmarks
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)


# Define Pydantic models for API request/response validation and documentation
class Profile(BaseModel):
    description: str


class MatchResult(BaseModel):
    similarity: float
    explanation: str
    profile1: str
    profile2: str


def distance(a, b):
    """
    Calculate Euclidean distance between two points a and b.
    Points are numpy arrays representing coordinates.
    """
    return np.linalg.norm(a - b)


def angle(a, b, c):
    """
    Calculate the angle (in degrees) at point b formed by points a and c.
    Uses the cosine rule via dot product.
    """
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


def get_point(landmarks, idx, w, h):
    """
    Convert normalized landmark coordinates to pixel coordinates.
    landmarks: Mediapipe landmarks object
    idx: index of the landmark point
    w, h: width and height of the image
    Returns a numpy array with pixel coordinates [x, y].
    """
    lm = landmarks.landmark[idx]
    return np.array([int(lm.x * w), int(lm.y * h)])


def extract_features(landmarks, image_shape):
    """
    Extract various facial features and ratios from landmarks and image dimensions.
    Features include distances, ratios, angles, and symmetry measures that describe
    facial structure and expressions.
    """
    h, w, _ = image_shape

    def gp(idx):  # shorthand to get pixel coordinates for a landmark index
        return get_point(landmarks, idx, w, h)

    # Define key facial points for measurements
    jaw_left = gp(234)  # Left jaw point
    jaw_right = gp(454)  # Right jaw point
    cheek_left = gp(127)  # Left cheekbone
    cheek_right = gp(356)  # Right cheekbone
    forehead_left = gp(70)  # Left forehead edge
    forehead_right = gp(300)  # Right forehead edge
    chin = gp(152)  # Chin point
    forehead_top = gp(10)  # Top of forehead

    # Calculate widths and lengths of facial regions
    jaw_width = distance(jaw_left, jaw_right)
    cheek_width = distance(cheek_left, cheek_right)
    forehead_width = distance(forehead_left, forehead_right)
    face_length = distance(forehead_top, chin)

    # Ratios to describe face shape
    jaw_cheek_ratio = jaw_width / (
        cheek_width + 1e-5
    )  # Ratio of jaw width to cheek width
    cheek_forehead_ratio = cheek_width / (
        forehead_width + 1e-5
    )  # Ratio of cheek width to forehead width
    face_length_ratio = face_length / (
        jaw_width + 1e-5
    )  # Ratio of face length to jaw width

    # Nose measurements
    nose_tip = gp(1)
    nose_bridge = gp(168)
    nose_left = gp(98)
    nose_right = gp(327)
    nose_length = distance(nose_tip, nose_bridge)
    nose_width = distance(nose_left, nose_right)
    nose_tip_angle = angle(
        nose_left, nose_tip, nose_right
    )  # Angle at nose tip between left and right nose edges
    nose_ratio = nose_length / (nose_width + 1e-5)  # Ratio of nose length to width

    # Eye measurements (using left eye as example)
    left_eye_left = gp(33)
    left_eye_right = gp(133)
    left_eye_top = gp(159)
    left_eye_bottom = gp(145)
    eye_width = distance(left_eye_left, left_eye_right)
    eye_height = distance(left_eye_top, left_eye_bottom)
    eye_ratio = eye_height / (eye_width + 1e-5)  # Eye height to width ratio

    # Calculate inter-eye distance and spacing relative to face width
    left_eye_center = (left_eye_left + left_eye_right) / 2
    right_eye_left = gp(362)
    right_eye_right = gp(263)
    right_eye_center = (right_eye_left + right_eye_right) / 2
    inter_eye_distance = distance(left_eye_center, right_eye_center)
    face_width = jaw_width
    eye_spacing_ratio = inter_eye_distance / (
        face_width + 1e-5
    )  # Ratio of inter-eye distance to face width

    # Eyebrow measurements (left eyebrow as example)
    brow_left_inner = gp(70)
    brow_left_center = gp(66)
    brow_height = (
        brow_left_inner[1] - brow_left_center[1]
    )  # Vertical height difference to indicate arch
    brow_upper = gp(70)
    brow_lower = gp(63)
    brow_thickness = distance(brow_upper, brow_lower)  # Thickness of eyebrow

    # Lips and mouth measurements
    top_lip = gp(13)
    bottom_lip = gp(14)
    lip_left = gp(61)
    lip_right = gp(291)
    lip_height = distance(top_lip, bottom_lip)
    lip_width = distance(lip_left, lip_right)
    lip_ratio = lip_height / (lip_width + 1e-5)  # Lip height to width ratio

    # Determine if subject is smiling based on lip midpoint vs corners height
    mouth_left_corner = gp(61)
    mouth_right_corner = gp(291)
    mouth_top = gp(13)
    mouth_bottom = gp(14)
    corner_height_avg = (mouth_left_corner[1] + mouth_right_corner[1]) / 2
    lip_midpoint_y = (mouth_top[1] + mouth_bottom[1]) / 2
    smiling = (
        lip_midpoint_y > corner_height_avg + 3
    )  # Smiling if lip midpoint is significantly lower (y increases downwards)

    # Chin and jawline measurements
    chin_point = gp(152)
    jaw_mid = (jaw_left + jaw_right) / 2
    chin_jaw_dist = distance(chin_point, jaw_mid)
    chin_jaw_ratio = chin_jaw_dist / (
        jaw_width + 1e-5
    )  # Ratio indicating chin prominence

    # Facial symmetry check by comparing horizontal positions of jaw points relative to image width
    left_face = jaw_left
    right_face = jaw_right
    symmetry_diff = abs(
        left_face[0] - (w - right_face[0])
    )  # Difference in x-coordinates for symmetry

    # Return all extracted features as a dictionary
    return {
        "jaw_cheek_ratio": jaw_cheek_ratio,
        "cheek_forehead_ratio": cheek_forehead_ratio,
        "face_length_ratio": face_length_ratio,
        "nose_ratio": nose_ratio,
        "nose_tip_angle": nose_tip_angle,
        "eye_ratio": eye_ratio,
        "eye_spacing_ratio": eye_spacing_ratio,
        "brow_height": brow_height,
        "brow_thickness": brow_thickness,
        "lip_ratio": lip_ratio,
        "smiling": smiling,
        "chin_jaw_ratio": chin_jaw_ratio,
        "symmetry_diff": symmetry_diff,
    }


def generate_profile_description(landmarks, image_shape, features=None):
    """
    Generates a detailed natural language facial profile description based on extracted features.

    """
    h, w, _ = image_shape

    def get_point(idx):
        lm = landmarks.landmark[idx]
        return np.array([int(lm.x * w), int(lm.y * h)])

    import random

    # Use provided features or compute if None
    if features is None:
        features = extract_features(landmarks, image_shape)

    desc = []

    # Describe face shape based on ratios of length and width
    f_ratio = features["face_length_ratio"]
    cf_ratio = features["cheek_forehead_ratio"]
    jc_ratio = features["jaw_cheek_ratio"]

    if f_ratio > 1.6 and cf_ratio > 1.1:
        face_shape = "an elongated oval face"
    elif abs(jc_ratio - 1) < 0.05:
        face_shape = "a round face"
    elif jc_ratio < 0.9:
        face_shape = "a heart-shaped face with a narrower jaw"
    elif jc_ratio > 1.1:
        face_shape = "a square face with a strong jawline"
    else:
        face_shape = "a uniquely shaped face"
    desc.append(f"The subject has {face_shape}.")

    # Describe nose shape based on nose length-to-width ratio and tip angle
    nr = features["nose_ratio"]
    nta = int(features["nose_tip_angle"])
    if nr > 1.8:
        nose_desc = "a long and narrow nose"
    elif nr < 1.3:
        nose_desc = "a short and broad nose"
    else:
        nose_desc = "a well-proportioned nose"
    desc.append(
        f"They possess {nose_desc}, with a tip angle of approximately {nta} degrees."
    )

    # Describe eyes based on eye height-to-width ratio
    er = features["eye_ratio"]
    if er > 0.4:
        eye_desc = "large, wide-open eyes"
    elif er < 0.2:
        eye_desc = "narrow, almond-shaped eyes"
    else:
        eye_desc = "moderately sized eyes"
    desc.append(f"The eyes are {eye_desc}.")

    # Describe eye spacing relative to face width
    esr = features["eye_spacing_ratio"]
    if esr < 0.28:
        eye_spacing_desc = "close-set eyes"
    elif esr > 0.34:
        eye_spacing_desc = "wide-set eyes"
    else:
        eye_spacing_desc = "averagely spaced eyes"
    desc.append(f"They have {eye_spacing_desc}.")

    # Describe eyebrow arch and thickness
    bh = features["brow_height"]
    bt = features["brow_thickness"]
    if bh > 18:
        brow_arch_desc = "highly arched eyebrows"
    elif bh > 10:
        brow_arch_desc = "moderately arched eyebrows"
    else:
        brow_arch_desc = "straight eyebrows"

    if bt > 10:
        brow_thickness_desc = "thick"
    elif bt < 6:
        brow_thickness_desc = "thin"
    else:
        brow_thickness_desc = "medium-thick"
    desc.append(f"The subject has {brow_arch_desc} that are {brow_thickness_desc}.")

    # Describe lips and smile expression
    lr = features["lip_ratio"]
    smiling = features["smiling"]
    if lr > 0.35:
        lip_desc = "full and prominent lips"
    elif lr < 0.15:
        lip_desc = "thin lips"
    else:
        lip_desc = "moderately full lips"
    smile_desc = "smiling" if smiling else "neutral expression"
    desc.append(f"They have {lip_desc} and a {smile_desc}.")

    # Describe chin and jawline shape
    cjr = features["chin_jaw_ratio"]
    if cjr > 0.45:
        jaw_desc = "a pointed chin"
    else:
        jaw_desc = "a softly rounded jawline"
    desc.append(f"The face features {jaw_desc}.")

    # Describe facial symmetry
    sym_diff = features["symmetry_diff"]
    if sym_diff < 10:
        symmetry_desc = "notably symmetrical facial features"
    else:
        symmetry_desc = "slightly asymmetrical features"
    desc.append(f"The face shows {symmetry_desc}.")

    # Choose a random introductory phrase for natural flow
    intro_phrases = [
        "Overall,",
        "In summary,",
        "The analysis shows that",
        "Observations indicate that",
        "Interestingly,",
    ]
    intro = random.choice(intro_phrases)
    return f"{intro} {' '.join(desc)}"


@app.post(
    "/create-profile",
    response_model=Profile,
    summary="Generate facial profile from an image",
    description="Uploads an image, analyzes facial landmarks using Mediapipe, "
    "and returns a detailed facial profile description.",
    response_description="A natural language description of the facial features.",
)
async def create_profile(file: UploadFile = File(...)):
    """
    Endpoint to generate a facial profile description from an uploaded image.
    - Read and convert the uploaded image to RGB format.
    - Use Mediapipe Face Mesh to detect facial landmarks.
    - If no face detected, raise HTTP error.
    - Extract facial features and generate a descriptive profile.
    - Return the description as JSON.
    """
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")
    image_np = np.array(image)
    # Convert RGB to BGR as Mediapipe expects BGR images
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    results = face_mesh.process(image_bgr)
    if not results.multi_face_landmarks:
        # Return error if no face detected in the image
        raise HTTPException(status_code=400, detail="No face detected in the image.")

    landmarks = results.multi_face_landmarks[0]
    features = extract_features(landmarks, image_np.shape)
    description = generate_profile_description(landmarks, image_np.shape, features)
    return {"description": description}


@app.post(
    "/match-profile",
    response_model=MatchResult,
    summary="Compare two facial profiles for similarity",
    description="Uploads two images, generates facial profiles for both, compares the profiles "
    "using extracted facial features, and returns a similarity score and comparison report.",
    response_description="Similarity score (0-1), profiles of both images, and textual explanation.",
)
async def match_profile(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    """
    Endpoint to compare two facial profiles from uploaded images.
    - Process each image to detect facial landmarks and extract features.
    - Generate descriptive profiles for both images
    - Convert feature dictionaries to numeric vectors for comparison.
    - Convert boolean features to floats.
    - Calculate Euclidean distance between feature vectors.
    - Normalize distance to compute a similarity score between 0 and 1.
    - Generate a textual explanation based on similarity score.
    - Return similarity, explanation, and both profile descriptions.
    """
    # Process first image
    contents1 = await file1.read()
    image1 = Image.open(BytesIO(contents1)).convert("RGB")
    image_np1 = np.array(image1)
    image_bgr1 = cv2.cvtColor(image_np1, cv2.COLOR_RGB2BGR)

    results1 = face_mesh.process(image_bgr1)
    if not results1.multi_face_landmarks:
        # Error if no face detected in first image
        raise HTTPException(
            status_code=400, detail="No face detected in the first image."
        )

    landmarks1 = results1.multi_face_landmarks[0]
    features1 = extract_features(landmarks1, image_np1.shape)
    description1 = generate_profile_description(landmarks1, image_np1.shape, features1)

    # Process second image
    contents2 = await file2.read()
    image2 = Image.open(BytesIO(contents2)).convert("RGB")
    image_np2 = np.array(image2)
    image_bgr2 = cv2.cvtColor(image_np2, cv2.COLOR_RGB2BGR)

    results2 = face_mesh.process(image_bgr2)
    if not results2.multi_face_landmarks:
        # Error if no face detected in second image
        raise HTTPException(
            status_code=400, detail="No face detected in the second image."
        )

    landmarks2 = results2.multi_face_landmarks[0]
    features2 = extract_features(landmarks2, image_np2.shape)
    description2 = generate_profile_description(landmarks2, image_np2.shape, features2)

    # Prepare feature vectors for similarity calculation
    keys = list(features1.keys())
    # Convert all features to float arrays; handle booleans later
    vec1 = np.array(
        [
            (
                features1[k]
                if isinstance(features1[k], (int, float))
                else float(features1[k])
            )
            for k in keys
        ],
        dtype=float,
    )
    vec2 = np.array(
        [
            (
                features2[k]
                if isinstance(features2[k], (int, float))
                else float(features2[k])
            )
            for k in keys
        ],
        dtype=float,
    )

    # Convert boolean features (like 'smiling') to floats (1.0 or 0.0)
    for i, k in enumerate(keys):
        if isinstance(features1[k], bool):
            vec1[i] = 1.0 if features1[k] else 0.0
        if isinstance(features2[k], bool):
            vec2[i] = 1.0 if features2[k] else 0.0

    # Calculate Euclidean distance between feature vectors
    dist = np.linalg.norm(vec1 - vec2)
    # Calculate max possible distance assuming feature values differ by 1 for all features
    max_dist = np.linalg.norm(np.ones_like(vec1))
    # Similarity score normalized between 0 and 1 (1 means identical)
    similarity = max(0.0, 1.0 - dist / (max_dist + 1e-8))

    # Generate explanation based on similarity thresholds
    if similarity > 0.8:
        explanation = "The two facial profiles are very similar."
    elif similarity > 0.5:
        explanation = "The two facial profiles share some similarities."
    else:
        explanation = "The two facial profiles are quite different."

    return {
        "similarity": similarity,
        "explanation": explanation,
        "profile1": description1,
        "profile2": description2,
    }
