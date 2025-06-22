# Facial Profile Creator — Documentation

## Project Overview

This project implements a FastAPI application to generate descriptive facial profiles from images and compare facial profiles for similarity. The main goal is to extract meaningful facial characteristics using Mediapipe’s Face Mesh and translate them into natural language descriptions. Additionally, it allows the uset to compare two images to determine likeness based on facial features.

---

## API Endpoints

### Enpoint 1: `/create-profile` (POST)

- **Input:** Single image file upload (`file`).  
- **Process:**  
  - Uses Mediapipe to detect facial landmarks.  
  - Extracts geometric facial features (ratios, distances, angles).  
  - Generates a human-readable profile description summarizing key traits.  
- **Output:** JSON with `"description"` string containing the facial profile.  
- **Error Handling:** Returns HTTP 400 if no face detected.

### Enpoint 2: `/match-profile` (POST)

- **Input:** Two image files uploaded as `file1` and `file2`.  
- **Process:**  
  - Extracts profiles and features from both images.  
  - Converts features into numeric vectors for comparison.  
  - Computes Euclidean distance and normalizes it to a similarity score (0 to 1).  
  - Returns both profile descriptions and a textual explanation of similarity.  
- **Output:** JSON with keys:  
  - `"similarity"` (float between 0 and 1)  
  - `"explanation"` (string)  
  - `"profile1"` (string)  
  - `"profile2"` (string)  
- **Error Handling:** Returns HTTP 400 if no face detected in either image.

---

## Facial Feature Extraction

- **Method:** Uses Mediapipe Face Mesh for 468 3D facial landmarks.  
- **Features computed include:**  
  - Jaw, cheek, forehead widths and their ratios  
  - Face length relative to jaw width  
  - Nose length, width, and tip angle  
  - Eye size and spacing  
  - Eyebrow arch height and thickness  
  - Lip fullness and smile detection (via lip curvature)  
  - Chin prominence  
  - Facial symmetry via jaw point alignment  
- **Purpose:** These metrics capture structural and expressive traits that define face shape and appearance.

---

## Profile Generation Logic

- Converts numeric features into descriptive phrases using thresholds and ranges.  
- Adds randomized introductory phrases to avoid repetition.  
- Includes emotional expression by detecting smile presence.  
- Produces a natural, readable paragraph summarizing the subject’s facial traits.

---

## Profile Matching Logic

- Features from both images converted into numeric vectors.  
- Boolean features converted to floats (`True` = 1.0, `False` = 0.0).  
- Euclidean distance computed between vectors.  
- Distance normalized to similarity score between 0 and 1.  
- Textual explanation generated based on similarity thresholds.

---

## Design Decisions & Rationale

### Main Challenge

Initially, I planned to use the **face_recognition** library due to its simplicity and built-in support for face encoding and landmark extraction. However, due to platform compatibility issues and difficulty with installing dlib (a dependency), I decided to use **Mediapipe**.

Mediapipe provides robust facial landmark detection via Face Mesh, which allows the app to generate a meaningful facial profile based on geometric analysis of features like the chin, eyebrows, and lips.

### Additional Design Decisions:

- Implemented `extract_features()` to compute scale-invariant geometric features (ratios, distances, angles) from 468 facial landmarks using Mediapipe, ensuring robustness across varying image sizes.
- Designed `generate_profile_description()` to convert numeric features into natural language summaries using randomized phrasing and emotion detection (e.g., smiling) to enhance readability and personality.
- Developed the `/match-profile` endpoint to convert facial profiles into numeric vectors and compare them using Euclidean distance, making the descriptions actionable for real-world identity and realism comparisons.
- Chose FastAPI for its asynchronous architecture, rapid development speed, and built-in interactive docs (`/docs`), which simplify testing and API exploration.
- Converted Mediapipe’s normalized landmark coordinates to pixel space for accurate facial geometry calculations.
- Normalized boolean values (like smile presence) into floats to enable unified numeric comparison.
- Measured facial symmetry by comparing mirrored jaw landmarks — a simple yet interpretable indicator of facial balance.

---

## How to Use

1. Run the FastAPI server by typing in terminal: uvicorn app.main:app --reload
2. Use `/create-profile` to generate a facial description from an image.  
3. Use `/match-profile` to compare two images and get similarity insights.  
4. Explore interactive Swagger docs at `/docs` for examples and testing.