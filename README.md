# Validia Recruiting Challege - Facial Recognition API

Welcome to my **Validia Recruiting Challenge**

This project implements a FastAPI application that:
- Accepts user-uploaded images
- Analyzes facial features using Mediapipe
- Generates a highly descriptive, human-readable "facial profile"
- Compares two images to determine how similar their facial characteristics are

## 🎯 Objective
Build an API that can generate and use facial profiles to describe and compare key traits of human faces — with creativity, precision, and real-world utility.

## 🚀 Features
- **POST /create-profile**: Upload an image and receive a detailed facial description.
- **POST /match-profile**: Upload two images to get a similarity score and side-by-side facial profile comparison.

## 📦 Tech Stack
- **Python 3.10**
- **FastAPI** – for API routing and interactive documentation
- **Mediapipe** – for facial landmark detection
- **OpenCV + NumPy** – for image processing and geometric calculations
- **Pillow** – to handle image I/O

---

## 🧠 Design Decisions & Explanation
(for additional information read "documentation.md")

### Facial Feature Extraction
We extract 468 facial landmarks using Mediapipe’s Face Mesh. Key facial characteristics are derived using:
- **Ratios** (e.g. jaw-to-cheek width, eye width-to-height)
- **Angles** (e.g. nose tip angle)
- **Distances** (e.g. symmetry offsets, smile curvature)

### Facial Profile Generation
Raw geometric features are translated into expressive, readable profiles covering:
- Face shape
- Nose proportions
- Eye size and spacing
- Eyebrow shape and thickness
- Lip fullness and smile detection
- Chin and jawline structure
- Symmetry

Natural language is varied using randomized phrases for more organic results. Emotion is lightly detected based on mouth curvature to infer smiles.

### Profile Matching
The `/match-profile` endpoint:
- Generates feature vectors for both images
- Compares them using Euclidean distance
- Converts distance into a normalized similarity score (0.0 to 1.0)
- Returns both profile descriptions and a plain-language similarity explanation

This demonstrates how a profile can be used for comparison or authentication use cases.

---

## 📄 API Usage

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run the Server
```bash
uvicorn main:app --reload
```

### Interactive Docs
Visit: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 📸 Video Walkthrough
[Click here to watch the video walkthrough](https://drive.google.com/file/d/1sjBNuwKtKhW2-So8ANQqgCeFxAdUDvst/view?usp=sharing)

---

## 🙌 Author
Built by Lucas Lopez for the Validia Recruiting Challenge.
