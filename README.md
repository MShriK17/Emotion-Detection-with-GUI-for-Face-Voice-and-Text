# Emotion Detection Project

This project provides a backend API for detecting human emotions using **face images**, **voice audio**, and **text input**. It uses machine learning and deep learning models, and is built with FastAPI.

## Features

- Detect emotion from face images using a CNN model
- Detect emotion from voice using MFCC features and an ML classifier
- Detect emotion from text using a pretrained BERT transformer

## File Structure

```
emotion-detection-project/
├── backend/
│   ├── api/               # API routes for face, voice, and text
│   ├── models/            # Pretrained and training scripts for models
│   ├── utils/             # Utility functions for predictions
│   ├── app.py             # Main FastAPI app
│   ├── requirements.txt   # Backend dependencies
└── README.md
```

## Getting Started

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the API:

```bash
uvicorn backend.app:app --reload
```

3. Test endpoints:

- POST /face/detect — image file
- POST /voice/detect — audio file
- POST /text/detect — raw text

## Author

Generated using ChatGPT.
