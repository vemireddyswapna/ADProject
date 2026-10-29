# Cervical Cell AI Classification System

## Project Overview

The Cervical Cell AI Classification System is a deep learning-based web application designed to classify cervical cell images into five different cell types.

The system uses a pretrained EfficientNetB0 convolutional neural network with transfer learning to analyze cervical cell images.

## Cell Classes

The model classifies images into the following five categories:

1. Dyskeratotic
2. Koilocytotic
3. Metaplastic
4. Parabasal
5. Superficial-Intermediate

## Features

- Patient information entry
- Cervical cell image upload
- AI-based cell classification
- Prediction confidence score
- Project-defined risk indication
- Prediction history
- SQLite database
- PDF prediction report
- Downloadable report
- Web-based interface

## Technologies Used

- Python
- TensorFlow
- Keras
- EfficientNetB0
- OpenCV
- NumPy
- Scikit-learn
- Flask
- HTML
- CSS
- SQLite
- ReportLab

## Project Structure

```text
cervialproject/
│
├── dataset/
│
├── models/
│   └── cervical_model.keras
│
├── uploads/
├── reports/
├── history/
│
├── templates/
│   ├── index.html
│   ├── result.html
│   └── history.html
│
├── static/
│   └── style.css
│
├── utils/
│   └── preprocessing.py
│
├── app.py
├── train.py
├── predict.py
├── database.py
├── risk_analysis.py
├── requirements.txt
└── README.md