import os
import sqlite3
from datetime import datetime

import cv2
import numpy as np
import tensorflow as tf

from flask import Flask, render_template, request, redirect, url_for, send_file

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


# ============================================================
# FLASK CONFIGURATION
# ============================================================

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
REPORT_FOLDER = os.path.join(BASE_DIR, "reports")
HISTORY_FOLDER = os.path.join(BASE_DIR, "history")


os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)
os.makedirs(HISTORY_FOLDER, exist_ok=True)
DB_PATH = os.path.join("history", "predictions.db")

DATABASE = os.path.join(HISTORY_FOLDER, "predictions.db")

MODEL_PATH = os.path.join(BASE_DIR, "models", "cervical_model.keras")


# ============================================================
# MODEL
# ============================================================

print("Loading cervical cell classification model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")


CLASSES = [
    "Dyskeratotic",
    "Koilocytotic",
    "Metaplastic",
    "Parabasal",
    "Superficial-Intermediate"
]


# ============================================================
# PROJECT-DEFINED RISK INDICATOR
# ============================================================

RISK_LEVELS = {
    "Dyskeratotic": "HIGH",
    "Koilocytotic": "MEDIUM",
    "Metaplastic": "LOW",
    "Parabasal": "LOW",
    "Superficial-Intermediate": "LOW"
}


# ============================================================
# DATABASE
# ============================================================

def init_database():

    conn = sqlite3.connect(DATABASE)

    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_name TEXT,
            age INTEGER,
            hospital_id TEXT,
            doctor_name TEXT,
            image_name TEXT,
            prediction TEXT,
            confidence REAL,
            risk TEXT,
            timestamp TEXT
        )
    """)

    conn.commit()
    conn.close()


def save_prediction(
    patient_name,
    age,
    hospital_id,
    doctor_name,
    image_name,
    prediction,
    confidence,
    risk
):

    conn = sqlite3.connect(DATABASE)

    cursor = conn.cursor()

    timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

    cursor.execute("""
        INSERT INTO predictions
        (
            patient_name,
            age,
            hospital_id,
            doctor_name,
            image_name,
            prediction,
            confidence,
            risk,
            timestamp
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        patient_name,
        age,
        hospital_id,
        doctor_name,
        image_name,
        prediction,
        confidence,
        risk,
        timestamp
    ))

    record_id = cursor.lastrowid

    conn.commit()
    conn.close()

    return record_id


def get_history():

    conn = sqlite3.connect(DATABASE)

    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            id,
            patient_name,
            age,
            hospital_id,
            doctor_name,
            prediction,
            confidence,
            risk,
            timestamp
        FROM predictions
        ORDER BY id DESC
    """)

    records = cursor.fetchall()

    conn.close()

    return records


# ============================================================
# IMAGE PREDICTION
# ============================================================

def predict_image(image_path):

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Unable to read uploaded image.")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (224, 224))

    image = image.astype("float32")

    image = np.expand_dims(image, axis=0)

    predictions = model.predict(image, verbose=0)[0]

    predicted_index = int(np.argmax(predictions))

    predicted_class = CLASSES[predicted_index]

    confidence = float(predictions[predicted_index]) * 100

    risk = RISK_LEVELS.get(predicted_class, "UNKNOWN")

    return predicted_class, confidence, risk


# ============================================================
# PDF REPORT
# ============================================================

def generate_pdf(
    record_id,
    patient_name,
    age,
    hospital_id,
    doctor_name,
    prediction,
    confidence,
    risk
):

    filename = f"prediction_report_{record_id}.pdf"

    filepath = os.path.join(REPORT_FOLDER, filename)

    pdf = canvas.Canvas(filepath, pagesize=A4)

    width, height = A4

    y = height - 70

    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(60, y, "Cervical Cell AI Analysis Report")

    y -= 50

    pdf.setFont("Helvetica", 11)

    pdf.drawString(60, y, f"Report ID: {record_id}")
    y -= 25

    pdf.drawString(60, y, f"Date: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}")
    y -= 40

    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(60, y, "Patient Information")

    y -= 30

    pdf.setFont("Helvetica", 11)

    pdf.drawString(60, y, f"Patient Name: {patient_name}")
    y -= 22

    pdf.drawString(60, y, f"Age: {age}")
    y -= 22

    pdf.drawString(60, y, f"Hospital ID: {hospital_id}")
    y -= 22

    pdf.drawString(60, y, f"Doctor Name: {doctor_name}")
    y -= 40

    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(60, y, "AI Prediction")

    y -= 30

    pdf.setFont("Helvetica", 11)

    pdf.drawString(60, y, f"Predicted Cell Type: {prediction}")
    y -= 22

    pdf.drawString(60, y, f"Confidence: {confidence:.2f}%")
    y -= 22

    pdf.drawString(60, y, f"Project Risk Indicator: {risk}")

    y -= 50

    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(60, y, "Disclaimer")

    y -= 25

    pdf.setFont("Helvetica", 9)

    disclaimer = (
        "This system is an AI-based research/project prototype for "
        "cervical cell image classification. The risk indicator is "
        "project-defined and is not a clinical diagnosis."
    )

    # Simple wrapping
    words = disclaimer.split()
    line = ""

    for word in words:

        test_line = line + " " + word

        if len(test_line) > 85:

            pdf.drawString(60, y, line.strip())

            y -= 15

            line = word

        else:

            line = test_line

    if line:
        pdf.drawString(60, y, line.strip())

    pdf.save()

    return filepath


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template("index.html")


# ============================================================
# PREDICTION
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    patient_name = request.form.get("patient_name", "").strip()

    age = request.form.get("age", "").strip()

    hospital_id = request.form.get("hospital_id", "").strip()

    doctor_name = request.form.get("doctor_name", "").strip()

    image = request.files.get("image")

    if not patient_name or not age or not hospital_id or not doctor_name:

        return "Please fill all patient details."

    if image is None or image.filename == "":

        return "Please upload an image."

    try:

        age = int(age)

    except ValueError:

        return "Age must be a number."

    filename = image.filename

    safe_filename = (
        datetime.now().strftime("%Y%m%d%H%M%S")
        + "_"
        + filename
    )

    image_path = os.path.join(
        UPLOAD_FOLDER,
        safe_filename
    )

    image.save(image_path)

    try:

        prediction, confidence, risk = predict_image(
            image_path
        )

    except Exception as e:

        return f"Prediction error: {str(e)}"

    record_id = save_prediction(
        patient_name,
        age,
        hospital_id,
        doctor_name,
        safe_filename,
        prediction,
        confidence,
        risk
    )

    report_path = generate_pdf(
        record_id,
        patient_name,
        age,
        hospital_id,
        doctor_name,
        prediction,
        confidence,
        risk
    )

    return render_template(
        "result.html",
        record_id=record_id,
        patient_name=patient_name,
        age=age,
        hospital_id=hospital_id,
        doctor_name=doctor_name,
        prediction=prediction,
        confidence=confidence,
        risk=risk,
        report_filename=os.path.basename(report_path)
    )


# ============================================================
# HISTORY
# ============================================================

@app.route("/history")
def history():

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id,
               patient_name,
               age,
               hospital_id,
               doctor_name,
               prediction,
               confidence,
               risk,
               timestamp
        FROM predictions
        ORDER BY id DESC
    """)

    records = cursor.fetchall()

    conn.close()

    print("History records:", records)

    return render_template("history.html", records=records)

# ============================================================
# DOWNLOAD REPORT
# ============================================================

@app.route("/download/<filename>")
def download_report(filename):

    filepath = os.path.join(
        REPORT_FOLDER,
        filename
    )

    if not os.path.exists(filepath):

        return "Report not found."

    return send_file(
        filepath,
        as_attachment=True
    )


# ============================================================
# START APPLICATION
# ============================================================

init_database()


if __name__ == "__main__":

    print("\n======================================")
    print(" CERVICAL CELL AI APPLICATION")
    print("======================================")
    print("Open: http://127.0.0.1:5000")
    print("======================================\n")

    app.run(
        debug=True
    )