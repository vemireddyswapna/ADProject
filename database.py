import sqlite3

# Connect to database
conn = sqlite3.connect("history/history.db")

cursor = conn.cursor()

# Create table if it doesn't exist
cursor.execute("""
CREATE TABLE IF NOT EXISTS prediction_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_name TEXT,
    prediction TEXT,
    confidence REAL,
    risk_level TEXT,
    date_time TEXT
)
""")

conn.commit()


def save_prediction(patient_name, prediction, confidence, risk_level, date_time):
    cursor.execute("""
    INSERT INTO prediction_history
    (patient_name, prediction, confidence, risk_level, date_time)
    VALUES (?, ?, ?, ?, ?)
    """, (patient_name, prediction, confidence, risk_level, date_time))

    conn.commit()


def view_history():
    cursor.execute("SELECT * FROM prediction_history")
    return cursor.fetchall()