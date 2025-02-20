import pandas as pd
import pickle
from flask import Flask, request, jsonify, render_template

# Create Flask app
app = Flask(__name__)

# Load dataset (English-French mapping)
try:
    df = pd.read_csv("eng_-french.csv")
    print("✅ Dataset loaded successfully.")
except FileNotFoundError:
    df = None
    print("❌ Dataset file not found!")

# Ensure correct column names
if df is not None:
    df.rename(columns={"English words/sentences": "english", "French words/sentences": "french"}, inplace=True)

    # Create a dictionary for fast translation lookup (case-insensitive & whitespace-trimmed)
    translation_dict = {str(key).strip().lower(): str(value).strip() for key, value in zip(df["english"], df["french"])}
else:
    translation_dict = {}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/translate", methods=["POST"])
def translate():
    if request.method == "POST":
        english_text = request.form["english_text"].strip().lower()  # Normalize input

        # Look up translation
        french_translation = translation_dict.get(english_text, "Translation not found")

        return jsonify({"english": english_text, "french": french_translation})  # Return JSON response

if __name__ == "__main__":
    app.run(debug=True)

