# ============================================================
# app.py
# This is the Flask web application.
# It loads the saved ML model and:
#   - Shows the home page (index.html)
#   - Takes user input from the form
#   - Makes a sales prediction
#   - Returns the result to the page
# ============================================================

from flask import Flask, render_template, request
import numpy as np
import joblib
import os

# ── Create the Flask app ──
app = Flask(__name__)

# ── Load the trained model ──
MODEL_PATH = os.path.join("models", "sales_model.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Model file not found!\n"
        "Please run  python train_model.py  first."
    )

saved = joblib.load(MODEL_PATH)
model    = saved["model"]
features = saved["features"]   # list of column names the model was trained on

print(f"[OK] Model loaded.  Features expected: {features}\n")


# ── Helper: build a feature vector from user inputs ──
def build_input_vector(form_data: dict) -> np.ndarray:
    """
    Map the HTML form fields to the exact feature order
    the model was trained on.
    Any feature not provided by the form is filled with 0.
    """
    row = {}
    for feat in features:
        row[feat] = float(form_data.get(feat, 0))
    return np.array([list(row.values())])


# ────────────────────────────────────────────
# ROUTE 1 – Home page  (GET)
# ────────────────────────────────────────────
@app.route("/")
def index():
    # Collect available chart images to show in the UI
    plot_dir   = os.path.join("static", "plots")
    plot_files = []
    if os.path.exists(plot_dir):
        plot_files = [
            f"plots/{f}" for f in os.listdir(plot_dir) if f.endswith(".png")
        ]
    return render_template("index.html",
                           features=features,
                           prediction=None,
                           error=None,
                           plots=plot_files)


# ────────────────────────────────────────────
# ROUTE 2 – Predict  (POST)
# ────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    plot_dir   = os.path.join("static", "plots")
    plot_files = []
    if os.path.exists(plot_dir):
        plot_files = [
            f"plots/{f}" for f in os.listdir(plot_dir) if f.endswith(".png")
        ]

    try:
        # Build feature vector from submitted form
        X_input = build_input_vector(request.form)

        # Make the prediction
        pred_value = model.predict(X_input)[0]
        prediction = round(float(pred_value), 2)

        return render_template("index.html",
                               features=features,
                               prediction=prediction,
                               error=None,
                               plots=plot_files)

    except Exception as e:
        # If something goes wrong, show an error message
        return render_template("index.html",
                               features=features,
                               prediction=None,
                               error=str(e),
                               plots=plot_files)


# ── Run the app ──
if __name__ == "__main__":
    print("Starting Flask app…  Open  http://127.0.0.1:5000  in your browser.\n")
    app.run(debug=True)
