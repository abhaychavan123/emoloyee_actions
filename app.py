import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Dynamic path loader for Vercel deployment
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

possible_paths = [
    os.path.join(BASE_DIR, "logistic_model.pkl"),
    os.path.join(BASE_DIR, "Logistic_model.pkl"),
]

MODEL_PATH = None
for path in possible_paths:
    if os.path.exists(path):
        MODEL_PATH = path
        break

if not MODEL_PATH:
    raise FileNotFoundError(
        f"Model file not found! Checked: {possible_paths}. "
        "Ensure logistic_model.pkl is committed to your repository."
    )

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Embedded Responsive Glassmorphism UI
HTML_LAYOUT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Employee Retention Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        body {
            background: linear-gradient(-45deg, #0f172a, #1e1b4b, #311042, #0284c7);
            background-size: 400% 400%;
            animation: gradientBG 12s ease infinite;
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 24px;
        }

        @keyframes gradientBG {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        .container {
            background: rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.18);
            border-radius: 24px;
            padding: 35px 40px;
            width: 100%;
            max-width: 680px;
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.4);
            animation: slideUp 0.8s cubic-bezier(0.16, 1, 0.3, 1);
        }

        @keyframes slideUp {
            0% { opacity: 0; transform: translateY(40px); }
            100% { opacity: 1; transform: translateY(0); }
        }

        h2 {
            text-align: center;
            color: #ffffff;
            margin-bottom: 24px;
            font-weight: 700;
            letter-spacing: -0.5px;
            text-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }

        .grid-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
        }

        .form-group {
            display: flex;
            flex-direction: column;
        }

        .full-width {
            grid-column: span 2;
        }

        label {
            color: #cbd5e1;
            font-size: 0.82rem;
            margin-bottom: 6px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        input, select {
            padding: 12px 16px;
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            background: rgba(255, 255, 255, 0.1);
            color: #ffffff;
            font-size: 0.95rem;
            outline: none;
            transition: all 0.3s ease;
        }

        input::placeholder { color: rgba(255, 255, 255, 0.5); }

        option { background: #0f172a; color: #ffffff; }

        input:focus, select:focus {
            background: rgba(255, 255, 255, 0.2);
            border-color: #38bdf8;
            box-shadow: 0 0 15px rgba(56, 189, 248, 0.4);
            transform: translateY(-2px);
        }

        .submit-btn {
            grid-column: span 2;
            margin-top: 10px;
            padding: 14px;
            border: none;
            border-radius: 14px;
            background: linear-gradient(135deg, #38bdf8, #818cf8);
            color: #ffffff;
            font-weight: 700;
            font-size: 1rem;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 20px rgba(56, 189, 248, 0.3);
        }

        .submit-btn:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(56, 189, 248, 0.5);
            background: linear-gradient(135deg, #0284c7, #6366f1);
        }

        .result-card {
            margin-top: 25px;
            padding: 18px;
            border-radius: 14px;
            text-align: center;
            background: rgba(255, 255, 255, 0.95);
            color: #0f172a;
            font-weight: 700;
            animation: popIn 0.5s ease;
        }

        @keyframes popIn {
            0% { transform: scale(0.9); opacity: 0; }
            100% { transform: scale(1); opacity: 1; }
        }

        @media(max-width: 600px) {
            .grid-container { grid-template-columns: 1fr; }
            .full-width { grid-column: span 1; }
            .submit-btn { grid-column: span 1; }
        }
    </style>
</head>
<body>

<div class="container">
    <h2>Logistic Regression Classifier</h2>
    <form method="POST" action="/predict" class="grid-container">
        
        <div class="form-group">
            <label>Education</label>
            <select name="education" required>
                <option value="0">Bachelors</option>
                <option value="1">Masters</option>
                <option value="2">PHD</option>
            </select>
        </div>

        <div class="form-group">
            <label>Joining Year</label>
            <input type="number" name="joining_year" placeholder="e.g. 2017" required>
        </div>

        <div class="form-group">
            <label>City</label>
            <select name="city" required>
                <option value="0">Bangalore</option>
                <option value="1">Pune</option>
                <option value="2">New Delhi</option>
            </select>
        </div>

        <div class="form-group">
            <label>Payment Tier</label>
            <select name="payment_tier" required>
                <option value="1">Tier 1</option>
                <option value="2">Tier 2</option>
                <option value="3">Tier 3</option>
            </select>
        </div>

        <div class="form-group">
            <label>Age</label>
            <input type="number" name="age" placeholder="e.g. 28" min="18" required>
        </div>

        <div class="form-group">
            <label>Gender</label>
            <select name="gender" required>
                <option value="1">Male</option>
                <option value="0">Female</option>
            </select>
        </div>

        <div class="form-group">
            <label>Ever Benched</label>
            <select name="ever_benched" required>
                <option value="0">No</option>
                <option value="1">Yes</option>
            </select>
        </div>

        <div class="form-group">
            <label>Experience in Domain (Years)</label>
            <input type="number" name="experience" placeholder="e.g. 3" min="0" required>
        </div>

        <button type="submit" class="submit-btn">Run Prediction</button>
    </form>

    {% if prediction_text %}
    <div class="result-card">
        <h3>Result: {{ prediction_text }}</h3>
    </div>
    {% endif %}
</div>

</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_LAYOUT)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = [
            float(request.form["education"]),
            float(request.form["joining_year"]),
            float(request.form["city"]),
            float(request.form["payment_tier"]),
            float(request.form["age"]),
            float(request.form["gender"]),
            float(request.form["ever_benched"]),
            float(request.form["experience"]),
        ]
        
        input_array = np.array([data])
        prediction = model.predict(input_array)
        
        return render_template_string(HTML_LAYOUT, prediction_text=str(prediction[0]))
    
    except Exception as e:
        return render_template_string(HTML_LAYOUT, prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
