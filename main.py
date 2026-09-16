from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib

app = FastAPI()

model = joblib.load("SpamModel_2.pkl")
vectorizer = joblib.load("vectorizer.pkl")


class Message(BaseModel):
    text: str


HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Spam Classifier</title>
    <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='0.9em' font-size='90'>&#128680;</text></svg>">
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            max-width: 640px;
            margin: 0 auto;
            padding: 40px 20px;
            background: #f4f5f7;
            color: #1f2933;
        }
        h1 { font-size: 1.6rem; margin-bottom: 4px; }
        p.subtitle { color: #616e7c; margin-top: 0; }
        textarea {
            width: 100%;
            box-sizing: border-box;
            padding: 12px;
            font-size: 1rem;
            border: 1px solid #cbd2d9;
            border-radius: 8px;
            resize: vertical;
            min-height: 120px;
        }
        button {
            margin-top: 12px;
            padding: 10px 24px;
            font-size: 1rem;
            border: none;
            border-radius: 8px;
            background: #2c7be5;
            color: #fff;
            cursor: pointer;
        }
        button:hover { background: #1f6fd0; }
        button:disabled { opacity: 0.6; cursor: not-allowed; }
        .result {
            margin-top: 20px;
            padding: 16px;
            border-radius: 8px;
            font-weight: 600;
            display: none;
        }
        .spam { background: #ffe3e3; color: #c92a2a; }
        .not-spam { background: #e3fbe3; color: #2b8a3e; }
    </style>
</head>
<body>
    <h1>Spam Classifier</h1>
    <p class="subtitle">Paste a message below to check if it is spam.</p>
    <textarea id="text" placeholder="Type or paste your message here..."></textarea>
    <br>
    <button id="btn" onclick="predict()">Check message</button>
    <div id="result" class="result"></div>

    <script>
        async function predict() {
            const text = document.getElementById("text").value.trim();
            const btn = document.getElementById("btn");
            const result = document.getElementById("result");

            if (!text) {
                result.className = "result spam";
                result.textContent = "Please enter a message.";
                result.style.display = "block";
                return;
            }

            btn.disabled = true;
            btn.textContent = "Checking...";
            result.style.display = "none";

            try {
                const res = await fetch("/predict", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ text: text })
                });
                const data = await res.json();
                result.className = "result " + (data.label === "Spam" ? "spam" : "not-spam");
                result.textContent = data.label;
            } catch (err) {
                result.className = "result spam";
                result.textContent = "Error: " + err;
            } finally {
                btn.disabled = false;
                btn.textContent = "Check message";
                result.style.display = "block";
            }
        }
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def home():
    return HTML


@app.post("/predict")
def predict(message: Message):

    X = vectorizer.transform([message.text])

    prediction = model.predict(X)

    return {
        "prediction": int(prediction[0]),
        "label": "Spam" if prediction[0] == 1 else "Not Spam"
    }