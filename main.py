from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

model = joblib.load("SpamModel.pkl")
vectorizer = joblib.load("vectorizer.pkl")


class Message(BaseModel):
    text: str


@app.get("/")
def home():
    return {"message": "Model is running"}


@app.post("/predict")
def predict(message: Message):

    X = vectorizer.transform([message.text])

    prediction = model.predict(X)

    return {
        "prediction": int(prediction[0]),
        "label": "Spam" if prediction[0] == 1 else "Not Spam"
    }