from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from sentiment_model import SentimentAnalyzer
from pydantic import BaseModel
from typing import List

app = FastAPI()
model = SentimentAnalyzer()

# HTML form template
html_form = """
<!DOCTYPE html>
<html>
<head>
    <title>Fast Sentiment Analyzer</title>
</head>
<body>
    <h2>Enter any text:</h2>
    <form action="/analyze" method="post">
        <textarea name="sentence" rows="6" cols="60" required></textarea><br><br>
        <button type="submit">Analyze</button>
    </form>
</body>
</html>
"""

# HTML form GET route
@app.get("/", response_class=HTMLResponse)
async def root():
    return html_form

# HTML form POST route
@app.post("/analyze", response_class=HTMLResponse)
async def analyze(sentence: str = Form(...)):
    sentiment = model.analyze(sentence)
    return f"""
    <html>
        <body>
            <h2>Input:</h2>
            <p>{sentence}</p>
            <h3>Predicted Sentiment: <b>{sentiment}</b></h3>
            <a href="/">Try Another</a>
        </body>
    </html>
    """

# Updated input model to accept list of strings
class SentimentArrayRequest(BaseModel):
    texts: List[str]

# Updated API endpoint to handle array of texts
@app.post("/predict")
async def predict_sentiments(request: SentimentArrayRequest):
    results = []
    for text in request.texts:
        sentiment = model.analyze(text)
        results.append({
            "input": text,
            "sentiment": sentiment
        })
    return {"results": results}

class SentimentRequest(BaseModel):
    text: str

# API for external apps (like your Node.js function)
@app.post("/api/analyze-sentiment")
async def analyze_sentiment_api(request: SentimentRequest):
    result = model.analyze(request.text)
    
    # Convert sentiment to label format expected by the Node.js app
    if result["sentiment"].lower() == "negative":
        label = "LABEL_0"
    elif result["sentiment"].lower() == "neutral":
        label = "LABEL_1"
    else:
        label = "LABEL_2"
    
    return {
        "result": [
            {
                "label": label,
                "score": float(result["score"])  # Return score as decimal
            }
        ]
    }
    
@app.post("/api/analyze-sentiment-batch")
async def analyze_batch(request: SentimentArrayRequest):
    results = []
    for text in request.texts:
        result = model.analyze(text)
        if result["sentiment"] == "Negative":
            label = "LABEL_0"
        elif result["sentiment"] == "Neutral":
            label = "LABEL_1"
        else:
            label = "LABEL_2"
        results.append({
            "label": label,
            "score": float(result["score"]) / 100
        })
    return {"result": results}
