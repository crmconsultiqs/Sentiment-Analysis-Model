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
