from nltk.sentiment.vader import SentimentIntensityAnalyzer

class SentimentAnalyzer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def analyze(self, text: str) -> dict:
        if not text.strip():
            return {
                "text": text,
                "sentiment": "Invalid",
                "score": "0.00"
            }

        score = self.analyzer.polarity_scores(text)["compound"]

        if score >= 0.05:
            sentiment = "Positive"
        elif score <= -0.05:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        return {
            "text": text,
            "sentiment": sentiment,
            "score": f"{abs(score) * 100:.2f}"
        }

