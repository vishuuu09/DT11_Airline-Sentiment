from flask import Flask, render_template, request, jsonify
from test import TextToNum
import pickle
import random  # Simulated sentiment analysis

app = Flask(__name__)

# Placeholder sentiment analysis function
def analyze_sentiment(text):
    sentiments = ['Positive', 'Negative', 'Neutral']
    return random.choice(sentiments)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    sentiment = None
    if request.method == "POST":
        # Get message from form
        message = request.form['message']
        print(f"User input: {message}")

        # Text Processing
        ob = TextToNum(message)
        ob.cleaner()
        ob.token()
        ob.removeStop()
        st = ob.stemme()  

        # Load vectorizer and transform text
        with open("vectorizer.pickle", "rb") as vcfile:
            vectorizer = pickle.load(vcfile)
        
        stvc = " ".join(st)
        data = vectorizer.transform([stvc])
        print(data)

        # Load model and predict
        with open("model.pickle", "rb") as mbfile:  # Fixed incorrect file reference
            model = pickle.load(mbfile)
        
        pred = model.predict(data)
        if pred[0]==0:
            return jsonify({"result":"Neutral"})
        elif pred[0]==1:
            return jsonify({"result":"Positive"})
        else:
            return jsonify({"result":"Negative"})
        return jsonify({"result": str(pred[0])})

    return render_template("predict.html", sentiment=sentiment)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050)
