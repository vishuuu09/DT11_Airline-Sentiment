from flask import Flask, render_template, request, jsonify
import pickle
from test import TextToNum  # Your preprocessing function

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")  # Landing Page

@app.route("/sentiment", methods=["GET"])
def sentiment_page():
    return render_template("predict.html")  # Sentiment Analysis Page

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        message = request.form["message"]
        
        # Preprocessing using your TextToNum class
        ob = TextToNum(message)
        ob.cleaner()
        ob.token()
        ob.removeStop()
        st = ob.stemme()

        # Load vectorizer
        with open("vectorizer.pickle", "rb") as vcfile:
            vc = pickle.load(vcfile)
        
        # Transform input text
        stvc = " ".join(st)
        data = vc.transform([stvc])
        
        # Load model and predict
        with open("model.pickle", "rb") as mbfile:
            model = pickle.load(mbfile)
        pred = model.predict(data)

        return jsonify({"result": str(pred[0])})  # Returns JSON

if __name__ == "__main__":
    app.run(debug=True)

