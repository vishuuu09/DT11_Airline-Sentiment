from flask import Flask, render_template, request, jsonify
import pickle
from test import TextToNum  # Ensure the correct class name

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        # Get the message from the form
        message = request.form["message"]
        
        # Print the user input message to the terminal (for debugging)
        print(f"User input: {message}")
        
        # Process input using TextToNum class
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

        # Load model
        with open("model.pickle", "rb") as mbfile:  # Ensure correct model filename
            model = pickle.load(mbfile)
        
        # Predict sentiment
        pred = model.predict(data)
        
        return jsonify({"result": str(pred[0])})

    return render_template("predict.html", sentiment=None)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)
