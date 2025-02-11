from flask import Flask, render_template, request,jsonify 
from test import TextToNum
import pickle 
import random  # For simulation of sentiment analysis; replace with your model

app = Flask(__name__)  # Fixed the typo

# Sample function to simulate sentiment analysis
def analyze_sentiment(text):
    # This is a placeholder for actual sentiment analysis.
    # For example, you could use a pre-trained model here.
    sentiments = ['Positive', 'Negative', 'Neutral']
    return random.choice(sentiments)  # Randomly choose sentiment for now.

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    sentiment = None
    if request.method == "POST":
        # Get the message from the form
        message = request.form['message']
        
        # Print the user input message to the terminal (debugging purpose)
        print(f"User input: {message}")
        ob=TextToNum(message)
        ob.cleaner()
        ob.token()
        ob.removeStop()
        st=ob.stemme()
        with open("vectorizer.pickle","rb") as vcfile:
            vc=pickle.load(vcfile)
        stvc=" ".join(st)
        data =vc.transform([stvc])
        print(data)
        with open("model.pickle","rb") as mbfile:
            model=pickle.load(mbfile)
        pred=model.predict(data)
        return jsonify({"result":str(pred[0])})

# Call the sentiment analysis function (replace with your actual function)
        sentiment = analyze_sentiment(message)
    
    # Render the page with the sentiment result
    return render_template("predict.html", sentiment=sentiment)

if __name__ == "__main__":  # Fixed the typo
    app.run(host="0.0.0.0", port=5050)
