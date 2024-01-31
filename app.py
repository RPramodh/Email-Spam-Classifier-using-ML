
# TASK -1 - 1st Play with this, Flask based environment, then go for, implementing the ML model
# from flask import Flask, render_template, request

# app=Flask(__name__)

# @app.route("/",methods=['GET', 'POST'])   # for fecthing
# def home():
#     text = ""
#     if request.method == 'POST':     # if any text or parameter to post method
#         text = request.form.get('email-content')
#     return render_template("index.html",text=text)



# if __name__=="__main__":
#     app.run(debug=True)

# Task -2 - This is were, it starts, how to implement the ML model
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
cv = pickle.load(open("model/cv.pkl","rb"))   # for converting the text emails into the numbers
clf = pickle.load(open("model/clf.pkl","rb"))  # for predicting the Output of ML model

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    email = request.form.get('email-content')
    tokenized_email = cv.transform([email]) 
    prediction = clf.predict(tokenized_email)
    prediction = "Spam" if prediction == 1 else "Not a Spam"
    return render_template("index.html", prediction=prediction, email=email)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
