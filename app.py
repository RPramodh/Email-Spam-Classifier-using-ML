# from flask import Flask, render_template, request

# app=Flask(__name__)

# @app.route("/",methods=['GET', 'POST'])
# def home():
#     text = ""
#     if request.method == 'POST':
#         text = request.form.get('email-content')
#     return render_template("index.html",text=text)



# if __name__=="__main__":
#     app.run(debug=True)


from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
cv = pickle.load(open("model/cv.pkl","rb"))
clf = pickle.load(open("model/clf.pkl","rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    email = request.form.get('email-content')
    tokenized_email = cv.transform([email]) # X 
    prediction = clf.predict(tokenized_email)
    prediction = "Spam" if prediction == 1 else "Not a Spam"
    return render_template("index.html", prediction=prediction, email=email)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
