from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Charger le modèle et le vectorizer
model = joblib.load("model/iphone_price_model.pkl")
vectorizer = joblib.load("model/iphone_price_vectorizer.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    predicted_price = None
    description = ""

    if request.method == "POST":
        description = request.form["description"]

        # Transformer la description
        vect = vectorizer.transform([description])

        # Prédire le prix
        predicted_price = round(model.predict(vect)[0], 2)

    return render_template(
        "index.html",
        predicted_price=predicted_price,
        description=description
    )

if __name__ == "__main__":
    app.run(debug=True)
