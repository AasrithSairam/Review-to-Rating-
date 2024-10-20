from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the model and vectorizer
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

def preprocess_text(text):
    text = text.lower()
    return text

@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_rating = None
    if request.method == 'POST':
        review = request.form['review']
        cleaned_review = preprocess_text(review)
        vectorized_review = vectorizer.transform([cleaned_review])
        predicted_rating = model.predict(vectorized_review)
        predicted_rating = round(predicted_rating[0])
    return render_template('index.html', predicted_rating=predicted_rating)

if __name__ == '__main__':
    app.run(debug=True)
