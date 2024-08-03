from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the trained model and vectorizer
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        text_vectorized = vectorizer.transform([text])
        prediction = model.predict(text_vectorized)[0]
        return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
