from flask import Flask, request, render_template
import pandas as pd
import pickle

app = Flask(__name__)

# Load model once
with open('linear_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        score1 = float(request.form['score1'])
        score2 = float(request.form['score2'])
        input_df = pd.DataFrame([[score1, score2]], columns=['PreviousScore1', 'PreviousScore2'])
        
        pred = model.predict(input_df)[0]
        predicted_label = 1 if pred >= 0.6 else 0
        
        result = "Pass" if predicted_label == 1 else "Fail"
        return render_template('index.html', prediction=result, score1=score1, score2=score2)
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)
