from flask import Flask, jsonify, request
from predictor import Predictor

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict():
    team1 = request.args.get('team1')
    team2 = request.args.get('team2')

    if not team1 or not team2:
        return jsonify({'error': 'Both team1 and team2 parameters are required.'}), 400

    result = predict_result(team1, team2)
    return jsonify(result)

def predict_result(team1, team2):
    predictor = Predictor()
    return predictor.predict(team1, team2)

if __name__ == '__main__':
    # Use command-line input for team names and output the prediction result
    team1 = input("Enter team1 name: ")
    team2 = input("Enter team2 name: ")
    result = predict_result(team1, team2)
    print("Prediction result:", result)
