import random

class Predictor:
    def predict(self, team1, team2):
        # Randomly choose one of the two teams as the predicted winner
        predicted_winner = random.choice([team1, team2])
        return {
            'team1': team1,
            'team2': team2,
        } 