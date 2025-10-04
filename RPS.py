import random
from sklearn.linear_model import LogisticRegression
import numpy as np


moves = ["R", "P", "S"]
n = 5

# One-hot encoding function
def one_hot(move):
    if move == "R":
        return np.array([1, 0, 0])
    elif move == "P":
        return np.array([0, 1, 0])
    elif move == "S":
        return np.array([0, 0, 1])
    else:
        return np.array([0, 0, 0])

# Prepare training data
def prepare_data_onehot(history, n):
    X, y = [], []
    for i in range(len(history) - n):
        X.append(np.concatenate([one_hot(m) for m in history[i:i+n]]))
        y.append(history[i+n])
    return np.array(X), np.array(y)

# Predict next move
def predict_next_move(model, history, n):
    last_moves = history[-n:]
    input_vector = np.concatenate([one_hot(m) for m in last_moves]).reshape(1, -1)
    return model.predict(input_vector)[0]

# Counter move (to beat the predicted move)
def counter_move(predicted):
    if predicted == "R":
        return "P"
    elif predicted == "P":
        return "S"
    else:
        return "R"


def player(prev_play, opponent_history=[], model_cache={'model': None, 'last_trained': 0}):
    if not prev_play:
        prev_play = "R"
    
    opponent_history.append(prev_play)
    
    # Need at least n+1 moves to train
    if len(opponent_history) < n + 1:
        return random.choice(moves)
    
    # Only retrain every 10 moves
    if model_cache['model'] is None or len(opponent_history) - model_cache['last_trained'] >= 10:
        # Prepare training data
        X, y = prepare_data_onehot(opponent_history, n)
        
        # Check if we have at least 2 different moves
        if len(np.unique(y)) < 2:
            return random.choice(moves)
        
        # Train model
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        
        # Cache the model
        model_cache['model'] = model
        model_cache['last_trained'] = len(opponent_history)
    
    # Use cached model to predict
    predicted = predict_next_move(model_cache['model'], opponent_history, n)
    
    # Return counter move
    return counter_move(predicted)
