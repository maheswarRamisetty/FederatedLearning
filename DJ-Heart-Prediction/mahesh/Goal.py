import syft as sy
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

# Create a hook for PySyft
hook = sy.TorchHook(torch)

# Initialize virtual workers (clients)
workers = [hook.create_worker(f"worker_{i}") for i in range(3)]  # Creating 3 clients

# Simulated federated dataset (we're generating synthetic data here)
def create_data():
    X, y = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=42)
    return X, y

# Split the data across the clients
def federated_split_data(data, num_clients=3):
    X, y = data
    split_size = len(X) // num_clients
    federated_data = [(X[i*split_size:(i+1)*split_size], y[i*split_size:(i+1)*split_size]) for i in range(num_clients)]
    return federated_data

# A function to train on each client using scikit-learn's Logistic Regression
def train_on_client(client_data):
    X, y = client_data
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model

# Federated Learning Aggregation Function: Average the coefficients of each client model
def aggregate_models(models):
    avg_coefficients = np.mean([model.coef_ for model in models], axis=0)
    avg_intercept = np.mean([model.intercept_ for model in models], axis=0)
    
    # Create a new Logistic Regression model with the averaged coefficients
    aggregated_model = LogisticRegression(max_iter=1000)
    aggregated_model.coef_ = avg_coefficients
    aggregated_model.intercept_ = avg_intercept
    return aggregated_model

# Federated learning training loop
def federated_training(num_rounds=10):
    data = create_data()  # Simulate data generation
    federated_data = federated_split_data(data)
    
    global_model = LogisticRegression(max_iter=1000)  # Initialize a global model
    
    for round_num in range(num_rounds):
        print(f"Round {round_num+1}/{num_rounds}")
        
        # Train the model on each client
        models = []
        for i, client_data in enumerate(federated_data):
            print(f"Training on client {i+1}")
            model = train_on_client(client_data)
            models.append(model)
        
        # Aggregate models on the server
        global_model = aggregate_models(models)
        print(f"Global model updated after round {round_num+1}")
        
        # Evaluate the aggregated model (Optional)
        X_test, y_test = create_data()  # Simulating test data
        y_pred = global_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy after round {round_num+1}: {accuracy}")

if __name__ == "__main__":
    federated_training()
