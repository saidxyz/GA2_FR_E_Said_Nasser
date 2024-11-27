import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

###-------- TASK 2 --------###


# 2a) function to read the data and create a data set using sliding window approach
def load_and_prepare_data(filename, window_size):
    data = pd.read_csv(filename)

    # creating the dataset using sliding window approach
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data["Demand"][i : (i + window_size)].values)
        y.append(data["Demand"][i + window_size])

    return data, np.array(X), np.array(y)


window_size = 10

data_2022, X_2022, y_2022 = load_and_prepare_data("data_2022.csv", window_size)


# 2b) train a single regression tree on the dataset
# train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_2022, y_2022, test_size=0.2, random_state=0
)


def train_decision_tree(X, y):
    model = DecisionTreeRegressor(random_state=0)
    model.fit(X, y)
    return model


model_2022 = train_decision_tree(X_train, y_train)


# 2c) using the regression tree to predict the next value and plot results
def make_predictions_task2(model, X):
    predictions = []
    for i in range(len(X)):
        predictions.append(model.predict(X[i].reshape(1, -1))[0])
    return predictions


decision_tree_predictions = make_predictions_task2(model_2022, X_test)


def plot_decision_tree(predictions, y_test, data):
    plt.figure(figsize=(14, 7))

    num_test_points = len(predictions)

    test_dates = data["Date"].iloc[-num_test_points:]

    plt.plot(test_dates, y_test, label="Original Data", color="blue")
    plt.plot(test_dates, predictions, label="Predictions", color="red", linestyle="--")
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(7))

    plt.xticks(rotation=45)
    plt.xlabel("Date")
    plt.ylabel("Demand")
    plt.title("Original Demand vs Predictions")
    plt.legend()
    plt.tight_layout()
    plt.show()


plot_decision_tree(decision_tree_predictions, y_test, data_2022)


###-------- TASK 3 --------###


# trying some different hyperparamaters for varience
def create_random_hyperparameters():
    max_depth_range = range(5, 20)
    min_samples_split_range = range(2, 10)
    max_features_range = [
        "sqrt",
        "log2",
        None,
    ]

    max_depth = np.random.choice(max_depth_range)
    min_samples_split = np.random.choice(min_samples_split_range)
    max_features = np.random.choice(max_features_range)

    return {
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "max_features": max_features,
    }


def create_ensemble(X, y, n_estimators, window_size):
    trees = []
    for _ in range(n_estimators):
        hyperparameters = create_random_hyperparameters()

        tree = DecisionTreeRegressor(
            max_depth=hyperparameters["max_depth"],
            min_samples_split=hyperparameters["min_samples_split"],
            max_features=hyperparameters["max_features"],
            random_state=0,
        )

        # bootstrapping
        X_sample, y_sample = bootstrap_sample(X, y)

        # Train the tree on the bootstrapped sample
        tree.fit(X_sample, y_sample)
        trees.append(tree)
    return trees


# func that create bootstrapped sample of the dataset
def bootstrap_sample(X, y):
    indices = np.random.choice(range(len(X)), size=len(X), replace=True)
    return X[indices], y[indices]


#  make predictions with the ensemble
def predict_ensemble(trees, X):
    predictions = np.array([tree.predict(X) for tree in trees])
    return np.mean(predictions, axis=0)


# Create the ensemble of decision trees
n_estimators = 100
trees_2022 = create_ensemble(X_train, y_train, n_estimators, window_size)

ensemble_predictions = predict_ensemble(trees_2022, X_test)


def plot_random_forest(decision_tree_predictions, ensemble_predictions, y_test, data):
    plt.figure(figsize=(14, 7))

    num_test_points_decision_tree = len(decision_tree_predictions)
    num_test_points_ensemble = len(ensemble_predictions)

    if num_test_points_decision_tree != num_test_points_ensemble:
        return "Test sets for decision tree and ensemble needs to be the same length"

    test_dates = data["Date"].iloc[-num_test_points_decision_tree:]

    plt.plot(test_dates, y_test, label="Original Data", color="blue")

    # plot the predictions from Task 2
    plt.plot(
        test_dates,
        decision_tree_predictions,
        label="Decision Tree Predictions",
        color="green",
        linestyle="--",
    )

    # Plot the ensemble predictions from Task 3
    plt.plot(
        test_dates,
        ensemble_predictions,
        label="Ensemble Predictions",
        color="red",
        linestyle="--",
    )

    # Formatting the plot
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(7))
    plt.xticks(rotation=45)
    plt.xlabel("Date")
    plt.ylabel("Demand")
    plt.title("Original Demand vs Decision Tree Predictions vs Ensemble Predictions")
    plt.legend()
    plt.tight_layout()
    plt.show()


plot_random_forest(decision_tree_predictions, ensemble_predictions, y_test, data_2022)


###-------- TASK 4 --------###


# Task 4a)
def create_mlp_ensemble(n_estimators):
    mlps = []

    for _ in range(n_estimators):
        mlp = MLPRegressor(
            max_iter=1000,
            hidden_layer_sizes=(100,),
            random_state=0,
        )
        mlps.append(mlp)
    return mlps


# Task 4b)
def train_mlp_ensemble(mlps, X_train, y_train):
    for mlp in mlps:
        # Create a bootstrap sample for each MLP
        X_sample, y_sample = bootstrap_sample(X_train, y_train)
        mlp.fit(X_sample, y_sample)
        print("training mlp ensemble...")
    return mlps


def predict_with_mlp_ensemble(mlps, X):
    predictions = np.array([mlp.predict(X) for mlp in mlps])
    return np.mean(predictions, axis=0)


n_estimators_mlp = 100

# Create, train and predict the ensemble of MLPRegressors
mlps = create_mlp_ensemble(n_estimators_mlp)
mlp_ensemble_trained = train_mlp_ensemble(mlps, X_train, y_train)
mlp_ensemble_predictions = predict_with_mlp_ensemble(mlp_ensemble_trained, X_test)


def plot_mlp(
    decision_tree_predictions, ensemble_predictions, mlp_predictions, y_test, data
):
    plt.figure(figsize=(14, 7))

    num_test_points_decision_tree = len(decision_tree_predictions)
    num_test_points_ensemble = len(ensemble_predictions)

    if num_test_points_decision_tree != num_test_points_ensemble:
        return "Test sets for decision tree and ensemble needs to be the same length"

    test_dates = data["Date"].iloc[-num_test_points_decision_tree:]

    plt.plot(test_dates, y_test, label="Original Data", color="blue")

    # plot the predictions from Task 2
    plt.plot(
        test_dates,
        decision_tree_predictions,
        label="Decision Tree Predictions",
        color="green",
        linestyle="--",
    )

    # Plot the ensemble predictions from Task 3
    plt.plot(
        test_dates,
        ensemble_predictions,
        label="Ensemble Predictions",
        color="red",
        linestyle="--",
    )
    # plot mlp predictions from task 4
    plt.plot(
        test_dates,
        mlp_predictions,
        label="MLP Ensemble Predictions",
        color="purple",
        linestyle="--",
    )

    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(7))
    plt.xticks(rotation=45)
    plt.xlabel("Date")
    plt.ylabel("Demand")
    plt.title(
        "Original Demand vs Decision Tree Predictions vs Ensemble Predictions vs MLP Ensemble Predictions"
    )
    plt.legend()
    plt.tight_layout()
    plt.show()


plot_mlp(
    decision_tree_predictions,
    ensemble_predictions,
    mlp_ensemble_predictions,
    y_test,
    data_2022,
)


# ###-------- TASK 5 --------###


# Task 5a)
def load_and_prepare_data_2023(filename, window_size, end_date):
    data = pd.read_csv(filename)
    data["Date"] = pd.to_datetime(data["Date"], dayfirst=True)
    data = data[data["Date"] <= pd.to_datetime(end_date)]

    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data.iloc[i : i + window_size]["Demand"].values)
        y.append(data.iloc[i + window_size]["Demand"])

    return data, np.array(X), np.array(y)


window_size = 10
end_date = "2023-08-25"
data_2023, X_2023, y_2023 = load_and_prepare_data_2023(
    "data_2023.csv", window_size, end_date
)


# Task 5b)
def predict_demand(ensemble, initial_data, start_date, end_date):
    predictions = []
    current_data = initial_data.copy()

    current_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    while current_date <= end_date:
        X_new = current_data[-window_size:].reshape(1, -1)

        prediction = make_predictions_task2(ensemble, X_new)[0]
        predictions.append(prediction)

        current_data = np.append(current_data, prediction)

        current_date += pd.Timedelta(days=1)

    return predictions


# predict the demand from 26.08.2023 to 31.12.2023
start_predict_date = "2023-08-26"
end_predict_date = "2023-12-31"
predictions_2023 = predict_demand(
    model_2022, X_2023[-1], start_predict_date, end_predict_date
)


# Task 5c)
def plot_predictions_2023(start_date, end_date, predictions):
    plt.figure(figsize=(14, 7))
    dates = pd.date_range(start=start_date, end=end_date, freq="D")

    plt.plot(
        dates,
        predictions,
        label="Predictions for 2023",
        color="magenta",
        linestyle="--",
    )
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(7))

    plt.xlabel("Date")
    plt.ylabel("Demand")
    plt.title("Forecasted Demand from 26.08.2023 to 31.12.2023")
    plt.legend()
    plt.tight_layout()
    plt.show()


plot_predictions_2023(start_predict_date, end_predict_date, predictions_2023)