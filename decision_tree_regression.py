"""
This python binary runs the DecisionTreeRegressor class from the
scikit-learn library and demonstrates its usage in the Boston Housing Dataset.

Author: David Damian Galan
Date: Sep 17, 2023
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

if __name__ == "__main__":
    # Read the dataset
    boston = pd.read_csv("hou_all.csv", header=None)
    # Assign column names
    col_name = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS",
                "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV", "NONE"]
    boston.columns = col_name

    # Drop the extra column in the dataset
    boston = boston.drop(columns=["NONE"])

    # Divide the dataset in train, validation and test
    # Column to predict is MEDV, median value of a house
    x_train_tmp, x_test, y_train_tmp, y_test = train_test_split(
        boston.drop(columns=["MEDV"]),
        boston["MEDV"],
        test_size=0.2
    )
    # Divide the training set in training and validation set
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_tmp, y_train_tmp, test_size=0.2
    )

    # Make several iterations with the validation set to find the best
    # hyperparameter
    best_max_depth = -1
    best_mse = float("inf")

    # List to keep track of the mse, mae and r2 scores
    mse_scores_val = []
    mse_scores_train = []
    mae_scores_val = []
    mae_scores_train = []
    r2_scores_val = []
    r2_scores_train = []

    for max_depth in range(1, 11):
        # Build the Decision Tree
        tree = DecisionTreeRegressor(max_depth=max_depth)
        tree.fit(x_train, y_train)

        # Make predictions with training set
        y_pred = tree.predict(x_train)
        mse_train = mean_squared_error(y_train, y_pred)
        mae_train = mean_absolute_error(y_train, y_pred)
        r2_train = r2_score(y_train, y_pred)
        r2_scores_train.append(r2_train)
        mse_scores_train.append(mse_train)
        mae_scores_train.append(mae_train)

        # Make predictions with validation set
        y_pred = tree.predict(x_val)

        # Measure quality with MSE and R^2
        mse = mean_squared_error(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        r2_scores_val.append(r2)
        mse_scores_val.append(mse)
        mae_scores_val.append(mae)

        # Select best MSE with validation set
        if (mse < best_mse):
            best_mse = mse
            best_max_depth = max_depth

        # Output statistics
        print(f"*** Max depth value: {max_depth} ***")
        print(f"Mean squared error in validation set= {mse}")
        print(f"Coefficient of determination in validation set = {r2}")

    # Construct best tree
    best_tree = DecisionTreeRegressor(max_depth=max_depth)
    best_tree.fit(x_train, y_train)
    y_pred = tree.predict(x_test)

    # Evaluate final model with the training set
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"FINAL MODEL")
    print(f"*** Max depth value: {best_max_depth} ***")
    print(f"Mean squared error in test set = {mse}")
    print(f"Coefficient of determination in test set = {r2}")

    # Validation curve for Mean squared error
    plt.figure()
    plt.title("Validation Curve for Decision Tree Regressor")
    plt.xlabel("Max Depth")
    plt.ylabel("Mean squared error")
    plt.plot(range(1, 11), mse_scores_train, label="Training score", color="b")
    plt.plot(range(1, 11), mse_scores_val, label="Validation score", color="r")
    plt.legend()
    plt.show()

    # Mean absolute error
    plt.figure()
    plt.title("Validation Curve for Decision Tree Regressor")
    plt.xlabel("Max Depth")
    plt.ylabel("Mean absolute error")
    plt.plot(range(1, 11), mae_scores_train, label="Training score", color="b")
    plt.plot(range(1, 11), mae_scores_val, label="Validation score", color="r")
    plt.legend()
    plt.show()

    # Coefficient of determination
    plt.figure()
    plt.title("Coefficient of determination for Decision Tree Regressor")
    plt.xlabel("Max Depth")
    plt.ylabel("Coefficient of determination")
    plt.plot(range(1, 11), r2_scores_train, label="Training R2", color="b")
    plt.plot(range(1, 11), r2_scores_val, label="Validation R2", color="r")
    plt.legend()
    plt.show()
