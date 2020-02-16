from sklearn import datasets, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error
import matplotlib.pyplot as plt
import numpy as np


def load_data():
    boston = datasets.load_boston()
    X = boston.data
    y = boston.target
    features = boston.feature_names
    return X, y, features


def visualize(X, y, features):
    plt.figure(figsize=(20, 5))
    feature_count = X.shape[1]

    # i: index
    for i in range(feature_count):
        plt.subplot(3, 5, i + 1)
        # TODO: Plot feature i against y
        plt.scatter(X[:, i], y)
        plt.title(features[i] + " V.S. MEDV")

        plt.xlabel(features[i])
        plt.ylabel("MEDV")

        # plt.ylabel(features[i])
        # plt.xlabel("MEDV")

    plt.tight_layout()
    plt.show()


def add_one_column(X):
    X_biased = []
    for i in range(len(X)):
        # print(type(X[i]))
        # print(len([1]+X[i].tolist()))
        X_biased.append(np.array([1]+X[i].tolist()))
    X_biased = np.array(X_biased)
    return X_biased


def fit_regression(X, Y):
    # TODO: implement linear regression
    # Remember to use np.linalg.solve instead of inverting!
    X_biased = add_one_column(X)

    # print(X_biased)
    # print(X_biased.shape)

    w = np.linalg.solve(np.dot(np.transpose(X_biased), X_biased),
                        np.dot(np.transpose(X_biased), Y))
    print(len(w))
    return w


def main():
    # Load the data
    X, y, features = load_data()
    print("Features: {}".format(features))
    print("Number of data points: ", len(X))
    print("Dimension: ", len(X[0]))

    # Visualize the features
    visualize(X, y, features)

    # TODO: Split data into train and test

    X_norm = preprocessing.normalize(X, norm='l2')
    train_data, test_data, train_targets, test_targets = train_test_split(
        X_norm, y, test_size=0.3)

    # Fit regression model
    w = fit_regression(train_data, train_targets)
    print("w:")
    print(w)

    # Compute fitted values, MSE, etc.
    preds = []
    test_data_biased = add_one_column(test_data)
    for i in range(len(test_data)):
        preds.append(np.dot(w, test_data_biased[i]))
    preds = np.array(preds)
    mse = mean_squared_error(preds, test_targets)
    mae = mean_absolute_error(preds, test_targets)
    mad = median_absolute_error(preds, test_targets)
    print("Mean Squared Error: ", mse)
    print("Mean Absolute Error: ", mae)
    print("Median Absolute Error: ", mad)

    # print("Standard Error; ")


if __name__ == "__main__":
    main()
