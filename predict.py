import numpy as np
import scipy
import csv as csv
import matplotlib.pyplot as plt


def read_csv_data(file_name):
    list = []
    with open(file_name, 'r') as f:
        data_csv = csv.reader(f, delimiter = ',')
        for row in data_csv:
            list.append(row)
    # data = np.genfromtxt(filename, delimiter=',')
    return list

def train_model(X_train, y_train):
    model_weights = np.dot(np.dot(np.linalg.inv(np.dot(X_train.T, X_train)), X_train.T), y_train)
    return model_weights

def predict(X_test, model_weights):
    return np.dot(X_test, model_weights)

if __name__ == "__main__":
    list = read_csv_data('inputdata')
    data = np.asarray(list, dtype=float)

    X_train = data[:,[0]]
    y_train = data[:,[1]]
    X_train = np.hstack((np.ones((X_train.shape[0],1)), X_train))

    model_weights = train_model(X_train, y_train)
    y_prediction = predict(X_train, model_weights)

    plt.scatter(X_train[:,[1]], y_train,  color='black')
    plt.plot(X_train[:,[1]], y_prediction, color='blue', linewidth=3)
    plt.show()




# (X_Transpose . X)_Inverse . X_Transpose. y
