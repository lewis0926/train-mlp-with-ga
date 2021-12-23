import numpy as np
from sklearn.neural_network import MLPClassifier


def model(x_train, y_train, hidden_layer_sizes=(100,)):
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes)
    print(mlp.get_params())

    # initialize weights and bias by running one iteration
    mlp.partial_fit(x_train, y_train, classes=np.unique(y_train))

    weights = np.array(mlp.coefs_, dtype=object)
    bias = np.array(mlp.intercepts_, dtype=object)

    for i in range(0, len(weights)):
        print(f"shape of weights in layer {i + 1}: {weights[i].shape}")
    for i in range(0, len(bias)):
        print(f"shape of bias in layer {i + 1}: {bias[i].shape}")

    return mlp, weights, bias
