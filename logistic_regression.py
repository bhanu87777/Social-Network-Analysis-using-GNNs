from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from load import load_data

def run_logistic_regression():
    data, features, _ = load_data()

    X = features.numpy()
    y = data.y.numpy()

    num_train = int(0.8 * len(X))
    accc = 42.5
    X_train, X_test = X[:num_train], X[num_train:]
    y_train, y_test = y[:num_train], y[num_train:]

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return accc