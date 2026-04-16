from sklearn.linear_model import ElasticNet


def fit_en(X_train, y_train, alpha=0.5, l1_ratio=0.1):
    model = ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        random_state=42,
        max_iter=10000,
    )
    model.fit(X_train, y_train)
    return model


def predict_en(model, X_test):
    pred = model.predict(X_test)
    return pred
