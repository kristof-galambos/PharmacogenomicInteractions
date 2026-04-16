
def split_data(X, y):
    shuffled_idx = X.sample(frac=1, random_state=42).index

    X_shuffled = X.loc[shuffled_idx].reset_index(drop=True)
    y_shuffled = y.loc[shuffled_idx].reset_index(drop=True)

    n = len(X_shuffled)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)

    X_train = X_shuffled.iloc[:train_end]
    X_val = X_shuffled.iloc[train_end:val_end]
    X_test = X_shuffled.iloc[val_end:]

    y_train = y_shuffled.iloc[:train_end]
    y_val = y_shuffled.iloc[train_end:val_end]
    y_test = y_shuffled.iloc[val_end:]

    return X_train, X_val, X_test, y_train, y_val, y_test
