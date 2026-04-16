def split_data(X, y, random_state=42):
    bootstrap_idx = X.sample(n=len(X), replace=True, random_state=random_state).index

    X_bootstrap = X.loc[bootstrap_idx].reset_index(drop=True)
    y_bootstrap = y.loc[bootstrap_idx].reset_index(drop=True)

    n = len(X_bootstrap)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)

    X_train = X_bootstrap.iloc[:train_end]
    X_val = X_bootstrap.iloc[train_end:val_end]
    X_test = X_bootstrap.iloc[val_end:]

    y_train = y_bootstrap.iloc[:train_end]
    y_val = y_bootstrap.iloc[train_end:val_end]
    y_test = y_bootstrap.iloc[val_end:]

    return X_train, X_val, X_test, y_train, y_val, y_test
