import logging
from typing import Optional, Tuple

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def stratified_split(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
    n_bins: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Perform a stratified train/test split by binning a continuous target.

    Arguments:
        df (pd.DataFrame): Full dataset including features and the target column
        target_col (str): Name of the continuous target column to stratify on
        test_size (float): Fraction of data to reserve for the test set
        random_state (int): Random seed for reproducibility
        n_bins (int): Number of equally sized frequency bins

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        - X_train (pd.DataFrame): Training features
        - X_test (pd.DataFrame):  Test features
        - y_train (pd.Series):   Training targets
        - y_test (pd.Series):    Test targets
    """
    # Make a copy to avoid modifying the original
    data = df.copy()

    # Create quantile bins for the target
    data["_y_bin"] = pd.qcut(data[target_col], q=n_bins, labels=False, duplicates="drop")

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)

    # Perform the split on the bins
    train_idx, test_idx = next(splitter.split(data, data["_y_bin"]))

    # Extract train/test sets
    train = data.iloc[train_idx].drop(columns=["_y_bin"])
    test = data.iloc[test_idx].drop(columns=["_y_bin"])

    X_train = train.drop(columns=[target_col])
    y_train = train[target_col]
    X_test = test.drop(columns=[target_col])
    y_test = test[target_col]

    return X_train, X_test, y_train, y_test


def get_logger(
    name: str = __name__, level: int = None, logfile: Optional[str] = None
) -> logging.Logger:
    """
    Create and configure a logger.

    Arguments:
        name (str): Logger name (typically __name__ of the calling module).
        level (int, optional): Logging level (e.g. logging.INFO). Defaults to INFO.
        logfile (str, optional): Path to a file for logging output. If None, logs only to console.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level or logging.INFO)

    # Prevent adding multiple handlers if this is called repeatedly
    if not logger.handlers:
        fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        formatter = logging.Formatter(fmt)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(level or logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # Optional file handler
        if logfile:
            fh = logging.FileHandler(logfile)
            fh.setLevel(level or logging.INFO)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    return logger
