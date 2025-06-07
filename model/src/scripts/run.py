import logging
import time

from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import r2_score, root_mean_squared_error

from model.src.data_preparation.outlier_treatment import treat_outliers
from model.src.data_preparation.preprocessing import load_data
from model.src.model_training.main import get_optimized_pipeline
from model.src.postprocessing.main import get_feat_importance
from model.src.utils.helpers import get_logger, stratified_split

default_logger = logging.getLogger(__name__)


def run(cfg: DictConfig, logger: logging.Logger = default_logger) -> None:
    logger.info("Reading data...")
    df, scaling_map = load_data(data_path=cfg.paths.data_path, scaling_path=cfg.paths.scaling_path)
    logger.info("Data loaded successfully.")

    # Split into train and test
    X_train, X_test, y_train, y_test = stratified_split(
        df=df,
        target_col=cfg.model.target_col,
        test_size=cfg.model.test_size,
        random_state=cfg.model.random_state,
    )

    # Outlier Treatment on TRAIN
    demand_cols = [f"{c}_{i}" for c in cfg.data.countries for i in (1, 2, 3)]
    X_train_clean = treat_outliers(
        df=X_train,
        cols=demand_cols,
        lower_quantile=cfg.data.lower_quantile,
        upper_quantile=cfg.data.upper_quantile,
    )

    # Training
    logger.info("Training begins...")
    logger.info(f"Using estimator: {cfg.model.estimator}")
    best_pipeline, best_model = get_optimized_pipeline(
        cfg=cfg, scaling_map=scaling_map, X=X_train_clean, y=y_train
    )
    logger.info(f"Best CV score ({cfg.model.scoring}): {-best_pipeline.best_score_:.2f}")

    # Inference on TEST
    logger.info("Running inference on test set...")
    y_pred = best_model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    logger.info("Inference completed.")
    logger.info("======= Test Set Evaluation =======")
    logger.info(f"Test RMSE: {rmse:.2f}")
    logger.info(f"Test R_squared: {r2:.4f}")

    # Feature importance
    feat_imp = get_feat_importance(model=best_model, df=X_train_clean)
    logger.info("======= Top 10 Feature Importances =======")
    for name, imp in feat_imp[:10]:
        logger.info(f"{name}: {imp:.4f}")


if __name__ == "__main__":
    logger = get_logger(name="run_script", level=logging.INFO)
    cfg = OmegaConf.load("./conf/config.yaml")
    start_time = time.time()
    run(cfg, logger=logger)
    diff = time.time() - start_time
    logger.info(f"The whole training and inference took: {diff:.2f} seconds.")
