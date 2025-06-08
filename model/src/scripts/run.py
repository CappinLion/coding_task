import logging
import time

import mlflow
import mlflow.sklearn
from omegaconf import DictConfig, OmegaConf

from model.src.data_preparation.outlier_treatment import treat_outliers
from model.src.data_preparation.preprocessing import load_data
from model.src.model_training.main import get_optimized_pipeline
from model.src.postprocessing.main import evaluate_metrics, get_feat_importance
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
    cv_rmse = -best_pipeline.best_score_
    logger.info(f"Best CV score ({cfg.model.scoring}): {cv_rmse:.2f}")

    # Inference on TEST
    logger.info("Running inference on test set...")
    y_pred = best_model.predict(X_test)
    metrics = evaluate_metrics(y_test, y_pred)
    logger.info("Inference completed.")
    logger.info("======= Test Set Evaluation =======")
    for metric, value in metrics.items():
        logger.info(f"TEST {metric}: {value:.4f}")

    # Feature importance
    feat_imp = get_feat_importance(model=best_model, df=X_train_clean)
    logger.info("======= Top 10 Feature Importances =======")
    for name, imp in feat_imp[:10]:
        logger.info(f"{name}: {imp:.4f}")

    # Mlflow logging
    if cfg.model.mlflow.enable:
        mlflow.set_experiment(cfg.model.mlflow.experiment_name)

        with mlflow.start_run(
            run_name=f"{cfg.model.estimator}_run_{time.strftime('%Y%m%d_%H%M%S')}"
        ):
            mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))
            mlflow.log_params(best_pipeline.best_params_)
            mlflow.log_metric("cv_rmse", cv_rmse)
            for metric, val in metrics.items():
                mlflow.log_metric(f"test_{metric}", val)
            for name, imp in feat_imp:
                mlflow.log_metric(f"feat_importance_{name}", imp)

            mlflow.sklearn.log_model(best_model, "model")
            logger.info("Model and metrics logged to MLflow.")


if __name__ == "__main__":
    cfg = OmegaConf.load("./conf/config.yaml")
    logger = get_logger(name="run_script", level=logging.INFO, logfile=cfg.paths.logging_path)
    start_time = time.time()
    run(cfg, logger=logger)
    diff = time.time() - start_time
    logger.info(f"The whole training and inference took: {diff:.2f} seconds.")
