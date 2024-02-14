import sys
import json
import mlflow
import optuna
from optuna.integration.mlflow import MLflowCallback
import tempfile
import joblib
import warnings
from pathlib import Path
from argparse import Namespace
from numpyencoder import NumpyEncoder
sys.path.append(".")
import config.config as config
from topic_classification import train, utils, predict
import typer
warnings.filterwarnings("ignore")

app = typer.Typer()


@app.command(name="train")
def train_model(args_fp: str = "config/args.json", 
                experiment_name: str = "logistic_regression_exp",
                run_name: str = "logistic_regression_run",
                test_run: bool = False) -> None:
    """Train a model given arguments."""

    # Load labeled data
    df = utils.load_data(str(Path(config.RAW_DATA_DIR, "tweet_topic.csv")))
    print(f"data_size = {len(df)}")
    # Train
    args = Namespace(**utils.load_dict(filepath=args_fp))

    mlflow.set_experiment(experiment_name=experiment_name)
    with mlflow.start_run(run_name=run_name):
        run_id = mlflow.active_run().info.run_id
        print(f"Run ID: {run_id}")
        artifacts = train.train(df=df, args=args)
        performance = artifacts["performance"]
        print(json.dumps(performance, indent=2))

        # Log metrics and parameters
        performance = artifacts["performance"]
        mlflow.log_metrics({"precision": performance["overall"]["precision"]})
        mlflow.log_metrics({"recall": performance["overall"]["recall"]})
        mlflow.log_metrics({"f1": performance["overall"]["f1"]})
        mlflow.log_params(vars(artifacts["args"]))

        # Log artifacts
        with tempfile.TemporaryDirectory() as dp:
            utils.save_dict(vars(artifacts["args"]), Path(dp, "args.json"), cls=NumpyEncoder)
            joblib.dump(artifacts["label_encoder"], Path(dp, "label_encoder.pkl"))
            joblib.dump(artifacts["vectorizer"], Path(dp, "vectorizer.pkl"))
            joblib.dump(artifacts["model"], Path(dp, "model.pkl"))
            utils.save_dict(performance, Path(dp, "performance.json"))
            mlflow.log_artifacts(dp)

    # Save to config
    if not test_run:  # pragma: no cover, actual run
        open(Path(config.CONFIG_DIR, "run_id.txt"), "w").write(run_id)
        utils.save_dict(performance, Path(config.CONFIG_DIR, "performance.json"))

@app.command(name="optimize")
def optimize(num_trials :int = 10,
             args_fp: str = "config/args.json",
             study_name: str = "logistic_regression_study"):
    """Optimize hyperparameters."""
    # Load labeled data
    df = utils.load_data(str(Path(config.RAW_DATA_DIR, "tweet_topic.csv")))

    args = Namespace(**utils.load_dict(filepath=args_fp))
    # Optimize
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    study = optuna.create_study(study_name=study_name, direction="maximize", pruner=pruner)
    mlflow_callback = MLflowCallback(
        tracking_uri=mlflow.get_tracking_uri(), metric_name="f1")
    study.optimize(
        lambda trial: train.objective(args, df, trial),
        n_trials=num_trials,
        callbacks=[mlflow_callback])

    # Best trial
    trials_df = study.trials_dataframe()
    trials_df = trials_df.sort_values(["user_attrs_f1"], ascending=False)
    utils.save_dict({**args.__dict__, **study.best_trial.params}, args_fp, cls=NumpyEncoder)
    print(f"\nBest value (f1): {study.best_trial.value}")
    print(f"Best hyperparameters: {json.dumps(study.best_trial.params, indent=2)}")

@app.command(name="predict")
def predict_topic(text, run_id=None):
    """Predict the topic of a text."""
    if not run_id:
        run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()
    artifacts = utils.load_artifacts(run_id=run_id)
    prediction = predict.predict(texts=[text], artifacts=artifacts)
    print(json.dumps(prediction, indent=2))
    return prediction

if __name__ == "__main__":
    app()