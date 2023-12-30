import optuna
import json
import pandas as pd
from topic_classification import data, utils, evaluate, predict
import config.config as config
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from imblearn.over_sampling import RandomOverSampler

def train(args, df, trial=None):
    """Train model on data."""
    # Setup
    utils.set_seeds()
    if args.shuffle: df = df.sample(frac=1).reset_index(drop=True)
    df = df[: args.subset]  # None = all samples
    print("start preprocessing")
    df = data.preprocess(
        df,
        lower=args.lower,
        stem=args.stem,
        strip_stopwords=args.strip_stopwords,
        strip_links=args.strip_links,
        strip_punctuations=args.strip_punctuations,
        replace_emojis=args.replace_emojis,
    )
    print("finish preprocessing")
    label_encoder = LabelEncoder()
    label_encoder.fit(df.label_name)
    print(f"classes_ = {label_encoder.classes_}")

    X_train, X_test, y_train, y_test = \
        data.get_data_splits(X=df.text.to_numpy(), y=label_encoder.transform(df.label_name))
    
    test_df = pd.DataFrame({"text": X_test, "label_name": label_encoder.inverse_transform(y_test)})

    print("start vectorization")
    # Tf-idf (For feature extraction)
    vectorizer = TfidfVectorizer(analyzer=args.analyzer,
                                 ngram_range=(1,args.ngram_max_range), # char n-grams
                                 min_df=args.min_freq,
                                 max_df=args.max_freq)  
    
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    print("finish vectorization")
    # Oversample
    oversample = RandomOverSampler(random_state=5000)
    X_over, y_over = oversample.fit_resample(X_train, y_train)

    # Model    
    classifier = LogisticRegression(max_iter=args.num_epochs,
                                    solver=args.solver,
                                    penalty=args.penalty,
                                    C=args.C)
    
    model = OneVsRestClassifier(classifier)
    print("start training")
    # Training
    model.fit(X_over, y_over)
    print("finish training")
    # Evaluation
    y_pred = model.predict(X_test)
    performance = evaluate.get_metrics(
        y_true=y_test, y_pred=y_pred, classes=label_encoder.classes_, df=test_df
    )
    print(f"performance = {performance}")

    test_df["prediction"] = label_encoder.inverse_transform(y_pred)
    test_df.to_csv(config.PREDICTIONS_DATA_FILE, index=False)

    return {
        "args": args,
        "label_encoder": label_encoder,
        "vectorizer": vectorizer,
        "model": model,
        "performance": performance,
    }


def objective(args, df, trial: optuna.Trial):
    """Objective function for optimization trials."""
    # Parameters to tune
    args.analyzer = trial.suggest_categorical("analyzer", ["word", "char", "char_wb"])
    args.ngram_max_range = trial.suggest_int("ngram_max_range", 3, 10)
    args.min_freq = trial.suggest_int("min_freq", 1, 5)
    args.max_freq = trial.suggest_float("max_freq", 0.98, 1.0, step=0.01)
    args.stem = trial.suggest_categorical("stem", [False, True])
    args.strip_stopwords = trial.suggest_categorical("strip_stopwords", [False, True])
    args.strip_punctuations = trial.suggest_categorical("strip_punctuations", [False, True])
    args.strip_links = trial.suggest_categorical("strip_links", [False, True])
    args.replace_emojis = trial.suggest_categorical("replace_emojis", [False, True])
    args.C = trial.suggest_categorical("C", [100, 10, 1.0, 0.1, 0.01])
    args.solver = trial.suggest_categorical("solver", ["newton-cg", "lbfgs", "liblinear"])
    args.penalty = trial.suggest_categorical("penalty", ["l2"])

    # Train & evaluate
    artifacts = train(args=args, df=df, trial=trial)

    # Set additional attributes
    overall_performance = artifacts["performance"]["overall"]
    print(json.dumps(overall_performance, indent=2))
    trial.set_user_attr("precision", overall_performance["precision"])
    trial.set_user_attr("recall", overall_performance["recall"])
    trial.set_user_attr("f1", overall_performance["f1"])

    return overall_performance["f1"]
