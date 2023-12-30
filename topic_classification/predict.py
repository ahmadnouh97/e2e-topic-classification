from typing import List, Dict
import numpy as np
from topic_classification import data

def custom_predict(y_prob):
    """Custom predict function that defaults
    to an index if conditions are not met."""
    y_pred = [np.argmax(p) for p in y_prob]
    return np.array(y_pred)

def predict(texts: List, artifacts: Dict) -> List:
    """Predict topics for given texts.

    Args:
        texts (List): raw input texts to classify.
        artifacts (Dict): artifacts from a run.

    Returns:
        List: predictions for input texts.
    """
    args = artifacts["args"]
    print(args)
    texts = [
        data.clean_text(text,
                        lower=args.lower,
                        strip_stopwords=args.strip_stopwords,
                        strip_links=args.strip_links,
                        strip_punctuations=args.strip_punctuations,
                        replace_emojis=args.replace_emojis,
                        stem=args.stem) 
        for text in texts
    ]
    x = artifacts["vectorizer"].transform(texts)
    y_pred = custom_predict(
        y_prob=artifacts["model"].predict_proba(x),
    )
    topics = artifacts["label_encoder"].inverse_transform(y_pred)
    predictions = [
        {
            "input_text": texts[i],
            "predicted_topic": topics[i],
        }
        for i in range(len(topics))
    ]
    return predictions