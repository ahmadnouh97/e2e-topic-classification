
import re
import pickle
import emot
from config import config
from nltk.stem import PorterStemmer
# from skmultilearn.model_selection import IterativeStratification
from sklearn.model_selection import train_test_split

stemmer = PorterStemmer()
EMOTICONS = emot.emo_unicode.EMOTICONS_EMO

with open(config.EMOJI_DICT_FILE, 'rb') as fp:
    emoji_dict = pickle.load(fp)
    emoji_dict = {v: k for k, v in emoji_dict.items()}

def get_data_splits(X, y, train_size=0.7):
    """Generate balanced data splits."""
    X_train, X_, y_train, y_ = train_test_split(
        X, y, train_size=train_size, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_, y_, train_size=0.5, stratify=y_)
    return X_train, X_val, X_test, y_train, y_val, y_test

# def iterative_train_test_split(X, y, train_size):
#     """Custom iterative train test split which
#     'maintains balanced representation with respect
#     to order-th label combinations.'
#     """
#     stratifier = IterativeStratification(
#         n_splits=2, order=1, sample_distribution_per_fold=[1.0-train_size, train_size, ])
#     train_indices, test_indices = next(stratifier.split(X, y))
#     X_train, y_train = X[train_indices], y[train_indices]
#     X_test, y_test = X[test_indices], y[test_indices]
#     return X_train, X_test, y_train, y_test


def preprocess(df,
               lower=True,
               strip_stopwords=True,
               strip_links=True,
               strip_punctuations=True,
               replace_emojis=False,
               stem=True):
    df["text"] = df["text"].apply(
        clean_text,
        lower=lower,
        strip_stopwords=strip_stopwords,
        strip_links=strip_links,
        strip_punctuations=strip_punctuations,
        replace_emojis=replace_emojis,
        stem=stem
    )

    df.to_csv(config.PROCESSED_DATA_FILE, index=False)
    return df

def clean_text(text: str, **kwargs):
    if kwargs["lower"]:
        text = text.lower()
    if kwargs["strip_links"]:
        text = remove_links(text)
    if kwargs["replace_emojis"]:
        text = convert_emojis(text)
        text = convert_emoticons(text)
    if kwargs["strip_stopwords"]:
        text = remove_stopwords(text)
    if kwargs["strip_punctuations"]:
        text = remove_punctuations(text)
    if kwargs["stem"]:
        text = " ".join([stemmer.stem(token) for token in text.split()])
    return remove_extra_spaces(text)

def remove_links(text: str):
    return re.sub(r'http\S+', "", text)

def convert_emojis(text):
    for emot in emoji_dict:
        text = re.sub(r"("+emot+")", "_".join(emoji_dict[emot].replace(",","").replace(":","").split()), text)
    return text

def convert_emoticons(text):
    for k, v in EMOTICONS.items():
        text = re.sub(u"("+re.escape(k)+")", "_".join(v.replace(",","").split()), text)
    return text

def remove_punctuations(text):
    tokens = text.split()

    # Define a regular expression pattern to match all punctuation except "@", "#", "!", "{", "}", ".", ","
    punctuation_pattern = re.compile(r'[^\w\s@#\!\}\{\.\,]')

    # Replace all matches of the punctuation pattern with an empty string
    tokens = [punctuation_pattern.sub("", token) for token in tokens]
    
    # Join tokens
    text = " ".join(tokens)
    # Return the updated text
    return text

def remove_extra_spaces(text):
    text = text.strip()
    text = " ".join(text.split())
    return text

def remove_stopwords(text):
    pattern = re.compile(r'\b(' + r"|".join(config.STOPWORDS) + r")\b\s*")
    text = pattern.sub('', text)
    return text