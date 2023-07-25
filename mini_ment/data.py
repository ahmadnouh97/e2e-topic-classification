
import re
import pickle
import emot
from config import config
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split

stemmer = PorterStemmer()
EMOTICONS = emot.emo_unicode.EMOTICONS_EMO

with open(config.EMOJI_DICT_FILE, 'rb') as fp:
    emoji_dict = pickle.load(fp)
    emoji_dict = {v: k for k, v in emoji_dict.items()}

def get_data_splits(X, y, train_size=0.7):
    """Generate balanced data splits."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, stratify=y)

    return X_train, X_test, y_train, y_test

def preprocess(df,
               lower=True,
               strip_stopwords=True,
               strip_links=True,
               strip_punctuations=True,
               replace_emojis=False,
               stem=True):
    """Preprocess the data.

    Args:
        df (pd.DataFrame): Pandas DataFrame with data.
        lower (bool): whether to lowercase the text.
        strip_stopwords (bool): whether to strip stopwords.
        strip_links (bool): whether to strip links.
        strip_punctuations (bool): whether to strip punctuations.
        replace_emojis (bool): whether to replace emojis.
        stem (bool): whether to stem the text.

    Returns:
        pd.DataFrame: Dataframe with preprocessed data.
    """
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
    """ Clean text"""
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
    """Remove links"""
    return re.sub(r'http\S+', "", text)

def convert_emojis(text):
    """Convert emojis to words"""
    for emot in emoji_dict:
        text = re.sub(r"("+emot+")", "_".join(emoji_dict[emot].replace(",","").replace(":","").split()), text)
    return text

def convert_emoticons(text):
    """Convert emoticons to words"""
    for k, v in EMOTICONS.items():
        text = re.sub(u"("+re.escape(k)+")", "_".join(v.replace(",","").split()), text)
    return text

def remove_punctuations(text):
    """Remove punctuations"""
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
    """Remove extra spaces"""
    text = text.strip()
    text = " ".join(text.split())
    return text

def remove_stopwords(text):
    """Remove stopwords"""
    words = [word for word in text.split() if word not in config.STOPWORDS]
    return " ".join(words)