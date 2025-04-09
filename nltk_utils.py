import nltk
import numpy as np
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

stemmer = PorterStemmer()


def sanitize_input(text):
    """Clean and validate input"""
    if not text or len(text.strip()) < 2:
        return None
    return text.lower().strip()


def tokenize(sentence):
    """Tokenize with commerce term protection"""
    protected_terms = {
        "Dynamics 365 Commerce": "Dynamics365Commerce",
        "Commerce Scale Unit": "CommerceScaleUnit",
    }
    for term, placeholder in protected_terms.items():
        sentence = sentence.replace(term, placeholder)
    tokens = word_tokenize(sentence)
    for term, placeholder in protected_terms.items():
        tokens = [t.replace(placeholder, term) for t in tokens]
    return tokens


def stem(word):
    """Stemming with technical term exceptions"""
    exceptions = {
        "pos": "pos",
        "csu": "csu",
        "sdk": "sdk",
        "api": "api",
        "azure": "azure",
        "commerce": "commerce",
    }
    return exceptions.get(word.lower(), stemmer.stem(word.lower()))


def bag_of_words(tokens, all_words):
    """Create BoW vector"""
    bag = np.zeros(len(all_words), dtype=np.float32)
    stemmed_tokens = [stem(t) for t in tokens]
    for i, word in enumerate(all_words):
        if word in stemmed_tokens:
            bag[i] = 1
    return bag
