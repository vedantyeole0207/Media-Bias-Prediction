# --- custom_transformers.py ---
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

# Ensure this class is exactly the one used in your ColumnTransformer
class VaderSentimentExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
    
    def transform(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            # Assumes the input is a single column named 'content'
            X = X.iloc[:, 0] 
        
        vader_features = X.apply(lambda x: self.analyzer.polarity_scores(x) if isinstance(x, str) else {'compound': 0, 'pos': 0, 'neg': 0})
        
        vader_df = pd.DataFrame(vader_features.tolist())[['compound', 'pos', 'neg']]
        return vader_df.values

    def fit(self, X, y=None):
        return self

    def get_feature_names_out(self, input_features=None):
        return ['vader_compound', 'vader_pos', 'vader_neg']

def sparse_to_dense(X): # <-- MUST be defined as a standalone function
    """Converts a sparse matrix (e.g., from TfidfVectorizer) to a dense array."""
    if hasattr(X, 'toarray'):
        return X.toarray()
    return X
    