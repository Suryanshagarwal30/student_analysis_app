# utils.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data(path="engineering_students datasets.csv"):
    df = pd.read_csv(path)
    df['High_CGPA'] = (df['CGPA'] >= 8.0).astype(int)
    df['Interest_Code'] = df['Interest'].astype('category').cat.codes
    return df

def prepare_features(df):
    tfidf = TfidfVectorizer(max_features=10)  # <-- You forgot to define tfidf here
    tfidf_matrix = tfidf.fit_transform(df['Skills'])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())

    X = pd.concat([
        df[['IA Marks', 'Attendance (%)', 'Year', 'Interest_Code']].reset_index(drop=True),
        tfidf_df.reset_index(drop=True)
    ], axis=1)

    y = df['High_CGPA']
    return X, y

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model, X_test
