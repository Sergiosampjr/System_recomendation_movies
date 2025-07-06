import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


df_yelp = pd.read_csv('yelp_labelled.txt', sep='\t', header=None, names=['text', 'label'])
df_amazon = pd.read_csv('amazon_cells_labelled.txt', sep='\t', header=None, names=['text', 'label'])
df_imdb = pd.read_csv('imdb_labelled.txt', sep='\t', header=None, names=['text', 'label'])


df = pd.concat([df_yelp, df_amazon, df_imdb])


X = df['text']
y = df['label']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', MultinomialNB())
])


pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia: {accuracy:.2f}")
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))
