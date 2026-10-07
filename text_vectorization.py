from sklearn.feature_extraction.text import CountVectorizer

documents = [
    "Python developer with FastAPI experience",
    "Java developer with Spring experience"
]

vectorizer = CountVectorizer()

matrix = vectorizer.fit_transform(documents)

print(vectorizer.get_feature_names_out())
print(matrix.toarray())
