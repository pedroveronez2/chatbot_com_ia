from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Supondo que você tenha um conjunto de perguntas
perguntas = [
    "Qual é o nome do Pedro?",
    "Qual a idade do Pedro?",
    "Onde Pedro mora?",
    "Qual o curso do Pedro?"
]

# Converter perguntas para vetores TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(perguntas)

# Aplicar K-means para agrupar perguntas semelhantes
n_clusters = 2  # Número de clusters (ajustável)
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
kmeans.fit(X)

# Ver os clusters
print(kmeans.labels_)
