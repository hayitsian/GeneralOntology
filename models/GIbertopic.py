
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic import BERTopic



def train(_texts, _embeddingModel = None):
    # dimensional reduction
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine')


    # clustering
    hdbscan_model = HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom', prediction_data=True)


    # tokenize
    vectorizer_model = CountVectorizer(stop_words="english")


    # topic weighting
    ctfidf_model = ClassTfidfTransformer()

    # BERTopic pipeline
    if (_embeddingModel):
        topic_model = BERTopic(
        embedding_model=_embeddingModel,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        diversity=0.5,
        low_memory=False,
        verbose=True,
        calculate_probabilities=True
        )
    else:
        topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        diversity=0.5,
        low_memory=False,
        verbose=True,
        calculate_probabilities=True
        )
    
    # fit the texts
    topics, probabilities = topic_model.fit_transform(_texts)

    # return topic model
    return topic_model