
### Experiment 1
Ian Hay - 2023-03-21

https://github.com/hayitsian/General-Index-Visualization

-------------------------------------------------------------------------------

Hypothesis: testing different clustering algorithms across the arXiv abstract data.


Data: 25,000 arXiv abstracts across the top 10 topics
Preprocessing: none (raw) or spaCy lemmatization and stopwords & punctuation removal
Models: SKLearn's KMeans, LDA, and NMF clustering
Features: BoW, TF-iDF, and BERT embeddings
Metrics: calinski, completeness, davies, homogeneity, rand, silhouette, and v-measure

Total runtime: 