#!/usr/bin/env python
#
# Ian Hay - 2023-02-04

import sys
from bertopic import BERTopic



# hard coded things
_modelFilename = "BERTopic_doc_ngrams_0_10M_ngrams_model"


# load in the pretrained model
topic_model= BERTopic.load(_modelFilename)


# print the topic info
print(topic_model.get_topic_info())


# print representative documents
# print(topic_model.get_representative_docs(2))


# print topics
print(topic_model.get_topics())


# visualize (bar charts)
fig = topic_model.visualize_barchart()
fig.write_html(_modelFilename + "_bar_chart.html")


# visualize probability distribution
fig = topic_model.visualize_heatmap()
fig.write_html(_modelFilename + "_heatmap.html")


# visualize hierarchy
fig = topic_model.visualize_hierarchy()
fig.write_html(_modelFilename + "_hierarchy.html")


# visualize term rank
fig = topic_model.visualize_term_rank()
fig.write_html(_modelFilename + "term_rank.html")


# visualize topics
fig = topic_model.visualize_topics()
fig.write_html(_modelFilename + ".html")

