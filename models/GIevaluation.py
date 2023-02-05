#!/usr/bin/env python
#
# Ian Hay - 2023-02-04

import sys
from bertopic import BERTopic



# hard coded things
_modelFilename = "BERTopic_doc_ngrams_0_1M_ngrams_model"


# load in the pretrained model
topic_model= BERTopic.load(_modelFilename)


# print the topic info
print(topic_model.get_topic_info())


# visualize topics
fig = topic_model.visualize_topics()
fig.write_html(_modelFilename + ".html")

