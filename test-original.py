import sys
import os
#sys.path.insert(0, os.path.join('C:/', 'Users', 'pharn', 'flair'))
#sys.path.insert(0, os.path.join('C:/', 'Users', 'pharn', 'AppData', 'Local', 'Packages', 'PythonSoftwareFoundation.Python.3.8_qbz5n2kfra8p0', 'LocalCache', 'local-packages', 'Python38', 'site-packages'))
sys.path.insert(0, "/vol/fob-vol7/mi19/harnisph/flair")
sys.path.insert(0, os.path.join("vol", "fob-vol7", "mi19", "harnisph", "flair"))

import flair
from flair.data import Corpus
#from flair.datasets import TREC_6
from flair.models import TARSSequenceTagger
from flair.models.text_classification_model import TARSClassifier
from flair.trainers import ModelTrainer
from flair.embeddings import WordEmbeddings, TransformerWordEmbeddings, TransformerDocumentEmbeddings
from flair.data import Sentence
from flair.data import MultiCorpus
from flair.datasets import TREC_6



# 1. define label names in natural language since some datasets come with cryptic set of labels
label_name_map = {'ENTY':'question about entity',
                  'DESC':'question about description',
                  'ABBR':'question about abbreviation',
                  'HUM':'question about person',
                  'NUM':'question about number',
                  'LOC':'question about location'
                  }
corpus = TREC_6(label_name_map=label_name_map)
print(corpus)
corpus = corpus.downsample(0.1)
print(corpus)


label_dictionary = corpus.make_label_dictionary()
print(label_dictionary)
tagger = TARSClassifier(label_dictionary=label_dictionary,label_type="label", task_name="TEST_CLASS")

trainer = ModelTrainer(tagger, corpus)
trainer.train(
    base_path='resources/taggers/tars',
    learning_rate=0.01,
    mini_batch_size=16,
    mini_batch_chunk_size=4,
    max_epochs=1,
)
