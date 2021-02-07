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
from flair.trainers import ModelTrainer
from flair.embeddings import WordEmbeddings, TransformerWordEmbeddings, TransformerDocumentEmbeddings
from flair.data import Sentence
from flair.data import MultiCorpus
from flair.datasets import CONLL_03

flair.set_seed(1)

label_name_map = {

}

print(label_name_map)
corpus = CONLL_03(label_name_map=label_name_map, base_path="/vol/fob-vol7/mi19/harnisph/studienprojekt-dokumentation")
print(corpus)
corpus = corpus.downsample(0.1)
print(corpus)


tag_type = "pos"
label_dictionary = corpus.make_label_dictionary(tag_type)
print(label_dictionary)


tagger = TARSSequenceTagger(tag_dictionary=label_dictionary, tag_type=tag_type, task_name="TEST_POS")

trainer = ModelTrainer(tagger, corpus)
trainer.train(
    base_path='resources/taggers/tars',
    learning_rate=0.1,
    mini_batch_size=16,
    mini_batch_chunk_size=4,
    max_epochs=10,
)
