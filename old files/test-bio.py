import sys
import os
sys.path.insert(0, os.path.join('C:/', 'Users', 'pharn', 'flair'))
sys.path.insert(0, os.path.join('C:/', 'Users', 'pharn', 'AppData', 'Local', 'Packages', 'PythonSoftwareFoundation.Python.3.8_qbz5n2kfra8p0', 'LocalCache', 'local-packages', 'Python38', 'site-packages'))
#sys.path.insert(0, "/vol/fob-vol7/mi19/harnisph/flair")
#sys.path.insert(0, os.path.join("vol", "fob-vol7", "mi19", "harnisph", "flair"))


import flair
from flair.data import Corpus
#from flair.datasets import TREC_6
from flair.models import SequenceTagger, TARSSequenceTagger, TARSSequenceTagger2
from flair.trainers import ModelTrainer
from flair.embeddings import WordEmbeddings, TransformerWordEmbeddings, TransformerDocumentEmbeddings
from flair.data import Sentence, Dictionary
from flair.data import MultiCorpus
from flair.datasets import CONLL_03

flair.set_seed(1)

#label_name_map = {
#"LOC":"Location","PER":"Person","ORG":"Organization","MISC":"Miscellaneous"
#}

#print(label_name_map)
#corpus = CONLL_03(tag_to_bioes=None, tag_to_bio2="ner", label_name_map=label_name_map, base_path="/vol/fob-vol7/mi19/harnisph/studienprojekt-dokumentation")
#corpus = corpus.downsample(0.1)

tag_type = "ner"
#label_dictionary = corpus.make_label_dictionary(tag_type)
#print(label_dictionary)
label_dictionary = Dictionary(add_unk=False)

#embeddings = WordEmbeddings("glove")
#embeddings = TransformerWordEmbeddings()

tagger = TARSSequenceTagger2(tag_dictionary=label_dictionary, tag_type=tag_type, task_name="TEST_NER")
#tagger = TARSSequenceTagger(tag_dictionary=label_dictionary, tag_type=tag_type, task_name="TEST_NER")
#tagger = SequenceTagger(tag_dictionary=corpus.make_tag_dictionary(tag_type), tag_type=tag_type, hidden_size=256, embeddings=embeddings)

""" trainer = ModelTrainer(tagger, corpus)
trainer.train(
    base_path='resources/taggers/tars/test-bio',
    learning_rate=0.1,
    mini_batch_size=32,
    mini_batch_chunk_size=None,
    max_epochs=10,
)
 """

sent = Sentence("This is a funny and sad test sentence!")
tagger.predict_zero_shot(sent, ["O", "I-Happy", "I-Sad"], use_O=False)
print(sent.to_tagged_string)