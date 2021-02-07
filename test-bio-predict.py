import sys
import os
#sys.path.insert(0, os.path.join('C:/', 'Users', 'pharn', 'flair'))
#sys.path.insert(0, os.path.join('C:/', 'Users', 'pharn', 'AppData', 'Local', 'Packages', 'PythonSoftwareFoundation.Python.3.8_qbz5n2kfra8p0', 'LocalCache', 'local-packages', 'Python38', 'site-packages'))
sys.path.insert(0, "/vol/fob-vol7/mi19/harnisph/flair")
sys.path.insert(0, os.path.join("vol", "fob-vol7", "mi19", "harnisph", "flair"))


import flair
from flair.data import Corpus
#from flair.datasets import TREC_6
from flair.models import SequenceTagger, TARSSequenceTagger, TARSSequenceTagger2
from flair.trainers import ModelTrainer
from flair.embeddings import WordEmbeddings, TransformerWordEmbeddings, TransformerDocumentEmbeddings
from flair.data import Sentence
from flair.data import MultiCorpus
from flair.datasets import CONLL_03

flair.set_seed(1)

label_name_map = {
"LOC":"Location","PER":"Person","ORG":"Organization","MISC":"Miscellaneous"
}

tagger = TARSSequenceTagger2.load("resources/taggers/tars/test-bio/final-model.pt")

sentence1 = Sentence("I like Berlin and Germany")
sentence2 = Sentence("I like Berlin and Germany")

print()
print()
print("NEW SENTENCES FOR PRELEARNED CLASSES:\n")

tagger.predict(sentence2)
print(sentence2.to_tagged_string)

sent3 = Sentence("Donald Trump is the former president of the United States of America.")
tagger.predict(sent3)
print(sent3.to_tagged_string)
print(sent3.get_spans("ner"))

sent4 = Sentence("Nadim Latki")
tagger.predict(sent4)
print(sent4.to_tagged_string)

sent5 = Sentence("AUGUST 1996 CDU / CSU SPD FDP Greens PDS")
tagger.predict(sent5)
print(sent5.to_tagged_string)
print()
print()


print("ZERO SHOT:\n")

tagger.predict_zero_shot(sentence1, ["O", "I-Place", "I-Other", "I-Verb", "B-Place", "B-Other", "B-Verb"])
print(sentence1.to_tagged_string)

