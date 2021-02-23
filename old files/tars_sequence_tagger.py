import sys
import os
sys.path.insert(0, os.path.join('C:/', 'Users', 'pharn', 'flair'))
sys.path.insert(0, os.path.join('C:/', 'Users', 'pharn', 'AppData', 'Local', 'Packages', 'PythonSoftwareFoundation.Python.3.8_qbz5n2kfra8p0', 'LocalCache', 'local-packages', 'Python38', 'site-packages'))

import flair
from flair.data import Corpus
#from flair.datasets import TREC_6
from flair.models import TARSSequenceTagger
from flair.trainers import ModelTrainer
from flair.embeddings import WordEmbeddings, TransformerWordEmbeddings, TransformerDocumentEmbeddings
from flair.data import Sentence
from flair.data import MultiCorpus
from flair.datasets import UD_ENGLISH, UD_GERMAN

corpus_ger = flair.datasets.UD_GERMAN()
label_dict_ger = corpus_ger.make_label_dictionary("upos")
print(label_dict_ger)
corpus = flair.datasets.UD_ENGLISH()
label_dict = corpus.make_label_dictionary('upos')

#tagger = TARSSequenceTagger(embeddings=WordEmbeddings("glove"), tag_dictionary=label_dict, tag_type="upos", task_name="UD_ENGLISH")
tagger = TARSSequenceTagger2(tag_dictionary=label_dictionary, tag_type=tag_type, task_name="TEST_NER")
tagger.add_and_switch_to_new_task("UD_GERMAN", label_dict_ger, "upos")
tagger.list_existing_tasks()

print("_get_tars_formatted_sentence")
sent = Sentence("loving hating asdfj tests")
sent[0].add_label("upos", "B-VERB")
sent[1].add_label("upos", "I-VERB")
sent[2].add_label("upos", "O")
sent[3].add_label("upos", "B-NOUN")
print(tagger._get_tars_formatted_sentence("VERB", sent))

print("_get_tars_formatted_sentences")
sent2 = Sentence("Test nice afsdj")
sent2[0].add_label("upos", "B-NOUN")
sent2[1].add_label("upos", "B-ADJ")
sent2[2].add_label("upos", "O")
sentences = [sent, sent2]
tagger.train()
print(tagger._get_tars_formatted_sentences(sentences))

#print("_compute_tag_similarity_for_current_epoch (sets tag_nearest_map)")
#print(tagger.tag_nearest_map)

tagger.eval()
tagger.forward([sent])
