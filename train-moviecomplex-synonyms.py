import sys
import os
sys.path.insert(0, "/vol/fob-vol7/mi19/harnisph/flair")

import flair
import torch
from flair.data import Corpus
from flair.models import SequenceTagger, TARSSequenceTagger, TARSSequenceTagger2
from flair.trainers import ModelTrainer
from flair.embeddings import WordEmbeddings, TransformerWordEmbeddings, TransformerDocumentEmbeddings
from flair.data import Sentence
from flair.data import MultiCorpus
from flair.datasets import MIT_MOVIE_NER_COMPLEX

flair.set_seed(3)

label_name_map = {
"Actor":"Actor Artist Character",
"Award":"Award Trophy",
"Character_Name":"Character Name",
"Director":"Director Leader Organizer",  
"Genre":"Genre Kind Style Category Class",    
"Opinion":"Opinion Impression",  
"Origin":"Origin Provenance",     
"Plot":"Plot Scene Narrative",      
"Quote":"Quote Cite",      
"Relationship":"Relationship Link",
"Soundtrack":"Soundtrack Music",
"Year":"Year Time"
}

print(label_name_map)
corpus = MIT_MOVIE_NER_COMPLEX(tag_to_bioes=None, tag_to_bio2="ner",label_name_map=label_name_map)
corpus = corpus.downsample(0.1)

tag_type = "ner"
label_dictionary = corpus.make_label_dictionary(tag_type)
print(label_dictionary)

tagger = TARSSequenceTagger2(tag_dictionary=label_dictionary, tag_type=tag_type, task_name="TEST_NER")

trainer = ModelTrainer(tagger, corpus, optimizer=torch.optim.AdamW)
from torch.optim.lr_scheduler import OneCycleLR
trainer.train(
    base_path='resources/v3/moviecomplex-synonyms',
    learning_rate=5.0e-5,
    mini_batch_size=32,
    mini_batch_chunk_size=None,
    max_epochs=10,
    weight_decay=0.,
    embeddings_storage_mode="none",
    scheduler=OneCycleLR,
)
