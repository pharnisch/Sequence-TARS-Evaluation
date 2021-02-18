import sys
import os
sys.path.insert(0, "/vol/fob-vol7/mi19/harnisph/flair")

import flair
import torch
from flair.models import TARSSequenceTagger2
from flair.data import Sentence
from flair.datasets import MIT_MOVIE_NER_COMPLEX, CONLL_03, WNUT_17, WNUT_2020_NER, BIOSCOPE
from flair.trainers import ModelTrainer
from torch.optim.lr_scheduler import OneCycleLR

flair.set_seed(2)

tagger = TARSSequenceTagger2.load("resources/v2/sequence-4/final-model.pt")

label_name_map = {
"NEGATION":"Negation", "SPECULATION":"Speculation"
}
print(label_name_map)
corpus = BIOSCOPE(tag_to_bioes=None, tag_to_bio2="ner", label_name_map=label_name_map)
corpus = corpus.downsample(0.1)
tag_type = "tag"
tag_dictionary = corpus.make_label_dictionary(tag_type)

tagger.add_and_switch_to_new_task("sequence-5-train", tag_dictionary=tag_dictionary, tag_type=tag_type)

trainer = ModelTrainer(tagger, corpus, optimizer=torch.optim.AdamW)
trainer.train(
    base_path='resources/v2/sequence-5',
    learning_rate=5.0e-5,
    mini_batch_size=32,
    mini_batch_chunk_size=None,
    max_epochs=10,
    weight_decay=0.,
    embeddings_storage_mode="none",
    scheduler=OneCycleLR,
)
