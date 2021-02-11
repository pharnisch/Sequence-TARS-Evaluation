import sys
import os
sys.path.insert(0, "/vol/fob-vol7/mi19/harnisph/flair")

import flair
import torch
from flair.models import TARSSequenceTagger2
from flair.data import Sentence, Corpus
from flair.datasets import MIT_MOVIE_NER_COMPLEX, SentenceDataset
from flair.trainers import ModelTrainer
from torch.optim.lr_scheduler import OneCycleLR

flair.set_seed(1)

tagger = TARSSequenceTagger2.load("resources/v1/conll_3-cryptic/final-model.pt")

### train k sentences for each tag in new corpus
k = 5
label_name_map = {
"Character_Name":"Character Name"
}
print(label_name_map)
corpus = MIT_MOVIE_NER_COMPLEX(tag_to_bioes=None, tag_to_bio2="ner", label_name_map=label_name_map)
corpus_small = corpus.downsample(0.1)
tag_type = "ner"
tag_dictionary = corpus.make_label_dictionary(tag_type)
corpus_sents = []
tag_countdown = [k for i in range(len(tag_dictionary.idx2item))]

for idx in range(len(corpus.train)):
    sent = corpus.train[idx]
    sent_picked = False
    for tkn in sent:
        if sent_picked:
            break
        tag_encoded = tkn.get_tag("ner").value.encode("UTF-8")
        if tag_encoded in tag_dictionary.idx2item and tag_countdown[tag_dictionary.item2idx[tag_encoded]] > 0:
            corpus_sents.append(sent)
            tag_countdown[tag_dictionary.item2idx[tag_encoded]] -= 1
            sent_picked = True

print("sents for training: " + str(len(corpus_sents)))
print("amount of items in dict: " + str(len(tag_dictionary.item2idx)))

training_dataset = SentenceDataset(corpus_sents)
training_corpus = Corpus(train=training_dataset, dev=corpus_small.dev, test=corpus_small.test, sample_missing_splits=False)
trainer = ModelTrainer(tagger, training_corpus, optimizer=torch.optim.AdamW)
tag_dictionary = training_corpus.make_label_dictionary(tag_type)
tagger.add_and_switch_to_new_task("fewshot-conll3-cryptic-to-moviecomplex", tag_dictionary=tag_dictionary, tag_type=tag_type)
trainer.train(
    base_path='resources/v1/fewshot-conll_3-cryptic-to-moviecomplex-k' + str(k),
    learning_rate=5.0e-5,
    mini_batch_size=32,
    mini_batch_chunk_size=None,
    max_epochs=10,
    weight_decay=0.,
    embeddings_storage_mode="none",
    scheduler=OneCycleLR,
)

# evaluation

sentences = [
Sentence("The Parlament of the United Kingdom is discussing a variety of topics."),
Sentence("A man fell in love with a woman. This takes place in the last century. The film received the Golden Love Film Award."),
Sentence("The Company of Coca Cola was invented in 1901."),
Sentence("This is very frustrating! I was smiling since I saw you."),
Sentence("The Green Party received only a small percentage of the vote."),
Sentence("Bayern Munich won the german soccer series the sixth time in a row.")
]

tags = [
["O", "B-Institution", "I-Institution", "B-Place", "I-Place", "B-Diverse", "I-Diverse"],
["O", "B-Story", "I-Story", "B-Price", "I-Price", "B-Time", "I-Time"],
["O", "B-Institution", "I-Institution", "B-Time", "I-Time"],
["O", "B-Happy", "I-Happy", "B-Sad", "I-Sad", "B-Neutral", "I-Neutral"],
["O", "B-Politician", "I-Politician", "B-Party", "I-Party"],
["O", "B-Sports Club", "I-Sports Club"]
]

for idx in range(len(sentences)):
	sent = sentences[idx]
	tagger.predict_zero_shot(sent, tags[idx])
	print(str(idx))
	print(sent.to_tagged_string)
	print("-------------")
