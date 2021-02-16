import sys
import os
sys.path.insert(0, "/vol/fob-vol7/mi19/harnisph/flair")

import flair
import torch
from flair.models import TARSSequenceTagger2
from flair.data import Sentence, Corpus, Dictionary
from flair.datasets import CONLL_03, SentenceDataset
from flair.trainers import ModelTrainer
from torch.optim.lr_scheduler import OneCycleLR

# helper

def _get_tag_dictionary_no_prefix(tag_dictionary):
    candidate_tag_list = []
    for tag in tag_dictionary.idx2item:
        tag = tag.decode("utf-8")
        prefix, tag_no_prefix = _split_tag(tag)
        if prefix == "B" or prefix == "I":
            candidate_tag_list.append(tag_no_prefix)
    candidate_tag_list = _remove_not_unique_items_from_list(candidate_tag_list)

    tag_dictionary_no_prefix: Dictionary = Dictionary(add_unk=False)
    for tag in candidate_tag_list:
        tag_dictionary_no_prefix.add_item(tag)

    return tag_dictionary_no_prefix

def _split_tag(tag: str):
    if tag == "O":
        return tag, None
    elif "-" in tag:
        tag_split = tag.split("-")
        return tag_split[0], "-".join(tag_split[1:])
    else:
        return None, None

def _remove_not_unique_items_from_list(l: list):
    new_list = []
    for item in l:
        if item not in new_list:
            new_list.append(item)
    return new_list

####

flair.set_seed(1)

tagger = TARSSequenceTagger2.load("resources/v1/moviecomplex-cryptic/final-model.pt")

### train k sentences for each tag in new corpus
k = 1
label_name_map = {
"LOC":"Location","PER":"Person","ORG":"Organization","MISC":"Miscellaneous"
}
print(label_name_map)
corpus = MIT_MOVIE_NER_COMPLEX(tag_to_bioes=None, tag_to_bio2="ner", label_name_map=label_name_map, base_path="/vol/fob-vol7/mi19/harnisph/studienprojekt-dokumentation")
corpus_small = corpus.downsample(0.1)
tag_type = "ner"
tag_dictionary = corpus.make_label_dictionary(tag_type)
corpus_sents = []
tag_dictionary_no_prefix = _get_tag_dictionary_no_prefix(tag_dictionary)
tag_countdown = [k for i in range(len(tag_dictionary_no_prefix.idx2item))]

for idx in range(len(corpus.train)):
    sent = corpus.train[idx]
    sent_picked = False
    for tkn in sent:
        if sent_picked:
            break
        tag = tkn.get_tag("ner").value
        pref, tag_no_pref = _split_tag(tag)
        if tag_no_pref is None:
            break
        tag_no_pref_encoded = tag_no_pref.encode("utf-8")
        if tag_no_pref_encoded in tag_dictionary_no_prefix.idx2item and tag_countdown[tag_dictionary_no_prefix.item2idx[tag_no_pref_encoded]] > 0:
            corpus_sents.append(sent)
            tag_countdown[tag_dictionary_no_prefix.item2idx[tag_no_pref_encoded]] -= 1
            sent_picked = True

print("sents for training: " + str(len(corpus_sents)))
print("amount of items in dict: " + str(len(tag_dictionary.item2idx)))

training_dataset = SentenceDataset(corpus_sents)
training_corpus = Corpus(train=training_dataset, dev=corpus_small.dev, test=corpus_small.test, sample_missing_splits=False)
trainer = ModelTrainer(tagger, training_corpus, optimizer=torch.optim.AdamW)
tag_dictionary = training_corpus.make_label_dictionary(tag_type)
tagger.add_and_switch_to_new_task("fewshot-moviecomplex-cryptic-to-conll3", tag_dictionary=tag_dictionary, tag_type=tag_type)
trainer.train(
    base_path='resources/v1/fewshot-moviecomplex-cryptic-to-conll3-k' + str(k),
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
