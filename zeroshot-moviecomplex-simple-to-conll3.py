import sys
import os
sys.path.insert(0, "/vol/fob-vol7/mi19/harnisph/flair")

import flair
import torch
from flair.models import TARSSequenceTagger2
from flair.data import Sentence
from flair.datasets import CONLL_03

tagger = TARSSequenceTagger2.load("resources/v2/moviecomplex-simple/final-model.pt")

flair.set_seed(2)

label_name_map = {
"LOC":"Location","PER":"Person","ORG":"Organization","MISC":"Miscellaneous"
}
print(label_name_map)
corpus = CONLL_03(tag_to_bioes=None, tag_to_bio2="ner", label_name_map=label_name_map, base_path="/vol/fob-vol7/mi19/harnisph/studienprojekt-dokumentation")
corpus = corpus.downsample(0.1)
tag_type = "ner"
tag_dictionary = corpus.make_label_dictionary(tag_type)
tagger.add_and_switch_to_new_task("zeroshot-moviecomplex-simple-to-conll3", tag_dictionary=tag_dictionary, tag_type=tag_type)
result, eval_loss = tagger.evaluate(corpus.test)
print(result.main_score)
print(result.log_header)
print(result.log_line)
print(result.detailed_results)
print(eval_loss)

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
