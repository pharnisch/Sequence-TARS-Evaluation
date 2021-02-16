import sys
import os
sys.path.insert(0, "/vol/fob-vol7/mi19/harnisph/flair")

import flair
import torch
from flair.models import TARSSequenceTagger2
from flair.data import Sentence
from flair.datasets import MIT_MOVIE_NER_COMPLEX, CONLL_03, WNUT_17, WNUT_2020_NER, BIOSCOPE

flair.set_seed(1)

tagger = TARSSequenceTagger2.load("resources/v1/sequence-2/final-model.pt")

# CONLL3

label_name_map = {
"LOC":"Location","PER":"Person","ORG":"Organization","MISC":"Miscellaneous"
}
print(label_name_map)
corpus = CONLL_03(tag_to_bioes=None, tag_to_bio2="ner", label_name_map=label_name_map, base_path="/vol/fob-vol7/mi19/harnisph/studienprojekt-dokumentation")
corpus = corpus.downsample(0.1)
tag_type = "ner"
tag_dictionary = corpus.make_label_dictionary(tag_type)
tagger.add_and_switch_to_new_task("sequence-1-conll3", tag_dictionary=tag_dictionary, tag_type=tag_type)
result, eval_loss = tagger.evaluate(corpus.test)
print(result.main_score)
print(result.log_header)
print(result.log_line)
print(result.detailed_results)
print(eval_loss)

# MOVIECOMPLEX

label_name_map = {
"Character_Name":"Character Name"
}
print(label_name_map)
corpus = MIT_MOVIE_NER_COMPLEX(tag_to_bioes=None, tag_to_bio2="ner", label_name_map=label_name_map)
corpus = corpus.downsample(0.1)
tag_type = "ner"
tag_dictionary = corpus.make_label_dictionary(tag_type)
tagger.add_and_switch_to_new_task("sequence-1-moviecomplex", tag_dictionary=tag_dictionary, tag_type=tag_type)
result, eval_loss = tagger.evaluate(corpus.test)
print(result.main_score)
print(result.log_header)
print(result.log_line)
print(result.detailed_results)
print(eval_loss)

# WNUT17

label_name_map = {
"person":"Person", "location":"Location", "creative-work":"Creative Work", "product":"Product", "corporation":"Corporation", "group":"Group"
}
print(label_name_map)
corpus = WNUT_17(tag_to_bioes=None, tag_to_bio2="ner", label_name_map=label_name_map)
corpus = corpus.downsample(0.1)
tag_type = "ner"
tag_dictionary = corpus.make_label_dictionary(tag_type)
tagger.add_and_switch_to_new_task("sequence-1-wnut17", tag_dictionary=tag_dictionary, tag_type=tag_type)
result, eval_loss = tagger.evaluate(corpus.test)
print(result.main_score)
print(result.log_header)
print(result.log_line)
print(result.detailed_results)
print(eval_loss)

# WNUT2020

label_name_map = {
}
print(label_name_map)
corpus = WNUT_2020_NER(tag_to_bioes=None, tag_to_bio2="ner", label_name_map=label_name_map)
corpus = corpus.downsample(0.1)
tag_type = "ner"
tag_dictionary = corpus.make_label_dictionary(tag_type)
tagger.add_and_switch_to_new_task("sequence-1-wnut20", tag_dictionary=tag_dictionary, tag_type=tag_type)
result, eval_loss = tagger.evaluate(corpus.test)
print(result.main_score)
print(result.log_header)
print(result.log_line)
print(result.detailed_results)
print(eval_loss)

# BIOSCOPE

label_name_map = {
"NEGATION":"Negation", "SPECULATION":"Speculation"
}
print(label_name_map)
corpus = BIOSCOPE(tag_to_bioes=None, tag_to_bio2="ner", label_name_map=label_name_map)
corpus = corpus.downsample(0.1)
tag_type = "tag"
tag_dictionary = corpus.make_label_dictionary(tag_type)
tagger.add_and_switch_to_new_task("sequence-1-bioscope", tag_dictionary=tag_dictionary, tag_type=tag_type)
result, eval_loss = tagger.evaluate(corpus.test)
print(result.main_score)
print(result.log_header)
print(result.log_line)
print(result.detailed_results)
print(eval_loss)

# evaluation

sentences = [
Sentence("The Parlament of the United Kingdom is discussing a variety of topics."),
Sentence("Bayern Munich won the german soccer series the sixth time in a row."),
Sentence("The Green Party received only a small percentage of the vote."),
Sentence("The film received the Golden Love Film Award."),
Sentence("I hated the Sci-Fi Genre since I saw this movie."),
Sentence("Today, this pair of shoes costs 100 Dollar."),
Sentence("The Company of Coca Cola was invented in 1901."),
Sentence("BAYER is based in Leverkusen."),
Sentence("The Parlament of the United Kingdom is discussing a variety of topics."),
Sentence("This is very frustrating!"),
Sentence("I am happy, yay!"),
Sentence("This does not affect me."),
Sentence("The Green Party received only a small percentage of the vote."),
Sentence("The Parlament of the United Kingdom is discussing a variety of topics."),
Sentence("The Republican Party is split in regard to Donald Trump."),
Sentence("Biden is the new president of the United States of America."),
Sentence("There are many elections in Germany this year."),
Sentence("Bayern Munich won the german soccer series the sixth time in a row.")
]

tags = [
["O", "B-Institution", "I-Institution", "B-Place", "I-Place", "B-Diverse", "I-Diverse"],
["O", "B-Institution", "I-Institution", "B-Place", "I-Place", "B-Diverse", "I-Diverse"],
["O", "B-Institution", "I-Institution", "B-Place", "I-Place", "B-Diverse", "I-Diverse"],
["O", "B-Story", "I-Story", "B-Price", "I-Price", "B-Time", "I-Time"],
["O", "B-Story", "I-Story", "B-Price", "I-Price", "B-Time", "I-Time"],
["O", "B-Story", "I-Story", "B-Price", "I-Price", "B-Time", "I-Time"],
["O", "B-Institution", "I-Institution", "B-Time", "I-Time"],
["O", "B-Institution", "I-Institution", "B-Time", "I-Time"],
["O", "B-Institution", "I-Institution", "B-Time", "I-Time"],
["O", "B-Happy", "I-Happy", "B-Sad", "I-Sad", "B-Neutral", "I-Neutral"],
["O", "B-Happy", "I-Happy", "B-Sad", "I-Sad", "B-Neutral", "I-Neutral"],
["O", "B-Happy", "I-Happy", "B-Sad", "I-Sad", "B-Neutral", "I-Neutral"],
["O", "B-Politician", "I-Politician", "B-Party", "I-Party"],
["O", "B-Politician", "I-Politician", "B-Party", "I-Party"],
["O", "B-Politician", "I-Politician", "B-Party", "I-Party"],
["O", "B-Politician", "I-Politician", "B-Party", "I-Party"],
["O", "B-Politician", "I-Politician", "B-Party", "I-Party"],
["O", "B-Sports Club", "I-Sports Club"]
]

for idx in range(len(sentences)):
	sent = sentences[idx]
	tagger.predict_zero_shot(sent, tags[idx])
	print(str(idx))
	print(sent.to_tagged_string)
	print("-------------")
