import sys
import os
#sys.path.insert(0, os.path.join('C:/', 'Users', 'pharn', 'flair'))
#sys.path.insert(0, os.path.join('C:/', 'Users', 'pharn', 'AppData', 'Local', 'Packages', 'PythonSoftwareFoundation.Python.3.8_qbz5n2kfra8p0', 'LocalCache', 'local-packages', 'Python38', 'site-packages'))
sys.path.insert(0, "/vol/fob-vol7/mi19/harnisph/flair")
sys.path.insert(0, os.path.join("vol", "fob-vol7", "mi19", "harnisph", "flair"))




import flair
from flair.models import TARSSequenceTagger2
from flair.data import Sentence
#from flair.datasets import CONLL_3, MIT_MOVIE_NER_COMPLEX
from flair.datasets import MIT_MOVIE_NER_COMPLEX


flair.set_seed(1)

tagger = TARSSequenceTagger2.load("resources/testfaelle-studproj/conll_3+mit10sent/final-model.pt")


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



label_name_map = {
"Character_Name":"Character Name"
}
corpus = MIT_MOVIE_NER_COMPLEX(tag_to_bioes=None, tag_to_bio2="ner", label_name_map=label_name_map)
corpus = corpus.downsample(0.1)
corpus_sents = []

tag_dictionary = corpus.make_label_dictionary()
tagger.add_and_switch_to_new_task("EVALUATION", tag_dictionary, "ner")

for idx in range(len(corpus.test)):
        corpus_sents.append(corpus.test[idx])

result, eval_loss = tagger.evaluate(corpus_sents)

print(result.main_score)
print(result.log_header)
print(result.log_line)
print(result.detailed_results)
print(eval_loss)
