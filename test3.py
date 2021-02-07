import sys
import os
#sys.path.insert(0, os.path.join('C:/', 'Users', 'pharn', 'flair'))
#sys.path.insert(0, os.path.join('C:/', 'Users', 'pharn', 'AppData', 'Local', 'Packages', 'PythonSoftwareFoundation.Python.3.8_qbz5n2kfra8p0', 'LocalCache', 'local-packages', 'Python38', 'site-packages'))
sys.path.insert(0, "/vol/fob-vol7/mi19/harnisph/flair")
sys.path.insert(0, os.path.join("vol", "fob-vol7", "mi19", "harnisph", "flair"))


import flair
from flair.data import Corpus
#from flair.datasets import TREC_6
from flair.models import SequenceTagger, TARSSequenceTagger
from flair.trainers import ModelTrainer
from flair.embeddings import WordEmbeddings, TransformerWordEmbeddings, TransformerDocumentEmbeddings
from flair.data import Sentence
from flair.data import MultiCorpus
from flair.datasets import CONLL_03

flair.set_seed(1)

#label_name_map = {
#"S-LOC":"Single Location", "B-PER":"Begin Person", "E-PER":"End Person", "S-ORG":"Single Organization", "S-PER":"Single Person", "B-ORG":"Begin Organization", "E-ORG":"End Organization", "I-ORG":"In Organization", 
#"S-MISC":"Single Miscellaneous", "B-MISC":"Begin Miscellaneous", "E-MISC":"End Miscellaneous", "I-PER":"In Person", "B-LOC":"Begin Location", "E-LOC":"End Location", "I-MISC":"In Miscellaneous", "I-LOC":"In Location"
#}
label_name_map = {
"PRP":"Personal pronoun",
"VBD":"Verb, past tense",
"JJ":"Adjective",
"NN":"Noun, singular or mass",
"VBN":"Verb, past participle",
"CC":"Coordinating conjunction",
"IN":"Preposition or subordinating conjunction",
"MD":"Modal",
"VB":"Verb, base form",
"DT":"Determiner",
"NNP":"Proper noun, singular",
".":"Punct",
"NNS":"Noun, plural",
"EX":"Existential there",
"TO":"to",
"PRP$":"Possessive pronoun",
",":"Comma",
"VBZ":"Verb, 3rd person singular present",
"VBG":"Verb, gerund or present participle",
"POS":"Possessive ending",
"WRB":"Wh-adverb",
"RB":"Adverb",
"WDT":"Wh-determiner",
"VBP":"Verb, non-3rd person singular present",
"CD":"Cardinal number",
"-X-":"Nothing",
"SYM":"Symbol",
":":"Punctuation",
"RP":"Particle",
"''":"Quotation marks",
"(":"Opening bracket",
")":"Closing bracket",
"NNPS":"Proper noun, plural",
"$":"$",
"":"Empty",
"WP":"Wh-pronoun",
"FW":"Foreign word",
"JJS":"Adjective, superlative",
"RBR":"Adverb, comparative",
"UH":"Interjection",
"LS":"List item marker",
"PDT":"Predeterminer",
"RBS":"Adverb, superlative",
"WP$":"Possessive wh-pronoun",
"JJR":"Adjective, comparative",
"\"":"Quotation marks"
}


#print(label_name_map)
corpus = CONLL_03(label_name_map=None, base_path="/vol/fob-vol7/mi19/harnisph/studienprojekt-dokumentation")
print(corpus)
corpus = corpus.downsample(0.1)
print(corpus)

tag_type = "pos"
label_dictionary = corpus.make_label_dictionary(tag_type)
print(label_dictionary)

embeddings = WordEmbeddings("glove")
#embeddings = TransformerWordEmbeddings()

tagger = TARSSequenceTagger(tag_dictionary=label_dictionary, tag_type=tag_type, task_name="TEST_UPOS")
#tagger = SequenceTagger(tag_dictionary=corpus.make_tag_dictionary(tag_type), tag_type=tag_type, hidden_size=256, embeddings=embeddings)

trainer = ModelTrainer(tagger, corpus)
trainer.train(
    base_path='resources/taggers/tars',
    learning_rate=0.1,
    mini_batch_size=32,
    mini_batch_chunk_size=None,
    max_epochs=10,
)
