import sys
import os
#sys.path.append('''C:\Users\pharn\flair\flair''')
sys.path.insert(0, os.path.join('C:/', 'Users', 'pharn', 'flair'))
sys.path.insert(0, os.path.join('C:/', 'Users', 'pharn', 'AppData', 'Local', 'Packages', 'PythonSoftwareFoundation.Python.3.8_qbz5n2kfra8p0', 'LocalCache', 'local-packages', 'Python38', 'site-packages'))

import flair
from flair.data import Corpus
#from flair.datasets import TREC_6
from flair.models import SimpleSequenceTagger
from flair.trainers import ModelTrainer
from flair.embeddings import WordEmbeddings
from flair.data import Sentence
from flair.data import MultiCorpus
from flair.datasets import UD_ENGLISH, UD_GERMAN




# 1. get the corpora - English and German UD
corpus: MultiCorpus = MultiCorpus([UD_ENGLISH(), UD_GERMAN()]).downsample(0.1)

# 2. what tag do we want to predict?
tag_type = 'upos'

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
print(tag_dictionary)



#4 embedding
glove_embedding = WordEmbeddings('glove')

#5 sequence tagger
tagger = SimpleSequenceTagger(
    embeddings=glove_embedding,
    tag_dictionary=tag_dictionary,
    tag_type=tag_type
)
         


# 6. initialize trainer
#trainer: ModelTrainer = ModelTrainer(tagger, corpus)

# 7. start training
#trainer.train('resources/taggers/example-universal-pos',
#              learning_rate=0.1,
#              mini_batch_size=32,
#              max_epochs=1,
#              )



sentence = Sentence('Who built the Eiffel Tower ?')
tagger.predict(sentence)
print(sentence)