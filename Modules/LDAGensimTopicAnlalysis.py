# Import needed libraries
import pickle
import string as string
from time import time

import gensim
import nltk
import pyLDAvis.gensim
from gensim import corpora
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from spacy.lang.en import English

# Download needed nltk resources
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')


# Function to tokenize words
def tokenize(text):
    parser = English()
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens


# Function to lemma the word by looking at the pos tag first
def get_lemma(word):
    lemmatizer = WordNetLemmatizer()
    for w, tag in pos_tag(tokenize(word)):
        wntag = tag[0].lower()
        wntag = wntag if wntag in ['a', 'r', 'n', 'v'] else None
        if not wntag:
            return word
        else:
            return lemmatizer.lemmatize(word, wntag)


# Function to prepare text data for LDA
def prepare_text_for_lda(text, min_word_len=2):
    exclude = set(string.punctuation)
    # en_stop = set(nltk.corpus.stopwords.words('english'))
    # en_stop.remove('not')
    # en_stop.remove('no')
    if text is None:
        pass
    else:
        text = ''.join(ch for ch in text if ch not in exclude)
        tokens = tokenize(text)
        tokens = [token for token in tokens if len(token) > min_word_len]
        # tokens = [token for token in tokens if token not in en_stop]
        tokens = ' '.join([get_lemma(token) for token in tokens])
        return tokens


# Function to vectorize the input text for training the LDA model
def vect_train_text(data, text_column, ngram_range=(1, 1), save_path=""):
    vect = CountVectorizer(ngram_range=ngram_range)
    corpus_vect = vect.fit_transform(data[text_column])
    corpus = gensim.matutils.Sparse2Corpus(corpus_vect, documents_columns=False)
    dictionary = corpora.Dictionary.from_corpus(corpus,
                                                id2word=dict((id, word) for word, id in vect.vocabulary_.items()))
    pickle.dump(vect, open("{}_vect.pkl".format(save_path), 'wb'))
    pickle.dump(corpus, open('{}_corpus.pkl'.format(save_path), 'wb'))
    dictionary.save('{}_dictionary.gensim'.format(save_path))


# Function to create new corpus to feed into the pre-trained model
def build_new_input_corpus(text, pre_fitted_vect):
    text = [text]
    new_x = pre_fitted_vect.transform(text)
    new_c = gensim.matutils.Sparse2Corpus(new_x, documents_columns=False)
    return new_c


# Function to get the most probable topic from the model output
def get_most_proba_topic(tuple_list):
    tuple_list = list(tuple_list[0])
    most_proba = max(tuple_list, key=lambda x: x[1])
    return most_proba[0]


# Create function to train the LDA model on the given corpus and dictionary
def train_lda_model(n_topics, corpus, dictionary, save_path="", random_state=101):
    start_time = time()
    lda_model = gensim.models.ldamodel.LdaModel(corpus, num_topics=n_topics, id2word=dictionary, passes=15,
                                                random_state=random_state)
    lda_model.save('{}.gensim'.format(save_path))
    end_time = time()
    print('Done fitting LDA with {} topics... Time elapsed {}'.format(n_topics, ((end_time - start_time) / 60)))


# Take trained model and extract the prepared data from the LDAvis library to be able to visualise the inner workings
# of the model
def prepare_model_json(lda_model, corpus, dictionary, save_path=""):
    start_time = time()
    lda_model_json = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary, sort_topics=False)
    pyLDAvis.save_json(lda_model_json, '{}.json'.format(save_path))
    end_time = time()
    print('Took {} minutes to prepare model data'.format((end_time - start_time) / 60))
