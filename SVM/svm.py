from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
# from pattern.en import sentiment
import pickle
import numpy as np
import nltk
from argparse import ArgumentParser
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score)
from textblob import TextBlob
from process_text import process_labels,process_train

TRAIN_DATA_DIR = "data\\training\\"
TRAIN_LABEL_FILE = "data\\training_labels"
MODEL_FILE_TRUTH = "model\\model_file_truth"
MODEL_FILE_BIAS = "model\\model_file_bias"
TEST_LABEL_FILE = "data\\validation_labels"


class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]

class HeadlineBodyFeaturesExtractor(BaseEstimator, TransformerMixin):
    """Extracts the components of each input in the data: headline, body, and POS tags for each"""
    def fit(self, x, y=None):
        return self

    def transform(self, posts):
        features = np.recarray(shape=(len(posts),), dtype=[('headline', object), ('article_body', object)])
        for i, post in enumerate(posts): 
            headline, article = post[:2]
            features['headline'][i] = headline
            features['article_body'][i] = article
        
        return features

class Text_Stats(BaseEstimator, TransformerMixin):
    """Extract text statistics from each document"""

    def fit(self, x, y=None):
        return self

    def transform(self, text_fields):
        stats = []
        for field in text_fields:
            field_stats = {}
            tok_text = field.split()
            try:
                sent_lengths = [len(nltk.word_tokenize(s)) for s in nltk.sent_tokenize(field)]
                av_sent_len = float(sum(sent_lengths))/len(sent_lengths)
            except:
                av_sent_len = 0
            
            polarity, subjectivity = TextBlob(field).sentiment
            field_stats['sent_len'] = av_sent_len
            field_stats['polarity'] = polarity
            field_stats['subjectivity'] = subjectivity
            stats.append(field_stats)
        return stats


def train():
    train_title,train_url,train_body = process_train(TRAIN_DATA_DIR)

    # print(train_body)
    # vocab_list = list(set(vocab))
    train_truth,train_bias = process_labels(TRAIN_LABEL_FILE)
    # print(len(train_truth))

    train_texts = list(zip(train_title,train_body))
    # print(list(train_texts))
    pipeline = Pipeline([
        # Extract the subject & body
        ('HeadlineBodyFeatures', HeadlineBodyFeaturesExtractor()),

        # Use FeatureUnion to combine the features from subject and body
        ('union', FeatureUnion(
            transformer_list=[

                #Pipeline for pulling features from articles

                ('ngrams_title', Pipeline([
                    ('selector', ItemSelector(key='headline')),
                    ('vect', TfidfVectorizer(ngram_range=(1,2), token_pattern = r'\b\w+\b',max_df=0.5,stop_words=None)),
                ])),

                ('ngrams_text', Pipeline([
                    ('selector', ItemSelector(key='article_body')),
                    ('vect', TfidfVectorizer(ngram_range=(1,2), token_pattern = r'\b\w+\b',max_df=0.5,stop_words=None)),
                ])),

                ('text_stats_headline', Pipeline([
                ('selector', ItemSelector(key='headline')),
                ('stats', Text_Stats()),  # returns a list of dicts
                ('vect', DictVectorizer()),  # list of dicts -> feature matrix
            ])),

            ('text_stats_body', Pipeline([
                ('selector', ItemSelector(key='article_body')),
                ('stats', Text_Stats()),  # returns a list of dicts
                ('vect', DictVectorizer()),  # list of dicts -> feature matrix
            ])),
            

                ],
                )),
                ('svc', LinearSVC(C=1.0,max_iter=10000)),
    ])

    pipeline.fit(train_texts, train_truth)
    # pipeline.classify()

    with open(MODEL_FILE_TRUTH,"wb") as f_pk:

        pickle.dump(pipeline,f_pk,pickle.HIGHEST_PROTOCOL)

    # pipeline_bias = Pipeline([
    #     # Extract the subject & body
    #     ('HeadlineBodyFeatures', HeadlineBodyFeaturesExtractor()),

    #     # Use FeatureUnion to combine the features from subject and body
    #     ('union', FeatureUnion(
    #         transformer_list=[

    #             #Pipeline for pulling features from articles

    #             ('ngrams_title', Pipeline([
    #                 ('selector', ItemSelector(key='headline')),
    #                 ('vect', TfidfVectorizer(ngram_range=(1,2), token_pattern = r'\b\w+\b',max_df=0.5,stop_words=None)),
    #             ])),

    #             ('ngrams_text', Pipeline([
    #                 ('selector', ItemSelector(key='article_body')),
    #                 ('vect', TfidfVectorizer(ngram_range=(1,2), token_pattern = r'\b\w+\b',max_df=0.5,stop_words=None)),
    #             ])),

    #                  ('text_stats_headline', Pipeline([
    #             ('selector', ItemSelector(key='headline')),
    #             ('stats', Text_Stats()),  # returns a list of dicts
    #             ('vect', DictVectorizer()),  # list of dicts -> feature matrix
    #         ])),

    #         ('text_stats_body', Pipeline([
    #             ('selector', ItemSelector(key='article_body')),
    #             ('stats', Text_Stats()),  # returns a list of dicts
    #             ('vect', DictVectorizer()),  # list of dicts -> feature matrix
    #         ])),

    #             ],
    #             )),
    #             ('svc', OneVsRestClassifier(LinearSVC(max_iter=10000))),
    # ])

    # pipeline_bias.fit(train_texts, train_bias)

    # with open(MODEL_FILE_BIAS,"wb") as f_pk:

    #     pickle.dump(pipeline_bias,f_pk,pickle.HIGHEST_PROTOCOL)

    

def load_model(model_file=MODEL_FILE_TRUTH):

    try:
        with open(model_file, "rb") as pkl:
            pipeline = pickle.load(pkl)
    except (IOError, pickle.UnpicklingError) as e:
        raise e

    return pipeline


def main():

    argparser = ArgumentParser(description=__doc__)
    argparser.add_argument("-t", "--trainset", action="store",
                           default=None,
                           help=("Path to training data "
                                 "[default: %(default)s]"))
    argparser.add_argument("-tb", "--trainset_bias", action="store",
                           default=None,
                           help=("Path to training data "
                                 "[default: %(default)s]"))
    
    argparser.add_argument("-m", "--model", action="store",
                           help="Path to model")
    argparser.add_argument("-c", "--classify", action="store",
                           default=None,
                           help=("Path to data to classify "
                                 "[default: %(default)s]"))
    args = argparser.parse_args()

    if args.classify:
        test_title,test_url,test_body = process_train(args.classify,cnt=2000)
        test_texts = list(zip(test_title,test_body))
        pipeline = load_model(MODEL_FILE_TRUTH)
        predictions = pipeline.predict(test_texts)
        pipeline1 = load_model(MODEL_FILE_BIAS)
        predictions1 = pipeline1.predict(test_texts)
       
        test_truth,test_bias = process_labels(TEST_LABEL_FILE,len(test_texts)-1)
        print(len(test_truth))
        print(len(predictions))
        accuracy = accuracy_score(test_truth, predictions)
        print(accuracy)
        accuracy = accuracy_score(test_bias, predictions1)
        print(accuracy)
    # elif args.train
    else:
        train()



if __name__ == "__main__":
    main()