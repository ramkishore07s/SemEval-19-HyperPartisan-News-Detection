import numpy as np
import pickle
from preprocessing import processData,readLabel,readData
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction import DictVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from textblob import TextBlob
from scipy.sparse import csr_matrix

biasPKLfile = "model/biasPKLfile"
trainLabelPath = "data/training_labels"
trainDataPath = "data/training/"
testLabelPath = "data/validation_labels"
testDataPath = "data/validation/"
truthPKLfile = "model/truthPKLfile"

def createNgram(input,size):
	input = input.split(" ")
	output = []
	for i in range(len(input) - size + 1):
		output.append(input[i:i + size])
	return output

def ngramsMatch(title,article,size):
	matchHit = 0
	matchEarlyHit = 0 
	titlegrams = [' '.join(x) for x in createNgram(title, size)]
	
	for gram in titlegrams:
		if gram in article:
			matchHit += 1
		if gram in article[:255]:
			matchEarlyHit += 1
	
	return matchHit,matchEarlyHit

def helpextractFeature(title,article,numcitiation):
	field_stats = []
	field_stats.append(numcitiation)	
	for size in range(1,3):	
		matchHit,matchEarlyHit = ngramsMatch(title,article,size)
		field_stats.append( matchHit)
		field_stats.append(matchEarlyHit)
		
	return field_stats

def load_model(modelFile):
	try:
		with open(modelFile, "rb") as pkl:
			clf = pickle.load(pkl)
	except (IOError, pickle.UnpicklingError) as e:
		raise e
	return clf

class extractFeature(BaseEstimator, TransformerMixin):
	def fit(self, x, y=None):
		return self
	def transform(self, posts):
		stats = []		
		for field in posts:
			stats.append(field)
		stats = np.matrix(stats,dtype=np.float128)
		stats = csr_matrix(stats)	
		
		return stats

class ItemSelector(BaseEstimator, TransformerMixin):
	def __init__(self, key):
		self.key = key

	def fit(self, x, y=None):
		return self

	def transform(self, data_dict):
		return data_dict[self.key]

class TitleArticleExtractor(BaseEstimator, TransformerMixin):
	def fit(self, x, y=None):
		return self

	def transform(self, posts):
		stats = []
		features = np.recarray(shape=(len(posts),), dtype=[('title', object), ('article', object),('ngram', object)])
		for i, post in enumerate(posts): 
			title,numcitiation,article = post[:3]
			features['title'][i] = title
			features['article'][i] = article 
			features['ngram'][i] = helpextractFeature(title,article,numcitiation)

		return features

class Text_Stats(BaseEstimator, TransformerMixin):
	def fit(self, x, y=None):
		return self

	def transform(self, posts):		
		stats = []
		for i,field in enumerate(posts):	
			field_stats = []					
			polarity, subjectivity = TextBlob(field).sentiment					
			if (polarity<0):
				polarity = 0
			if (subjectivity<0):
				subjectivity = 0
			field_stats.append(polarity)
			field_stats.append(subjectivity)
			stats.append(field_stats)
			
		stats = np.matrix(stats,dtype=np.float128)
		stats = csr_matrix(stats)	

		return stats


class DenseTransformer(TransformerMixin):
    def transform(self, X, y=None, **fit_params):
        return X.todense()

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self

def train():    
	docBiasLabel,docTruthLabel = readLabel(trainLabelPath)
	title,article,numCitation = readData(trainDataPath)
	traindata = list(zip(title,numCitation,article))
	dataExtractor = Pipeline([('TitleArticleExtractor', TitleArticleExtractor()),])
	TfidfTitle = Pipeline([
						('selector', ItemSelector(key='title')),
						('vect', TfidfVectorizer(min_df = 0.01)),
						('to_dense', DenseTransformer()),
					])
	TfidfArticle = Pipeline([
						('selector', ItemSelector(key='article')),
						('vect', TfidfVectorizer(min_df = 0.01)),
						('to_dense', DenseTransformer()),
					])
	textStatsTitle = Pipeline([
					('selector', ItemSelector(key='title')),
					('stats', Text_Stats()),  
					('to_dense', DenseTransformer()),
					
				])
	textStatsArticle = Pipeline([
					('selector', ItemSelector(key='article')),
					('stats', Text_Stats()),  
					('to_dense', DenseTransformer()),					
				])

	matchNgrams =  Pipeline([
					('selector', ItemSelector(key='ngram')),
					('func', extractFeature()), 
					('to_dense', DenseTransformer()),
										
				])

	
	bias_clf = Pipeline([
			('TitleArticleExtractor', dataExtractor),
			('union', FeatureUnion(
				transformer_list=[
									('tfidf_title', TfidfTitle),
									('tfidf_article', TfidfArticle),
									('text_stats_title', textStatsTitle),
									('text_stats_body', textStatsArticle),
									('matchngrams', matchNgrams),
								],
							)),
					('clf', MultinomialNB()),
			])

	bias_clf.fit(traindata, docBiasLabel)

	with open(biasPKLfile,"wb") as f_pk:
		pickle.dump(bias_clf,f_pk,pickle.HIGHEST_PROTOCOL)
	

	truth_clf = Pipeline([			
			('TitleArticleExtractor', dataExtractor),
			('union', FeatureUnion(
				transformer_list=[
									('tfidf_title', TfidfTitle),
									('tfidf_article',TfidfArticle),
									('text_stats_headline',textStatsTitle),
									('text_stats_body', textStatsArticle),
									('matchngrams', matchNgrams),
								],
							)),
					('clf', GaussianNB()),
				])
	truth_clf.fit(traindata, docTruthLabel)

	with open(truthPKLfile,"wb") as f_pk:
		pickle.dump(truth_clf,f_pk,pickle.HIGHEST_PROTOCOL)


def test():
	bias_clf = load_model(biasPKLfile)
	truth_clf = load_model(truthPKLfile)
	docBiasLabel,docTruthLabel = readLabel(testLabelPath,"test")
	title,article,numCitation = readData(testDataPath,"test")  
	testdata = list(zip(title,numCitation,article))
	truth_pred = truth_clf.predict(testdata)
	bias_pred = bias_clf.predict(testdata)
	print("Truth value accuracy: ",np.mean(truth_pred == docTruthLabel))
	print("Bias value accuracy: ",np.mean(bias_pred == docBiasLabel))

def main(): 
	train()
	test()

if __name__ == '__main__':
	main()






	
