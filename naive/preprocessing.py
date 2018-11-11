import re
from os import listdir
from Stemmer import Stemmer
from nltk.corpus import stopwords
import numpy as np

def stemming(data): 
	stemmer = Stemmer("english")
	stemmedData = [stemmer.stemWord(key) for key in data]
	return stemmedData

def removeStopWords(data):
	stop_words = set(stopwords.words('english'))
	filteredData = [word for word in data if not word in stop_words]
	return filteredData

def tokenization(data):
	tokenizedData = re.findall(r'[a-z]+',data)
	return tokenizedData

def dataPreprocessing(data):
	tokenizedData = tokenization(data)
	filteredData = removeStopWords(tokenizedData)
	stemmedData = stemming(filteredData)
	return (" ".join(stemmedData))

def processData(title,curtitle,article,curarticle):
	ttitle = dataPreprocessing(curtitle)
	tarticle = dataPreprocessing(curarticle)
	title.append(ttitle)
	article.append(tarticle)

def readData(path,t="train"):
	docFeature = []
	count = 0
	title = []
	article = []
	numCitation = []
	for articleName in listdir(path):
		filename = path + articleName
		file = open(filename,"r",encoding='utf8',errors='ignore')
		data = file.read().lower()
		file.close()
		data = data.split("\n\n")	
		
		curtitle = str(data[0])
		curnumcitation = len(str(data[1]).split("\n"))
		if len(data[2]) == 0 and len(data) > 3:
			data[2] = data[3]
		
		curarticle = str(data[2])
		numCitation.append(curnumcitation)	
		processData(title,curtitle,article,curarticle)
		print(t+" "+str(count))		
		count += 1		
	return title,article,numCitation

def readLabel(filename,t="train"):
	docBiasLabel = []
	docTruthLabel = []
	file = open(filename,"r",encoding='utf8',errors='ignore')
	data = file.read()
	file.close()
	count = 0
	biaslabels = {"left":0,"right":1,"right-center":2,"left-center":3,"least":4}
	truthlabels = {"false":0,"true":1}
	data = data.split("\n")
	for item in data:
		if len(item)>0:
			item = item.split(",")
			docBiasLabel.append(biaslabels[item[2]])
			docTruthLabel.append(truthlabels[item[1]])
			print(t+" "+str(count))
			count+=1
	return docBiasLabel,docTruthLabel

