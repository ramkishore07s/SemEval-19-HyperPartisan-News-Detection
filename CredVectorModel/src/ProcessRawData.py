
# coding: utf-8

# ## Parse XML

# In[1]:

import os
import xml.sax
import shutil


# In[2]:

class ArticleParser(xml.sax.ContentHandler):
    def __init__(self, folder):
        self.folder = folder
        self.CurrentData = ""
        self.title = ""
        self.id = ""
        self.p = ""
        self.published_at = ""
        self.content = []
        self.urls = []
        
    def startElement(self, tag, attributes):
        self.CurrentData = tag
        if tag == "article":
            try: self.published_at = attributes['published-at']
            except: self.published_at = None
            self.id = attributes['id']
            self.title = attributes['title']
        if tag in ["p", "a"]:
            self.content.append("")
        if tag == "a":
            try:
                self.urls.append(attributes['href'].split('/')[2])
                words = attributes['href'].split('/')[2].split('.')
                url = ""
                if not words[0] == 'www': url += words[0]
                else: url += words[1]
                self.content.append(' & ' +  url + ' & ')
            except: pass
            
            
    def endElement(self, tag):
        if tag == 'article':
            with open(self.folder + str(self.id), 'w+') as f:
                f.write(self.title)
                f.write('\n\n')
              #  for url in self.urls:
               #     f.write(url)
                #    f.write('\n')
                f.write('\n\n')
                f.writelines(self.content)
            self.CurrentData = ""
            self.title = ""
            self.id = ""
            self.p = ""
            self.published_at = ""
            self.content = []
            self.urls = []
            
    def characters(self, content):
        if self.CurrentData in ["p", "a"]:
            self.content[-1] += content


# In[3]:

parser = xml.sax.make_parser()
parser.setFeature(xml.sax.handler.feature_namespaces, 0)
Handler = ArticleParser('dataset/training/')
parser.setContentHandler(Handler)
#shutil.rmtree('dataset/training')
#print('deleted')
#os.mkdir('dataset/training')
parser.parse('dataset/articles-training-20180831.xml')


# In[ ]:

parser = xml.sax.make_parser()
parser.setFeature(xml.sax.handler.feature_namespaces, 0)
Handler = ArticleParser('dataset/validation/')
parser.setContentHandler(Handler)
#shutil.rmtree('dataset/validation')
#print('deleted')
#os.mkdir('dataset/validation/')
parser.parse('dataset/articles-validation-20180831.xml')


# In[ ]:

class GroundTruthHandler(xml.sax.ContentHandler):
    def __init__(self, file):
        self.file = file
        
    def startElement(self, tag, attributes):
        file = self.file
        if tag == 'article':
            file.write(attributes['id'])
            file.write(',')
            file.write(attributes['hyperpartisan'])
            file.write(',')
            file.write(attributes['bias'])
            file.write(',')
            file.write(attributes['url'].split('/')[2])
            file.write('\n')


# In[ ]:

file = open('training_labels', 'w+')
parser = xml.sax.make_parser()
parser.setFeature(xml.sax.handler.feature_namespaces, 0)
Handler = GroundTruthHandler(file)
parser.setContentHandler(Handler)
parser.parse('dataset/ground-truth-training-20180831.xml')


# In[ ]:

file = open('validation_labels', 'w+')
parser = xml.sax.make_parser()
parser.setFeature(xml.sax.handler.feature_namespaces, 0)
Handler = GroundTruthHandler(file)
parser.setContentHandler(Handler)
parser.parse('dataset/ground-truth-validation-20180831.xml')


# ## Process Data

# In[5]:

import nltk


# **SENTENCE SPLITTER**

# In[6]:

import re
alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|Mt)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov|me|edu)"

def split_into_sentences(text):
    text = text.replace('?', ' ')   # data cleaning, The given XML file has many ? for unknown or special characters. 
    text = " " + text + "  "
    text = text.replace("\n",".")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences


# **LOAD GLOVE VECTORS**

# In[7]:

import numpy as np

WORDS = []
WORD_ID_MAPPING = {}
VECTORS = []
id = 0
dims = 100
glove_file = '/home/ramkishore.s/glove/glove.6B.100d.txt'
with open(glove_file) as f:
    for l in f:
        line = l.split()
        word = line[0]
        WORDS.append(word)
        WORD_ID_MAPPING[word] , id = id, id + 1
        vect = np.array(line[1:]).astype(np.float)
        VECTORS.append(vect)
        
print(len(WORDS), len(WORD_ID_MAPPING), len(VECTORS))


# In[8]:

UNK = '@@@UNK@@@'
WORDS.append(UNK)
WORD_ID_MAPPING[UNK] = len(WORDS) - 1
VECTORS.extend(np.random.randn(1, 100))


# **PROCESSING DATA FOR DEEP LEARNING**

# In[9]:

TRAIN_FOLDER = 'dataset/training/'
VALIDATION_FOLDER = 'dataset/validation/'
TRAIN_FILES = os.listdir(TRAIN_FOLDER)
VALIDATION_FILES = os.listdir(VALIDATION_FOLDER)
START = '<start>'
END = '<end>'


# In[10]:

WORDS.append(START)
WORDS.append(END)
WORD_ID_MAPPING[START], WORD_ID_MAPPING[END] = len(WORDS) - 2, len(WORDS) - 1
VECTORS.extend(np.random.randn(2, 100))


# **ADD PADDING VECTOR**

# In[13]:

VECTORS.extend(np.zeros(1, 100))


# In[14]:

LINK_PATTERN = re.compile('\&.*\&')


# In[15]:

URLS = {}
URL_ID_COUNT = 0
URL_COUNT = {}


# **FUNCTION TO TOKENIZE FILE, EXTRACT CITED URLS AND TITLE**

# In[16]:

def process_file(file):
    f = open(file).read()
    lines = split_into_sentences(f)
    lines = [line.lower() for line in lines]
    urls, words = [], []
    title = nltk.word_tokenize(lines[0])
    for line in lines[1:]:
        if len(line) < 2: continue
        links = LINK_PATTERN.findall(line)
        if len(links) > 0: 
            try:
                urls.append(links[0].split()[1])
                line.replace(links[0], ' ')
            except: 
                urls.append('$')
        else: 
            urls.append('$')
        words.append([START] + nltk.word_tokenize(line) + [END])
    return title, urls, words


# **FUNCTION TO MAP TOKENS TO NUMBERS**

# In[17]:

def convert_to_vectors(title, urls, lines):
    global URL_ID_COUNT, URLS, WORD_ID_MAPPING, URL_COUNT
    
    url_ids, line_ids = [], []
    for url in urls:
        if url not in URLS:
            URLS[url], URL_ID_COUNT, URL_COUNT[url] = URL_ID_COUNT, URL_ID_COUNT + 1, 0
        URL_COUNT[url] += 1
        url_ids.append(URLS[url])
        
    for line in lines:
        line_ids.append([])
        for word in line:
            if word in WORD_ID_MAPPING:
                line_ids[-1].append(WORD_ID_MAPPING[word])      # word in vocabulary
            else:
                line_ids[-1].append(WORD_ID_MAPPING[UNK])       # unknown word

    title_ids = [WORD_ID_MAPPING[w] if w in WORD_ID_MAPPING else WORD_ID_MAPPING[UNK] for w in title]
    
    return  title_ids, url_ids, line_ids


# **PROCESSING OUTPUTS**

# In[18]:

OUTPUTS = {}


# In[19]:

def load_outputs(file, dict_):
    lines = open(file).readlines()
    for line in lines:
        words = line.split(',')
        output = []
        if words[1] == 'true':
            output.append(True)
        else:
            output.append(False)
        output.append(words[2])
        dict_[words[0]] = output


# In[20]:

load_outputs('dataset/training_labels', OUTPUTS)
load_outputs('dataset/validation_labels', OUTPUTS)


# **PROCESS DATA AND STORE IN PKL FORMAT**

# In[21]:

import _pickle as pickle


# In[22]:

def process(folder, per_file=100000, name='training'):              # per_file -> no of files to store in one pkl file
    # process each file
    # convert them to vectors
    # store them in pickle/json periodically
    FINAL_DATA = []
    count = 0
    file_id  = 0
    for file in os.listdir(folder):
        print(count + file_id * per_file, end='\r')
        title, urls, words = process_file(folder + file)
        title_ids, url_ids, line_ids = convert_to_vectors(title, urls, words)
        FINAL_DATA.append([title_ids, url_ids, line_ids] + OUTPUTS[file])
        count += 1
        if count == per_file -1:
            pickle.dump(FINAL_DATA, open(name + "_" + str(file_id), 'wb+'))
            FINAL_DATA = []
            file_id = file_id + 1
            count = 0


# In[23]:

process(TRAIN_FOLDER, name='training')


# In[ ]:

process(VALIDATION_FOLDER, name='validation')


# **STORE**
# * Word Vectors
# * Word Mappings
# * Url Mappings

# In[ ]:

pickle.dump(VECTORS, open('glove_100d_vectors', 'wb+'))


# In[ ]:

pickle.dump(WORD_ID_MAPPING, open('word_mappings', 'wb+'))
pickle.dump(URLS, open('urls', 'wb+'))
pickle.dump(URL_COUNT, open('url_count', 'wb+'))


# In[ ]:



