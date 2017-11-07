#imports
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

from nltk.stem import WordNetLemmatizer

from nltk.corpus import wordnet

import random
from nltk.corpus import movie_reviews

import pickle

from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
from statistics import mode

import io  

class VoteClassifier(ClassifierI):
	def __init__(self, *classifers):
		self._classifiers = classifers
	def classify(self, features):
		votes = []
		for c in self._classifiers:
			v = c.classify(features)
			votes.append(v)
		return mode(votes)
	def confidence(self, features):
		votes = []
		for c in self._classifiers:
			v = c.classify(features)
			votes.append(v)
		choice_votes =  votes.count(mode(votes))
		conf = choice_votes/len(votes)
		return conf


text = "HEllo there, how are you? python is not just awesome it's great What about some toast?"
text1 = "This is an example showing off stop word filteration"
#Preprocessing about breaking into words and sentences in text{Tokenizer}
# print sent_tokenize(text)
# print word_tokenize(text)

#Filtering the not important words from the sentence like a an the
# sw = set(stopwords.words('english'))
# words = word_tokenize(text1)
# filtered_sentence = []
# for w in words:
# 	if w not in sw:
# 		filtered_sentence.append(w)
# print filtered_sentence

#Making similar word to thier rood word ##UseWORDNET and SINSET
# ps = PorterStemmer()
# words = ['python', 'pythoner', 'pythonly','pythonable']
# for w in words:
# 	print ps.stem(w)

#Making custom tokenizdr and tagging them with what type of word it is noun pronoun etc
# traint = state_union.raw('2005-GWBush.txt')
# samplet = state_union.raw('2006-GWBush.txt')
# customtokenizer = PunktSentenceTokenizer(traint)
# tokenized = customtokenizer.tokenize(samplet)

# def process_content():
# 	try:
# 		for i in tokenized:
# 			words = nltk.word_tokenize(i)
# 			tagged = nltk.pos_tag(words)
# 			print tagged
# 	except Exception as e:
# 		print str(e)
# process_content()

#It lemmatizes that means it makes synonmus to a simliar group it's better to grouping the word better than stemmin
# lemmatizer = WordNetLemmatizer()
# print lemmatizer.lemmatize('cats')
# print lemmatizer.lemmatize('better', pos='a')
# print lemmatizer.lemmatize('worst')
# print lemmatizer.lemmatize('running', pos='v')

# syns = wordnet.synset('program')
# print (syns[0].examples())
#Finding similatrity between two words and getting the synnoms and antoymns of given words
# a,s=[],[]
# for syn in wordnet.synsets('good'):
# 	for l in syn.lemmas():
# 		s.append(l.name())
# 		if l.antonyms():
# 			a.append(l.antonyms()[0].name())

# print set(s)
# print set(a)

# #semantic similarity
# w1=wordnet.synset('ship.n.01')
# w2=wordnet.synset('boat.n.01')
# print w1.wup_similarity(w2)

# documents = [(list(movie_reviews.words(fileid)),category) for category in movie_reviews.categories() for fileid in movie_reviews.fileids(category)]
# # random.shuffle(documents)

# all_words = []
# for w in movie_reviews.words():
# 	all_words.append(w.lower())


###Getting more dataset
short_pos = open('positive.txt','rU').read().decode('latin-1')
short_neg = open('negative.txt','rU').read().decode('latin-1')

documents = []

for r in short_neg.split('\n'):
	documents.append((r,'neg'))
for r in short_pos.split('\n'):
	documents.append((r,'pos'))

all_words = []
short_pos_words = word_tokenize(short_pos)
short_neg_words = word_tokenize(short_neg)

for r in short_pos_words:
	all_words.append(r.lower())
for r in short_neg_words:
	all_words.append(r.lower())

###


all_words = nltk.FreqDist(all_words)
# print all_words.most_common(15)
# print all_words['stupid']

word_feature = list(all_words.keys()[:5000])
def find_feature(document):
	words = set(document)
	feature = {}
	for w in word_feature:
		feature[w]=(w in words)
	return feature
#print find_feature(movie_reviews.words('neg/cv000_29416.txt'))
featuresets = [(find_feature(rev), category) for (rev , category) in documents]
# print featuresets

random.shuffle(featuresets)

training_Set = featuresets[:10000]
testing_set = featuresets[10000:]

# #using naive bayes algo for creating a classifier
# classifier = nltk.NaiveBayesClassifier.train(training_Set)

# #Opening the saved classifier
# classifier_open = open('naivebayes.pickle','rb')
# classifier = pickle.load(classifier_open)
# classifier_open.close()

# print "Original Naive Bayes Algo accuray: ", (nltk.classify.accuracy(classifier, testing_set)*100)
# classifier.show_most_informative_features(15)

# #saving the classifer and not train it again ang again
# save_classifier = open('naivebayes.pickle','wb')
# pickle.dump(classifier, save_classifier)
# save_classifier.close()

MNB_classifier = SklearnClassifier(MultinomialNB())
print "MNB Classifier"
MNB_classifier.train(training_Set)
print "MNB_classifier Naive Bayes Algo accuray: ", (nltk.classify.accuracy(MNB_classifier, testing_set)*100)


# GNB_classifier = SklearnClassifier(GaussianNB())
# GNB_classifier.train(training_Set)
# print "GNB_classifier Naive Bayes Algo accuray: ", (nltk.classify.accuracy(GNB_classifier, testing_set)*100)


BNB_classifier = SklearnClassifier(BernoulliNB())
BNB_classifier.train(training_Set)
print "BNB_classifier Naive Bayes Algo accuray: ", (nltk.classify.accuracy(BNB_classifier, testing_set)*100)


LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_Set)
print "LogisticRegression_classifier Naive Bayes Algo accuray: ", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set)*100)


SGDC_classifier = SklearnClassifier(SGDClassifier())
SGDC_classifier.train(training_Set)
print "SGDC_classifier Naive Bayes Algo accuray: ", (nltk.classify.accuracy(SGDC_classifier, testing_set)*100)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_Set)
print "LinearSVC_classifier Naive Bayes Algo accuray: ", (nltk.classify.accuracy(LinearSVC_classifier, testing_set)*100)


NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_Set)
print "NuSVCB_classifier Naive Bayes Algo accuray: ", (nltk.classify.accuracy(NuSVC_classifier, testing_set)*100)


SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_Set)
print "SVC_classifier Naive Bayes Algo accuray: ", (nltk.classify.accuracy(SVC_classifier, testing_set)*100)



voted_classifier = VoteClassifier(MNB_classifier,BNB_classifier,LogisticRegression_classifier,SGDC_classifier,LinearSVC_classifier,NuSVC_classifier)

print 'Voted classifier accuaracy', (nltk.classify.accuracy(voted_classifier, testing_set)*100)
print 'Classisfication', voted_classifier.classify(testing_set[0][0]),'confidence',voted_classifier.confidence(testing_set[0][0])

