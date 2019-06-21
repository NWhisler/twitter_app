import pandas as pd
import numpy as np
import xgboost as xgb
import nltk
import spacy
import os
import logging
import json
from nltk.corpus import stopwords
from string import punctuation
from nltk.tokenize import RegexpTokenizer
from afinn import Afinn
from numpy.linalg import norm
from pandas.io.json import json_normalize
from collections import defaultdict
from datetime import datetime
from warnings import filterwarnings
from utils import twitter_model_logging

class twitter_model(object):

	def __init__(self,file):

		filterwarnings('ignore','elementwise')
		logging.basicConfig(filename='twitter_model.log',level=logging.DEBUG)
		self.twitter_logger = twitter_model_logging()

		if isinstance(file,str) != True:

			logging.debug('File path not string. Raised IOError.' + datetime.now().strftime('%a %b %d %Y at %H:%M:%S'))
			raise ValueError('File path not string.')

		else:
            
			if file.endswith('csv'):

				try:

					self.df = pd.read_csv(file)

				except:

					logging.debug('Incorrect data format. Raised IOError.' + datetime.now().strftime('%a %b %d %Y at %H:%M:%S'))
					raise IOError('Incorrect data format.')
                
			elif file.endswith('json'):

				try:

					self.df = pd.read_json(file)

				except:

					logging.debug('Incorrect data format. Raised IOError.' + datetime.now().strftime('%a %b %d %Y at %H:%M:%S'))
					raise IOError('Incorrect data format.')

			else:
                
				logging.debug('Incorrect file format. Raised IOError.' + datetime.now().strftime('%a %b %d %Y at %H:%M:%S'))
				raise IOError('Incorrect file format.')

		self.tickers = self.df.ticker.values              

		self.columns_required = ['text','background_image','favorite_count','geo_enabled','reply','urls',\
							'user.favourites_count','user.friends_count','user.listed_count',\
							'user.statuses_count','user_mentions','verified']

		self.current_columns = self.df.columns

		for col in self.columns_required:

			if col not in self.current_columns:

				logging.debug('Missing  ' + col + ' column. ' + datetime.now().strftime('%a %b %d %Y at %H:%M:%S'))
				raise ValueError('Missing ' + col + ' column.')

		self.df = self.df[self.columns_required]

		self.df = self.df.dropna()
		
		if len(self.df) == 0:

			logging.debug('All samples contained NaN values. ' + datetime.now().strftime('%a %b %d %Y at %H:%M:%S'))
			raise ValueError('Removed all samples due to NaN values.')

		self.column_dtypes = {'text': 'object','background_image': 'int64','favorite_count': 'int64','geo_enabled': 'int64',\
						 'reply': 'int64','urls': 'int64','user.favourites_count': 'int64','user.friends_count': 'int64',\
						 'user.listed_count': 'int64','user.statuses_count': 'int64','user_mentions': 'int64',\
						 'verified': 'int64'}

		for col in self.columns_required:

			if self.df[col].dtype != np.dtype(self.column_dtypes[col]):

				try:

					self.df[col] = self.df[col].astype(self.column_dtypes[col])
					
				except:
                    
					logging.debug('Invalid data type for ' + col + 'column. ' + \
								  datetime.now().strftime('%a %b %d %Y at %H:%M:%S'))
					raise TypeError('Invalid data input for ' + col)

		nltk.download('stopwords')
		punctuation_parse = [mark for mark in punctuation]
		word_parse = stopwords.words('english')

		self.text = self.df.text.values

		spacy_file = self.twitter_logger.check_file_path('en_core_web_sm-2.1.0/en_core_web_sm/en_core_web_sm-2.1.0')
		self.nlp = spacy.load(spacy_file)

		self.tknzr = RegexpTokenizer('[a-zA-Z]+')
		self.stop_words = punctuation_parse + word_parse

		embedding_file = self.twitter_logger.check_file_path('glove.6B.300d.txt')
		fh = open(embedding_file,'r',buffering=4096,encoding='UTF-8')
		self.word_embeddings = {}

		for line in fh:

			vec = line.split()
			self.word_embeddings[vec[0]] = np.asarray(vec[1:],dtype=np.float32)

		train_vec_file = self.twitter_logger.check_file_path('embedded_vecs/train_embedded_vecs.csv')                   
		df_train_embedded_vecs = pd.read_csv(train_vec_file)
		self.train_embedded_vecs = df_train_embedded_vecs[df_train_embedded_vecs.columns[1:]].values
		self.twitter_logger.check_array_shape(self.train_embedded_vecs,300)

		train_norm_file = self.twitter_logger.check_file_path('embedded_vec_norms/train_norms.csv')
		df_train_embedded_vec_norms = pd.read_csv(train_norm_file)
		self.train_norms = df_train_embedded_vec_norms[['vec_norms']].values.reshape(len(df_train_embedded_vec_norms))
  
		entity_file = self.twitter_logger.check_file_path('entities/train_entities.csv')
		self.entity_list = pd.read_csv(entity_file,header=None).values
		self.entity_list = sorted(self.entity_list.reshape(len(self.entity_list)))

		pos_file = self.twitter_logger.check_file_path('pos_tags/train_pos.csv')
		self.pos_tag_list = pd.read_csv(pos_file,header=None)
		self.pos_tag_list = sorted(self.pos_tag_list.values.reshape(len(self.pos_tag_list)))

	def text_tokens(self):

		tweet_tokens = []

		for tweet in self.text:

			token_list = []
			tokens = self.tknzr.tokenize(tweet)

			for token in tokens:

				token_value=token.lower()

				if token_value not in self.stop_words:

					token_list.append(token_value)

			tweet_tokens.append(token_list)

		self.df['tokens'] = tweet_tokens

		token_idx = [idx for idx,token in enumerate(self.df.tokens.values) if len(token) == 0]
		self.df = self.df.drop(token_idx,axis = 0)
		self.df = pd.DataFrame(self.df.values,columns=self.df.columns,index=list(range(len(self.df))))

		self.tokens = self.df.tokens
		self.text = self.df.text

	def sentiment(self):

		af = Afinn()

		self.df['sentiment'] = [af.score(' '.join(tweet)) for tweet in self.tokens]

	def pos(self):

		pos_tags = []

		for tokens in self.tokens:

			tags = []
			nlp_tags = self.nlp(' '.join(tokens))

			for token in nlp_tags:
				
				tags.append(token.pos_)

			tags = np.array(tags)
			current_tags = {}

			for tag in self.pos_tag_list:

				current_tags[tag] = len(tags[tags==tag])

			pos_tags.append(current_tags)

		self.df_pos = json_normalize(pos_tags)

	def entity(self):

		entities = []

		for tweet in self.text:

			current_entity = {}
			current_entities = []

			for ent in self.nlp(tweet).ents:

				current_entities.append(ent.label_)

			current_entities = np.array(current_entities)

			for ent in self.entity_list:

				current_entity[ent] = len(current_entities[current_entities==ent])

			entities.append(current_entity)

		self.df_entities = json_normalize(entities)

		self.df = pd.concat([self.df,self.df_pos,self.df_entities],axis=1)

	def embed_vecs(self):

		self.embedded_vecs = np.array([sum([self.word_embeddings.get(word,np.zeros(300,)) for word in tokens])/len(tokens) for \
				tokens in self.tokens])
		self.df_embedded_vecs = pd.DataFrame(self.embedded_vecs)

	def tweet_norms(self):

		self.norm_vecs = []

		for vec in self.embedded_vecs:

			self.norm_vecs.append(norm(vec))

		self.norm_vecs = np.array(self.norm_vecs)
		self.norm_vecs[np.isinf(self.norm_vecs)] = 0
		self.norm_vecs[np.isnan(self.norm_vecs)] = 0

	def embedded_cos_distance(self):

		tweets = []

		for idx,tweet in enumerate(self.embedded_vecs):

			distance = np.mean(self.train_embedded_vecs.dot(tweet)/np.sqrt(self.norm_vecs[idx]*self.train_norms))
			tweets.append([distance])

		tweets = np.array(tweets)
		tweets[np.isinf(tweets)] = 0
		tweets[np.isnan(tweets)] = 0

		self.df['cos_vecs'] = tweets

	def vec_diff(self):

		delta_vecs = []

		for vec in self.embedded_vecs:

			delta_vecs.append(np.sum(np.mean(np.abs(self.train_embedded_vecs - vec),axis=0)))

		self.df['delta_vecs'] = delta_vecs

		self.df = pd.concat([self.df,self.df_embedded_vecs],axis=1)

		self.df = self.df.drop(['text','tokens'],axis=1)

	def feature_generation(self):

		self.text_tokens()

		self.sentiment()

		self.pos()

		self.entity()

		self.embed_vecs()

		self.tweet_norms()

		self.embedded_cos_distance()

		self.vec_diff()

	def evaluate(self):

		params = {'max_depth':5,'objective':'binary:logistic'}
		x_test = xgb.DMatrix(self.df.values)
		bst = xgb.Booster(params)
		bst.load_model('boost_model.bin')
		predictions = np.round(bst.predict(x_test))

		return predictions