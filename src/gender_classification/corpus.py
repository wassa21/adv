import ast
import pdb
from sklearn.model_selection import StratifiedKFold
from collections import Counter
import random
import numpy as np
from transformers import BertTokenizer,AutoTokenizer
from helper import pad_sequences


import nltk
        
class _Document:
    def __init__(self, content, annotation, lang, document_id, target, do_lowercase=True):
        self.document_id       = document_id
        self.lang              = lang
        self.speaker           = annotation.speaker
        self.date              = annotation.date
        self.rated_bool        = annotation.rated_bool
        self.ratings           = annotation.ratings
        self.duration          = annotation.duration
        self.url               = annotation.url
        self.talk_id           = annotation.talk_id
        self.speaker_gender    = annotation.author_gender
        self.speaker_check     = True
        self.translator_name   = ''
        self.translator_gender = ''
        self.translator_check  = True
        self.reviewer_name     = ''
        self.reviewer_gender   = ''
        self.reviewer_check    = True
        d = ast.literal_eval(annotation.Translation_Info)[lang]
        if 'translator_name' in d.keys():
            self.translator_name   = d['translator_name']
            self.translator_gender = d['translator_gender']
        if 'reviewer_name' in d.keys():
            self.reviewer_name     = d['reviewer_name']
            self.reviewer_gender   = d['reviewer_gender']            

        if target.lower() == 'speaker':
            self.target_gender = self.speaker_gender
        elif target.lower() == 'translator':
            self.target_gender = self.translator_gender
        elif target.lower() == 'reviewer':
            self.target_gender = self.reviewer_gender
        else:
            assert 1 == 2
            
        tokenizer = nltk.word_tokenize
        
        all_texts,all_tokens = [],[]
        for ix,row in content.iterrows():
            text   = row[lang]
            all_texts.append(text)
            
            tokens = text.split(' ')
            all_tokens.append(tokens)
            #tokens = tokenizer(text)
            # all_tokens.append([ token.text if not do_lowercase else token.text.lower() for token in tokens])
            #all_tokens.append([ token if not do_lowercase else token.lower() for token in tokens])

            
        self.all_tokens = all_tokens
        self.all_text   = all_texts
        

        if (self.reviewer_name == '' or self.reviewer_gender == '' or self.reviewer_gender =='unknown'):
            self.reviewer_check = False

        if (self.translator_name == '' or self.translator_gender == '' or self.translator_gender =='unknown'):
            self.translator_check = False
            
        if self.speaker_gender == 'unknown':
            self.speaker_check = False
                
    def __repr__(self):
        return '[' + ' '.join([str(self.document_id),self.lang]) + ']'

class Document:
    def __init__(self,content, annotation, langs, document_id, target, do_lowercase=True):        
        self.documents      = {}
        self.langs          = langs
        self.speaker        = annotation.speaker
        self.speaker_gender = annotation.author_gender
        self.document_id    = document_id # paralel_talks_ted_985_ix.csv document_id: 985
        self.target         = target
        self.topic_label    = annotation.topic
        self.max_document_len = 5500
        for lang in langs:
            d = _Document(content,annotation,lang, document_id=document_id, target=target, do_lowercase=do_lowercase)
            self.documents[lang] = d
            
    def is_ok(self):
        if self.target == 'speaker':
            return self.speaker_gender != 'unknown' and all([len(self.documents[k].all_tokens) !=0 for k in self.documents.keys()]) and all([len([item for sublist in self.documents[k].all_tokens for item in sublist]) < self.max_document_len for k in self.documents.keys()])
        elif self.target == 'translator':
            return all([self.documents[k].translator_check for k in self.documents.keys()]+[len(self.documents[k].all_tokens) !=0 for k in self.documents.keys()]) and all([len([item for sublist in self.documents[k].all_tokens for item in sublist]) < self.max_document_len for k in self.documents.keys()])
        elif self.target == 'reviewer':
            return all([self.documents[k].reviewer_check for k in self.documents.keys()]+[len(self.documents[k].all_tokens) !=0 for k in self.documents.keys()]) and all([len([item for sublist in self.documents[k].all_tokens for item in sublist]) < self.max_document_len for k in self.documents.keys()])
        else:
            assert 1 == 2
        
    def __repr__(self):
        return '[' + ' '.join([str(self.document_id)]+self.langs) + ']'
        
class Corpus:
    def __init__(self, documents, bert_max_len, sector_max_word, bert_model_name='bert-base-cased'):
        self.documents =  documents
        self.indices   =  None
        self.w2ix      =  None
        self.ix2w      =  None
        self.l2ix      =  {'male':0,'female':1}
        self.bert_max_len = bert_max_len
        self.sector_max_word = sector_max_word
        self.bert_model_name = bert_model_name
        
        self.tokenizer =  AutoTokenizer.from_pretrained(
            self.bert_model_name, do_lower_case=False,use_fast=False,pad_token='[PAD]')
        
        #self.tokenizer =  BertTokenizer.from_pretrained(
        #    self.bert_model_name, do_lower_case=False)

        
    def cross_validation(self, lang, n_splits):
        indices = []
        corpus = [d for d in self.documents if d.is_ok()]
        #tmpx = []
        #for d in corpus:
        #    tmpx.append(len([item  for sublist in d.documents['en'].all_tokens for item in sublist]))
        
        labels = [d.documents[lang].target_gender for d in self.documents if d.is_ok()]
        skf = StratifiedKFold(n_splits=n_splits)
        for train_index, test_index in skf.split(corpus, labels):
            indices.append((train_index,test_index))
        self.indices = indices

        
    def make_data_ready(self, cv_ix, lang):
        train,test =  [],[]
        corpus = [d for d in self.documents if d.is_ok()]
        for train_index in self.indices[cv_ix][0]:
            d = corpus[train_index]
            label = d.documents[lang].target_gender
            topic_label = d.topic_label
            #tmp = [(section,label) for section in d.documents[lang].all_tokens] # Section by section
            tmp = [(self.split_document_for_bert(
                [item for sublist in d.documents[lang].all_tokens for item in sublist]),label,topic_label)] # Whole document 
            train += tmp
                                    
        for test_index in self.indices[cv_ix][1]:
            d = corpus[test_index]
            label = d.documents[lang].target_gender
            topic_label = d.topic_label
            #tmp = [(section,label) for section in d.documents[lang].all_tokens] # Section by section 
            tmp = [(self.split_document_for_bert(
                [item for sublist in d.documents[lang].all_tokens for item in sublist]),label,topic_label)] # Whole document
            test += tmp            
        self.train = train
        self.test  = test

        tst_labels = [x[1] for x in self.test]
        trn_labels = [x[1] for x in self.train]
        
        tst_topic_labels = [x[2] for x in self.test]
        trn_topic_labels = [x[2] for x in self.train]
        
        
        print("male/female ration in train set :{}/{}".format(trn_labels.count('male'),trn_labels.count('female')))
        print("male/female ration in test set :{}/{}".format(tst_labels.count('male'),tst_labels.count('female')))
        trn_stats = (trn_labels.count('male'),trn_labels.count('female'))
        tst_stats = (tst_labels.count('male'),tst_labels.count('female'))

        trn_pos_topic = sum(trn_topic_labels)
        trn_neg_topic = len(trn_topic_labels) - trn_pos_topic

        tst_pos_topic = sum(tst_topic_labels)
        tst_neg_topic = len(tst_topic_labels) - tst_pos_topic
        
        print("positive/negative ratio in train set :{}/{}".format(trn_pos_topic,trn_neg_topic))
        print("positive/negative ratio in test set :{}/{}".format(tst_pos_topic,tst_neg_topic))
        trn_stats = (trn_labels.count('male'),trn_labels.count('female'))
        tst_stats = (tst_labels.count('male'),tst_labels.count('female'))
        trn_topic_stats = (trn_pos_topic,trn_neg_topic)
        tst_topic_stats = (tst_pos_topic,tst_neg_topic)
        return (trn_stats,tst_stats,trn_topic_stats,tst_topic_stats)
    def split_document_for_bert(self,text):
        chunks = [" ".join(text[i:i+self.sector_max_word]) for i in range(0,len(text),self.sector_max_word)]
        return chunks 
        
    def iter_batches(self,batch_size, split_name = 'train', shuffle=False):
        if split_name =='train':
            documents = self.train
        else:
            documents = self.test
            
        if shuffle:
            random.shuffle(documents)

        for i in range(0, len(documents), batch_size):
            yield self.convert_batch(documents[i:i+batch_size])

    # batch: list of tuples        
    # batch[0]: Tuple where the first element is list of lists with strings and -> Each element of batch is a document
    # the second element is the label of the document ('male'/'female'). Each string is
    # a  part of the document whose length is shorther than max_len parameter of split_document_for_bert function.
    # batch[0][0][0]: String, text of a section 
    def convert_batch(self,batch):
        batch_input_ids,batch_segment_ids,batch_attention_masks,batch_labels,batch_topic_labels = [],[],[],[],[]
        for doc_tuple in batch:
            topic_label = doc_tuple[2]
            label    = doc_tuple[1]
            sections = doc_tuple[0] # sections of a document
            input_ids,segment_ids,attention_masks = self.bert_processor(sections)
            batch_labels.append(self.l2ix[label])
            batch_topic_labels.append(topic_label)
            batch_attention_masks.append(attention_masks)
            batch_input_ids.append(input_ids)
            batch_segment_ids.append(segment_ids)
        return batch_input_ids,batch_segment_ids,batch_attention_masks,batch_labels,batch_topic_labels


    def bert_processor(self, sentences):
        # Load the BERT tokenizer.
        #print('Loading BERT tokenizer...')
        input_ids = []
        for sent in sentences:
            encoded_sent = self.tokenizer.encode(sent,add_special_tokens = True)            
            input_ids.append(encoded_sent)     

        input_ids = pad_sequences(input_ids, maxlen=self.bert_max_len, dtype="long", 
                                  value=0, truncating="post", padding="post")
        segment_ids = []
        for iid in input_ids:
            segment_ids.append([0]*len(iid))
        
        #print('\nDone.')
        # Create attention masks
        attention_masks = []
        # For each sentence...
        for sent in input_ids:    
            att_mask = [int(token_id > 0) for token_id in sent]
            attention_masks.append(att_mask)
        return (input_ids,segment_ids,attention_masks)
    




# dbmdz/bert-base-turkish-cased
# bert-base-german-cased
# bert-base-cased
# dccuchile/bert-base-spanish-wwm-cased
# camembert-base

