import argparse
import pandas as pd
import numpy as np
import os
from itertools import combinations,groupby
import json
import re
from corpus import Document,Corpus
from models import *
import pdb
from sklearn.metrics import classification_report
from transformers import get_linear_schedule_with_warmup
import random
import logging
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

logger = logging.getLogger(__name__)

def _file_statistics(fpath,non_lang_fields=['title','speaker','duration','tags']):
    df = pd.read_table(fpath,sep=',')
    langs = [cn for cn in df.columns.tolist() if cn not in non_lang_fields]
    output = sum([list(map(list, combinations(langs, i))) for i in range(len(langs) + 1)], [])
    # Remove single_lang contribution
    output = list(filter(lambda x:len(x) >1,output))
    # Sort items 
    list(map(lambda x:x.sort(),output))
    # Remove duplicates
    output.sort()    
    unique_pairs = list(k for k,_ in groupby(output))    
    return (["_".join(x) for x in unique_pairs],len(df[df[langs[0]].apply(len)>1]))

def get_parallel_corpus_id_stats(corpus_dir):
    files = get_corpus(corpus_dir)
    result = {}
    for fname in files:
        langs,span_number = _file_statistics(corpus_dir+fname)
        for lang_tuple in langs:
            if lang_tuple not in result:
                result[lang_tuple] = [0,0]
            result[lang_tuple][0] += 1
            result[lang_tuple][1] += span_number
    return result

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True                  

def get_corpus(corpus_dir,extension='.csv'):
    files = [doc for doc in os.listdir(corpus_dir) if doc.endswith(extension)]
    return files

def clean(df,col_name='en'):
    # Remove rows with only \n
    df = df[~(df == '\n').any(axis=1)]
    # Remove multiple tabs and multiple \ns
    df = df.replace(r'\s+', ' ', regex=True).apply(lambda x: x.str.strip())
    # ? Remove rows with less than N words (N ? 3)
    df = df[df[col_name].apply(lambda x: len(x.split(' ')) > 3)]
    return df

def load_document(df, annotation, langs, iid, target, do_lowercase):        
    df = clean(df,col_name=langs[0])
    doc = Document(df,annotation,langs, document_id=iid, target=target, do_lowercase=do_lowercase)
    return doc

#def load_corpus(corpus_dir, annotation_file, langs, target, do_lowercase,args):
def load_corpus(args):
    files     = get_corpus(args.corpus_dir)
    annotations = pd.read_table(args.annotations,sep=',')
    documents = []
    for fname in files:
        fpath = args.corpus_dir+fname
        df    = pd.read_table(fpath,sep=',')
        if all([lang in df.columns for lang in args.langs]):
            iid = int(re.search('paralel_talks_ted_(.*)_ix.csv',fname).group(1))
            annotation = annotations.iloc[iid]
            document = load_document(
                df[args.langs +['title','speaker','duration','tags']],annotation,args.langs,iid,args.target,args.do_lowercase)
            documents.append(document)
    return Corpus(documents,args.bert_max_len,args.sector_max_word,bert_model_name=args.bert_model_name)

def metrics(preds, golds, target_names=['male','female']):
    result     = classification_report(golds,preds,target_names = target_names,output_dict=True)
    female_acc = sum((1 == golds) * (1 == preds)) / sum((1 == golds))
    male_acc   = sum((0 == golds) * (0 == preds)) / sum((0 == golds))
    result[target_names[0]]['accuracy'] = male_acc
    result[target_names[1]]['accuracy'] = female_acc
    #result['male']['accuracy'] = male_acc
    #result['female']['accuracy'] = female_acc
    #nice_print(result)
    return result

def nice_print(result,cats=['male','female']):
    result['accuracy'] = '{:.2f}'.format(result['accuracy'])
    for g in cats+['weighted avg','macro avg']:
        for met in result[g].keys():
            result[g][met] = '{:.2f}'.format(result[g][met])
    headers = ['precision','recall','f1-score','accuracy']
    header_title = ['    ','pre.','rec.','f1. ','acc.']
    print('\t'.join(header_title))
    logger.info('\t'.join(header_title))
    for g in cats:
        text = '{}\t'.format(g)
        for hix in range(len(headers)):
            head = headers[hix]            
            text = text + result[g][head] + '\t'
        print(text)
        logger.info(text)
    logger.info("overall:{}".format(result['accuracy']))
    print("overall:",result['accuracy'])

def predict(args,model, corpus, split_name, calculate_loss=False):
    model.eval()    
    all_preds,golds = np.array([]),np.array([])
    golds_gender = np.array([])
    ovrl_loss, step_num = 0.0, 0
    with torch.no_grad():
            for step,batch in enumerate(corpus.iter_batches(
                    batch_size=args.bs,split_name=split_name,shuffle=False)):

                batch_input_ids=[torch.tensor(
                    d_input_ids).to(device) for d_input_ids in batch[0]][0]
                
                batch_segment_ids = [torch.tensor(
                    d_segment_ids).to(device) for d_segment_ids in batch[1]][0]

                batch_attention_masks = [torch.tensor(
                    d_attn_masks).to(device) for d_attn_masks in batch[2]][0]

                batch_gender_labels = torch.tensor(batch[3])
                batch_topic_labels = torch.tensor(batch[4]).to(device)
                batch_labels = batch_topic_labels


                
                out,lss = model(input_ids=batch_input_ids,
                                token_type_ids=batch_segment_ids,
                                attention_mask=batch_attention_masks,                                       
                                labels = batch_labels)
                                       

                out = out.cpu()
                batch_labels = batch_labels.cpu()
                probs      = torch.nn.functional.softmax(out,dim=1)
                preds      = torch.argmax(probs, dim=1)

                all_preds  = np.concatenate((all_preds,preds.numpy()))
                golds      = np.concatenate((golds,batch_labels.numpy()))
                golds_gender = np.concatenate((golds_gender,batch_gender_labels.numpy()))
                if calculate_loss:
                    loss   = torch.nn.functional.cross_entropy(out,batch_labels).mean()
                    ovrl_loss += loss
                    step_num  += 1
                    


    return all_preds,golds,ovrl_loss/max(1,step_num),golds_gender


def train(args, corpus, lang):
    logger.info("****************************************  Running training {} ****************************".format(lang))
    corpus.cross_validation(lang,args.n_splits)
    for cv_ix in range(len(corpus.indices)):
        logger.info("**** Cross validation cv_ix:{} ****".format(cv_ix))
        trn_stats,tst_stats,trn_tp_stats,tst_tp_stats = corpus.make_data_ready(cv_ix,lang)
        logger.info("male/female ratio in train set:{}/{} = {:.2f}".format(trn_stats[0],trn_stats[1],trn_stats[0]/trn_stats[1]))
        logger.info("male/female ratio in test  set:{}/{} = {:.2f}".format(tst_stats[0],tst_stats[1],tst_stats[0]/tst_stats[1]))

        logger.info("pos/neg ratio in train set:{}/{} = {:.2f}".format(trn_tp_stats[0],trn_tp_stats[1],
                                                                       trn_tp_stats[0]/trn_tp_stats[1]))
        logger.info("pos/neg ratio in test  set:{}/{} = {:.2f}".format(tst_tp_stats[0],tst_tp_stats[1],
                                                                       tst_tp_stats[0]/tst_tp_stats[1]))
        
        if args.model_type == 'OLD_GenderPredictor_BERT_LSTM':
            model  = OLD_GenderPredictor_BERT_LSTM(
                args.embed_size,args.hidden_size,args.vocab_size,args.dropout,
                layer_num=args.layer_num).to(device)
        elif args.model_type == 'OLD_GenderPredictor_BERT_SUM':
            model = OLD_GenderPredictor_BERT_SUM(
                args.embed_size,args.hidden_size,args.vocab_size,args.dropout).to(device)
        elif args.model_type == 'GenderPredictor_BERT_SUM_MLP':
            model = GenderPredictor_BERT_SUM_MLP(
                args.embed_size,args.hidden_size,args.vocab_size,args.dropout,bert_model_name=args.bert_model_name).to(device)
        else:
            print("Unknown model type")
            model = None

        if args.do_not_train:
            load_fname = '../../models/{}_{}_{}'.format(lang,cv_ix,args.load_topic_model)
            model.load_state_dict(torch.load(load_fname))
            model.to(device)            
            model.eval()
            preds,golds,lss,golds_gender  = predict(args,model,corpus,'test',calculate_loss=True)
            result = metrics(preds,golds,target_names=['tech','not_th'])
            nice_print(result,cats=['tech','not_th'])
            pred_out_fname = '../../outputs/{}_{}_{}_best_model_outputs.csv'.format(lang,cv_ix,args.logfname)
            pd.DataFrame({'prediction':preds,  # Topic Prediction                              
                          'gender_gold':golds_gender, # Gender golds
                          'topic_gold':golds}).to_csv(pred_out_fname,index=False,sep=',') # topic gold labels
            continue
        
        if args.freeze_bert:
            for param in model.bert.parameters():
                param.requires_grad = False
                
       # model = torch.nn.DataParallel(model)
        
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
            ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=args.lr, eps=1e-8)
        #optimizer = AdamW(model.parameters(),
        #                  lr = args.lr, # args.learning_rate - default is 5e-5, our notebook had 2e-5
        #                  eps = 1e-8) # args.adam_epsilon  - default is 1e-8.

        steps_per_epoch = len(range(0,len(corpus.train),args.bs))
        total_steps = steps_per_epoch * args.epochs

        # Create the learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                    num_warmup_steps = 0, 
                                                    num_training_steps = total_steps)

        best_f1,trn_lss,step_num  = -1, 0.0, 0
        saved_any,saved_backup = False,False
        best_backup_f1 = -1

        
        for epoch in range(args.epochs):
            # Train
            model.train()
            model.zero_grad()
            trn_lss, trn_step_num = 0.0,0.0
            accumulation_steps = args.accumulation_steps
            for step,batch in enumerate(corpus.iter_batches(
                    batch_size=args.bs,split_name='train',shuffle=True)):
                
                batch_input_ids=[torch.tensor(
                    d_input_ids).to(device) for d_input_ids in batch[0]][0]
                batch_segment_ids = [torch.tensor(
                    d_segment_ids).to(device) for d_segment_ids in batch[1]][0]
                batch_attention_masks = [torch.tensor(
                    d_attn_masks).to(device) for d_attn_masks in batch[2]][0]
                
                #batch_labels = torch.tensor(batch[3]).to(device)
                batch_topic_labels = torch.tensor(batch[4]).to(device)
                logits,loss = model(input_ids=batch_input_ids,
                                              token_type_ids=batch_segment_ids,
                                              attention_mask=batch_attention_masks,
                                              labels = batch_topic_labels)

                loss = loss.mean()
                trn_lss += loss
                step_num += 1                
                loss.backward()
                if (step_num) % accumulation_steps == 0:
                    #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()

            # Evaluate
            preds,golds,lss,golds_gender  = predict(args,model,corpus,'test',calculate_loss=True)
            result = metrics(preds,golds,target_names=['tech','not_th'])
            nice_print(result,cats=['tech','not_th'])
            logger.info("Epoch:{}\tTrnLss:{:.2f}\tTstLss:{:.2f}\n".format(epoch+1,trn_lss/step_num,lss))

            avg_f1 = (float(result['tech']['f1-score'])+float(result['not_th']['f1-score']))/2
            if (float(result['tech']['accuracy']) > 0.4 and 
                float(result['not_th']['accuracy']) > 0.4 and avg_f1 > best_f1):
                saved_any = True
                best_f1 = avg_f1
                logger.info("Saving model with {} accuracy".format(float(result['accuracy'])))
                print("saving model...")
                model_save_name = '../../models/{}_{}_{}_best_model.pt'.format(lang,cv_ix,args.logfname)
                torch.save(model.state_dict(), model_save_name)
                # Print outputs:
                pred_out_fname = '../../outputs/{}_{}_{}_best_model_outputs.csv'.format(lang,cv_ix,args.logfname)
                pd.DataFrame({'prediction':preds,  # Topic Prediction                              
                              'gender_gold':golds_gender, # Gender golds
                              'topic_gold':golds}).to_csv(pred_out_fname,index=False,sep=',') # topic gold labels
                
            if avg_f1 > best_backup_f1 and saved_any == False:
                saved_backup = True
                best_backup_f1 = avg_f1
                backup_model_save_name = '../../models/{}_{}_{}_backup_best_model.pt'.format(lang,cv_ix,args.logfname)
                torch.save(model.state_dict(), backup_model_save_name)
                logger.info("Backup model saved with {} accuracy".format(float(result['accuracy'])))
                print("backup model saved..")
                # Print outputs:
                backup_pred_out_fname = '../../outputs/{}_{}_{}_backup_best_model_outputs.csv'.format(lang,cv_ix,args.logfname)
                pd.DataFrame({'prediction':preds,  # Topic Prediction                              
                              'gender_gold':golds_gender, # Gender golds
                              'topic_gold':golds}).to_csv(backup_pred_out_fname,index=False,sep=',') # topic gold labels
                


        if saved_any == False and saved_backup == True:
            model_save_name = '../../models/{}_{}_{}_best_model.pt'.format(lang,cv_ix,args.logfname)
            os.rename(backup_model_save_name,model_save_name)
            # Rename the prediction file too
            pred_out_fname = '../../outputs/{}_{}_{}_best_model_outputs.csv'.format(lang,cv_ix,args.logfname)
            os.rename(backup_pred_out_fname,pred_out_fname)
            logger.info("Backup model and prediction file were renamed as best_model")
        elif saved_any == True and saved_backup == True:
            os.remove(backup_model_save_name)
            os.remove(backup_pred_out_fname)
            logger.info("Backup model and prediction file were removed")
                
            #print("train loss:{:.2f} test loss:{:.2f}".format(trn_lss/step_num,lss))
            #if float(result['accuracy']) > best_acc:
             #   best_acc = float(result['accuracy'])
             #   logger.info("Saving model with {} accuracy".format(best_acc))
             #   print("saving model...")
              #  torch.save(model.state_dict(), args.model_save_dir)
                
        # Test model loading:
        #print("loading model...")
        #model = GenderPredictor(args.embed_size,args.hidden_size,args.vocab_size,args.dropout)
        #model.load_state_dict(torch.load(args.model_save_dir))
        #model.to(device)
        #preds,golds,lss = predict(args,model,corpus,split_name='test',calculate_loss=True)

    
def main():
    parser = argparse.ArgumentParser(description='Gender prediction on paralel multilingual data')
    parser.add_argument("--corpus_dir",default='../../data/tedtalks/',help='corpus directory')
    parser.add_argument("--annotations",default='../../data/TED_annotations.csv',help='annotation file')
    parser.add_argument("--langs",nargs='+',default=['tr','en'],help='languages to be considered')
    parser.add_argument("--target",default="translator",help='whose gender',choices=['translator','speaker','reviewer'])
    parser.add_argument("--is_gpu",type=str2bool, default=True,choices=['True (Default)', True, 'False', False])
    parser.add_argument("--n_splits",type=int,default=5,help='number of folds in training')
    parser.add_argument("--bs",type=int,default=1,help='batch size')
    parser.add_argument("--epochs",type=int,default=6,help='number of epochs')
    parser.add_argument("--embed_size",type=int,default=768,help='embedding size, must be equal to BERT hidden size')
    parser.add_argument("--dropout",type=float,default=0.5,help='dropout')
    parser.add_argument("--do_lowercase",type=str2bool,default=False,choices=['True', True, 'False', False])
    parser.add_argument("--freeze_bert",type=str2bool,default=False,choices=['True', True, 'False', False])
    parser.add_argument("--hidden_size",type=int,default=512,help='lstm hidden size')
    parser.add_argument("--layer_num",type=int,default=1,help='lstm layer number')
    parser.add_argument("--vocab_size",type=int,default=3000,help='vocabulary size')
    parser.add_argument("--lr",type=float,default=5e-5,help='learning rate')
    parser.add_argument("--accumulation_steps",type=int,default=48,help='gradient acc.step')
    parser.add_argument("--model_save_dir",default='../../models/best_model.pt',help='corpus directory')
    parser.add_argument("--bert_max_len", type=int,default=128,help='Max token number in bert input')
    parser.add_argument("--sector_max_word",type=int,default=60,help='Max numb. of tokens (w/o word-segment.) in a sector of doc')
    parser.add_argument("--model_type",default='GenderPredictor_BERT_SUM_MLP',help='which model to be used')
    parser.add_argument("--load_topic_model",default='GenderPredictor_BERT_SUM_MLP_2020-03-02_11-41-32_best_model.pt',
                        help='models used to get document encodings')
    parser.add_argument("--do_not_train",type=str2bool,default=False,choices=['True',True,'False',False])
    
    args = parser.parse_args()
    args.langs.sort()
    print(args)

    args.logfname = '{}_{:%Y-%m-%d_%H-%M-%S}'.format(args.model_type,datetime.now())
    fh = logging.FileHandler('./logs/topic_only_training_{}.log'.format(args.logfname))
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(args)
    logger.info("*****************************\n")
      
    stats = get_parallel_corpus_id_stats(args.corpus_dir)
    for k in {k: v for k, v in sorted(stats.items(), key=lambda item: item[1][0],reverse=True)}.keys():
        print(k,'\t',stats[k][0])



    for lang in args.langs:
        if lang == 'tr':
            bert_model_name = 'dbmdz/bert-base-turkish-cased'
        elif lang == 'de':
            bert_model_name = 'bert-base-german-cased'
        elif lang =='es':
            bert_model_name = 'dccuchile/bert-base-spanish-wwm-cased'
        elif lang == 'fr':
            bert_model_name = 'camembert-base'
        elif lang == 'en':
            bert_model_name = 'bert-base-cased'
        else:
            logger.info("lang code is not valid, model is set to bert-base-cased")
            bert_model_name = 'bert-base-cased'
        args.bert_model_name = bert_model_name
        logger.info("****************************************  Loading corpus {} ****************************".format(lang)) 
        corpus = load_corpus(args)
        train(args,corpus,lang)


    
if __name__ == "__main__":
    main()
