import sys
import pandas as pd
from sklearn.metrics import classification_report, f1_score

def split_measure(pred,gold,gold2):
    results = [] 
    for ix in [0,1]:
        indices = [i for i, x in enumerate(gold2) if x == ix]
        sub_gold = [gold[i] for i in indices]
        sub_pred = [pred[i] for i in indices]
        #f1_scores = f1_score(sub_gold, sub_pred, average=None)
        #f1_micro = f1_score(sub_gold, sub_pred, average='micro')
        f1_weighted = f1_score(sub_gold, sub_pred, average='weighted')
        results.append(f1_weighted)
    results.append(f1_score(gold, pred, average='weighted'))        
    print('{:.2f}\t{:.2f}\t{:.2f}'.format(results[0],results[1],results[2]))
        #print('{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}'.format(f1_scores[0],f1_scores[1],f1_macro,f1_micro,acc))        
        #print(classification_report(sub_gold,sub_pred))
        
def measure(pred,gold):
    f1_scores = f1_score(gold,pred, average=None)
    f1_micro = f1_score(gold, pred, average='micro')
    f1_weighted = f1_score(gold, pred, average='weighted')
    result = [gold[i] == pred[i] for i,x in enumerate(gold)]
    acc = 100*sum(result)/len(result)
    print('{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}'.format(f1_scores[0],f1_scores[1],f1_weighted,acc))

def accuracy(pred,gold):
    result = [gold[i] == pred[i] for i,x in enumerate(gold)]
    print("{:.2f}".format(100*sum(result)/len(result)))

def main():
    langs  = ['de','es','fr','tr']
    _fname = sys.argv[1]   
    task   = sys.argv[2].lower() # topic or gender

    g2ix  = {'male':0,'female':1}
    t2ix  = {'other':0,'tech_science':1}
    for i in range(5):
        print("-------- C_IX:{}----------".format(i))
        print("CLS0\tCLS1\tWGHT\tACC")
        for lang in langs:
            fname = '../outputs/{}_{}_{}_best_model_outputs.csv'.format(lang,i,_fname)
            df = pd.read_table(fname,sep=',')
            gender_gold = df['gender_gold'].tolist()
            topic_gold  = df['topic_gold'].tolist()
            if task == 'topic':
                #topic_pred = df['prediction'].tolist()
                count_0 = sum([x == 0.0 for x in topic_gold])
                count_1 = sum([x == 1.0 for x in topic_gold])
                if count_0 > count_1:
                    topic_pred = [0.0] * len(topic_gold)
                else:
                    topic_pred = [0.1] * len(topic_gold)                        
                gender_pred = None
            else:
                count_0 = sum([x == 0.0 for x in gender_gold])
                count_1 = sum([x == 1.0 for x in gender_gold])
                if count_0 > count_1:
                    gender_pred = [0.0] * len(gender_gold)
                else:
                    gender_pred = [0.1] * len(gender_gold)                                            
                topic_pred = None                    

            if task == 'topic':
                split_measure(topic_pred,topic_gold,gender_gold)
            else:
                split_measure(gender_pred,gender_gold,topic_gold)


main()
