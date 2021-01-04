# Lessons_for_adversarial_debiasing
Source code for WASSA21 submission

# Installation
```shell
$ git clone https://github.com/wassa21/adv.git
$ cd adv
$ pip install -r requirements.txt
```

# Data

Please see `./data` for information about the dataset

# Experiments

## Table 3: F1 scores for topic classification
```shell
$ cd src/topic_classification
$ bash run.sh
```


## Table 4: F1 scores for gender classification
```shell
$ cd src/gender_classification
$ bash run.sh
```


## Figure 3 (Right): F1 scores for topic classification with adversarial author gender training
```shell
$ cd src/topic_classification_with_adv_gender
$ bash run.sh
```


## Figure 4 (Right): F1 scores for author gender classification with adversarial topic training
```shell
$ cd src/gender_classification_with_adv_topic
$ bash run.sh
```


# Evaluation
- Each `run.sh` script above will save the model with best weighted F-Score to `lessons_for_adversarial_debiasing/models`
and save predictions on test set to `lessons_for_adversarial_debiasing/outputs`. 
- By default, prediction file names generated by the following template:

    `{LANG}_{ix}_BERT_SUM_MLP_{DATE}_best_model_outputs.csv` 
   
   `LANG`:  'de','es','fr' or 'tr';
   
   `ix`: 0,1,2,3,4 representing one of the five randomly generated test sets. 
   
   `DATE`: the system date and time when the scripts was runned. 

- In order to obtain weighted F-Score evaluation on these generation files one can use the `src/evaluate.py`. It expects 3 arguments:
    
    argv\[1]: ` path of prediction file` (Example: de_1_BERT_SUM_MLP_2020-05-25_21-45-03_best_model_outputs.csv)
    
    argv\[2]: `task type` (either ` gender` or `topic`)
    
    argv\[3]: `is_adv` (either `true` or `false`)

    For example, to evaluate the predictions of a gender classifier (Table 4) one can use the following command:
``` python
$ python evaluate.py GenderPredictor_BERT_SUM_MLP_2020-05-25_21-45-03 gender false
``` 
This command will evaluate gender classifiers trained on DE,ES,FR,TR transcripts of TED  talks.  

- Use `src/evaluate_mb.py` to evaluate majority baseline. (With same command line arguments)


