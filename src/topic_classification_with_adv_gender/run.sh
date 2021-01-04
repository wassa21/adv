CUDA_VISIBLE_DEVICES=0,1 python topic_classification_with_adv_gender.py --target speaker --lr 5e-5  --model_type GenderPredictor_BERT_SUM_MLP --langs de tr es fr --sector_max_word 60 --alpha 1.0
