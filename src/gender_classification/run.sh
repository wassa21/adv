CUDA_VISIBLE_DEVICES=0,1 python gender_prediction.py --target speaker --lr 5e-5  --model_type GenderPredictor_BERT_SUM_MLP --langs de tr es fr --sector_max_word 60
