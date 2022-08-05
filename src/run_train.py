
import os
# cd drive/MyDrive/RE_model/PreSumm/src
# pip install pytorch_transformers tensorboardX

# train-ext
# os.system("python train.py -task ext -mode train -bert_data_path ../bert_data/bert_data_cnndm_final/cnndm -ext_dropout 0.1 -model_path ../models/ -lr 2e-3 -visible_gpus 0 -report_every 50 -save_checkpoint_steps 1000 -batch_size 3000 -train_steps 50000 -accum_count 2 -log_file ../logs/ext_bert_cnndm -use_interval true -warmup_steps 10000 -max_pos 512")
# train-abs
# os.system("python train.py  -task abs -mode train -bert_data_path ../bert_data/bert_data_cnndm_final/cnndm -dec_dropout 0.2  -model_path ../models/ -sep_optim true -lr_bert 0.002 -lr_dec 0.2 -save_checkpoint_steps 2000 -batch_size 140 -train_steps 200000 -report_every 50 -accum_count 5 -use_bert_emb true -use_interval true -warmup_steps_bert 20000 -warmup_steps_dec 10000 -max_pos 512 -visible_gpus 0  -log_file ../logs/abs_bert_cnndm")

# dev/test-ext
# os.system("python train.py -task ext -mode test -batch_size 3000 -test_batch_size 500 -bert_data_path ../bert_data/bert_data_cnndm_final/cnndm -log_file ../logs/ext_bert_cnndm -model_path ../models/ -sep_optim true -use_interval true -visible_gpus 1 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 50 -result_path ../logs/ext_bert_cnndm -test_all")
# dev/test-abs
os.system("python train.py  -task abs -mode test -bert_data_path ../bert_data/bert_data_cnndm_final/cnndm -dec_dropout 0.2  -model_path ../models/ -sep_optim true -lr_bert 0.002 -lr_dec 0.2 -save_checkpoint_steps 2000 -batch_size 140 -train_steps 200000 -report_every 50 -alpha 0.95 -accum_count 5 -min_length 50 -use_bert_emb true -use_interval true -result_path ../logs/abs_bert_cnndm -max_pos 512 -visible_gpus 0  -log_file ../logs/abs_bert_cnndm")
