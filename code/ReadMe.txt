* Command to train the model:

CUDA_VISIBLE_DEVICES=0 python main.py --batch_size 30 --dropout 0.3 --model_type BERT --debug 0 --seed 2333 --percent 1.0 --training 1 --nargs [unused4] [unused5] [unused6]  --task_name task_name --bert_path /home/chenzhen/work/BERT 

* Command to test the model:

CUDA_VISIBLE_DEVICES=0 python main.py --batch_size 30 --model_type BERT --debug 0 --seed 2333 --percent 1.0 --training 0 --nargs [unused4] [unused5] [unused6] --data_name original -model_name task_name --bert_path /home/data/bert-base-uncased 


* Command to train the model:

CUDA_VISIBLE_DEVICES=0 python main.py --batch_size 30 --dropout 0.3 --model_type BERT --debug 0 --seed 2333 --percent 1.0 --training 1 --nargs [unused4] [unused5] [unused6] --data_name original --task_name task_name --bert_path /home/data/bert-base-uncased 

* Command to test the model:

CUDA_VISIBLE_DEVICES=0 python main.py --batch_size 30 --model_type BERT --debug 0 --seed 2333 --percent 1.0 --training 0 --nargs [unused4] [unused5] [unused6] --data_name original -model_name task_name --bert_path /home/data/bert-base-uncased 