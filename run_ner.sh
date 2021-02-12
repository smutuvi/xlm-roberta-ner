OUTPUT_DIR=models/epidemic/
PARAM_SET=base
DATA_DIR=data/epidemic/
python3 main.py \
    --data_dir=${DATA_DIR}  \
    --task_name=ner   \
    --output_dir=${OUTPUT_DIR}   \
    --max_seq_length=256   \
    --num_train_epochs 30 \
    --do_eval \
    --warmup_proportion=0.0 \
    --pretrained_path pretrained_models/xlmr.$PARAM_SET/ \
    --learning_rate 6e-5 \
    --do_train \
    --eval_on test \
    --prediction=models/epidemic/eval_predictions.txt \
    --train_batch_size 16 \