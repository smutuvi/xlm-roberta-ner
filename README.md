# RoBERTa Named Entity Recognition

This code is based on [xlm-roberta-ner](https://github.com/mohammadKhalifa/xlm-roberta-ner) by mohammadKhalifa. 

## Requirements 
* `python 3.6+`
* `torch 1.x`
* `fairseq`
* `pytorch_transformers` (for AdamW and WarmpUpScheduler)


## Setting up

Download the Polish RoBERTa base model.

```bash
wget https://github.com/sdadas/polish-roberta/releases/download/models/roberta_base_fairseq.zip
unzip roberta_base_fairseq.zip
```


## Training and evaluating
The code expects the data directory passed to contain 3 dataset splits: `train.txt`, `valid.txt` and `test.txt`.

Training arguments : 

```
 -h, --help            show this help message and exit
  --data_dir DATA_DIR   The input data dir. Should contain the .tsv files (or
                        other data files) for the task.
  --pretrained_path PRETRAINED_PATH
                        pretrained XLM-Roberta model path
  --task_name TASK_NAME
                        The name of the task to train.
  --output_dir OUTPUT_DIR
                        The output directory where the model predictions and
                        checkpoints will be written.
  --max_seq_length MAX_SEQ_LENGTH
                        The maximum total input sequence length after
                        WordPiece tokenization. Sequences longer than this
                        will be truncated, and sequences shorter than this
                        will be padded.
  --do_train            Whether to run training.
  --do_eval             Whether to run eval or not.
  --eval_on EVAL_ON     Whether to run eval on the dev set or test set.
  --do_lower_case       Set this flag if you are using an uncased model.
  --train_batch_size TRAIN_BATCH_SIZE
                        Total batch size for training.
  --eval_batch_size EVAL_BATCH_SIZE
                        Total batch size for eval.
  --learning_rate LEARNING_RATE
                        The initial learning rate for Adam.
  --num_train_epochs NUM_TRAIN_EPOCHS
                        Total number of training epochs to perform.
  --warmup_proportion WARMUP_PROPORTION
                        Proportion of training to perform linear learning rate
                        warmup for. E.g., 0.1 = 10% of training.
  --weight_decay WEIGHT_DECAY
                        Weight deay if we apply some.
  --adam_epsilon ADAM_EPSILON
                        Epsilon for Adam optimizer.
  --max_grad_norm MAX_GRAD_NORM
                        Max gradient norm.
  --no_cuda             Whether not to use CUDA when available
  --seed SEED           random seed for initialization
  --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
                        Number of updates steps to accumulate before
                        performing a backward/update pass.
  --fp16                Whether to use 16-bit float precision instead of
                        32-bit
  --fp16_opt_level FP16_OPT_LEVEL
                        For fp16: Apex AMP optimization level selected in
                        ['O0', 'O1', 'O2', and 'O3'].See details at
                        https://nvidia.github.io/apex/amp.html
  --loss_scale LOSS_SCALE
                        Loss scaling to improve fp16 numeric stability. Only
                        used when fp16 set to True. 0 (default value): dynamic
                        loss scaling. Positive power of 2: static loss scaling
                        value.
  --dropout DROPOUT     training dropout probability
  --freeze_model        whether to freeze the XLM-R base model and train only
                        the classification heads
```


## Tests on KPWr n82

The following commands and parameters were used to train and test a model fine-grained named entity recognition 
on the KPWr corpus.

### Base model

```bash
time python main.py  \
      --data_dir=data/kpwr_n82/  \
      --task_name=ner \
      --output_dir=models/kpwr_n82_base/   \
      --max_seq_length=128   \
      --num_train_epochs 50  \
      --do_eval \
      --warmup_proportion=0.0 \
      --pretrained_path roberta_base_fairseq \
      --learning_rate 6e-5 \
      --gradient_accumulation_steps 4 \
      --do_train \
      --eval_on test \
      --train_batch_size 32 \
      --dropout 0.2
```
```bash
Time: 113m29.552s
```

### Large model

```bash
time python main.py  \
      --data_dir=data/kpwr_n82/  \
      --task_name=ner \
      --output_dir=models/kpwr_n82_large/   \
      --max_seq_length=128   \
      --num_train_epochs 50  \
      --do_eval \
      --warmup_proportion=0.0 \
      --pretrained_path roberta_large_fairseq \
      --learning_rate 6e-5 \
      --gradient_accumulation_steps 4 \
      --do_train \
      --eval_on test \
      --train_batch_size 32 \
      --dropout 0.2
```
```bash
Time: 260m32.544s
```


### Summary 

Results on the test part of the KPWr n82 corpus.

| Model                 | Precision |	Recall |	F1 |  	 Time |	Memory usage | GPU memory | Embeddings size |
|-----------------------|----------:|---------:|------:|---------:|----------:|----------:|----------:|
| Polish RoBERTa large                                                   | 76.10 | 78.72 | 77.39 | ~ 0.9 m | 3.0 GB |  3.8 GB | 0.71 GB + 1.40 GB |
| Polish RoBERTa base                                                    | 74.37 | 76.72 | 75.52 | ~ 0.5 m | 3.0 GB |  2.0 GB | 0.25 GB + 0.50 GB |
| [PolDeepNer](https://github.com/CLARIN-PL/PolDeepNer) (n82-elmo-kgr10) | 73.97 | 75.49 | 74.72 | ~ 4.0 m | 4.5 GB |         | 0.4 GB |


### Detailed results for Polish RoBERTa large on KPWr n82 test

```bash
                           precision    recall  f1-score   support

      nam_loc_gpe_country     0.9254    0.9384    0.9318       357
            nam_eve_human     0.4000    0.4359    0.4172        78
  nam_org_political_party     0.8636    0.9828    0.9194        58
         nam_loc_gpe_city     0.8284    0.8947    0.8603       437
      nam_pro_title_album     0.5000    0.7143    0.5882         7
           nam_liv_person     0.9035    0.9416    0.9222       925
          nam_adj_country     0.6935    0.7771    0.7330       166
      nam_org_institution     0.6884    0.7143    0.7011       266
             nam_oth_tech     0.6607    0.6066    0.6325        61
       nam_pro_title_song     0.5000    0.5714    0.5333         7
     nam_org_organization     0.7425    0.7033    0.7223       246
              nam_liv_god     0.8857    0.8857    0.8857        35
nam_loc_historical_region     0.5417    0.5909    0.5652        22
           nam_org_nation     0.8571    0.7407    0.7947        81
            nam_pro_brand     0.5106    0.5217    0.5161        46
   nam_loc_hydronym_river     0.8913    0.8039    0.8454        51
          nam_org_company     0.6216    0.6053    0.6133        76
      nam_loc_land_region     0.6000    0.5455    0.5714        11
            nam_num_house     0.9167    1.0000    0.9565        11
             nam_fac_road     0.8317    0.8842    0.8571        95
           nam_fac_system     0.7500    0.6923    0.7200        26
     nam_loc_gpe_district     0.8889    0.4444    0.5926        18
   nam_pro_media_periodic     0.8205    0.7805    0.8000        82
        nam_pro_media_web     0.3731    0.6250    0.4673        40
       nam_loc_gpe_admin3     0.8409    0.7872    0.8132        47
      nam_eve_human_sport     0.6562    0.7636    0.7059        55
       nam_org_group_team     0.9189    0.9189    0.9189       148
              nam_fac_goe     0.5303    0.5469    0.5385        64
   nam_loc_land_continent     0.9688    0.9688    0.9688        32
                  nam_adj     0.5179    0.5577    0.5370        52
   nam_pro_title_document     0.4902    0.6024    0.5405        83
            nam_pro_title     0.4762    0.5714    0.5195        35
       nam_loc_gpe_admin1     0.7432    0.8594    0.7971        64
           nam_fac_square     0.7500    0.5000    0.6000         6
         nam_pro_media_tv     0.5455    0.8571    0.6667         7
         nam_pro_software     0.6569    0.6907    0.6734        97
    nam_pro_software_game     0.5000    0.3333    0.4000         3
            nam_org_group     0.3333    0.1667    0.2222        18
    nam_loc_land_mountain     1.0000    0.5556    0.7143         9
           nam_liv_animal     0.5000    0.1818    0.2667        11
                  nam_oth     0.2564    0.4545    0.3279        22
           nam_adj_person     1.0000    0.6111    0.7586        18
         nam_oth_currency     0.9600    0.9412    0.9505        51
        nam_pro_model_car     0.8519    0.8846    0.8679        26
       nam_pro_title_book     0.3158    0.5455    0.4000        11
            nam_pro_award     0.8000    0.6957    0.7442        23
   nam_eve_human_cultural     0.3158    0.2727    0.2927        22
              nam_oth_www     0.6667    0.1000    0.1739        20
             nam_adj_city     0.8049    0.7857    0.7952        42
      nam_oth_data_format     0.6000    0.3000    0.4000        10
      nam_loc_land_island     0.7500    0.8182    0.7826        11
         nam_pro_title_tv     0.7143    0.4167    0.5263        24
       nam_loc_gpe_admin2     0.8485    0.7778    0.8116        36
     nam_pro_title_treaty     0.1667    0.5000    0.2500         2
  nam_loc_gpe_subdivision     0.7368    0.5385    0.6222        26
    nam_eve_human_holiday     0.5714    0.4444    0.5000         9
            nam_pro_media     0.7500    0.3750    0.5000         8
          nam_oth_license     0.5714    0.7273    0.6400        11
           nam_fac_bridge     0.2857    0.5000    0.3636         4
                  nam_eve     1.0000    0.7500    0.8571         8
       nam_org_group_band     0.6471    0.5789    0.6111        19
             nam_loc_land     0.0000    0.0000    0.0000         2
             nam_fac_park     0.7500    0.6000    0.6667        10
         nam_oth_position     0.3529    0.6000    0.4444        10
 nam_org_organization_sub     0.5000    0.3333    0.4000         3
   nam_loc_country_region     0.1429    0.7500    0.2400         4
                  nam_loc     0.0000    0.0000    0.0000         4
     nam_loc_hydronym_sea     0.5000    0.6667    0.5714         3
                  nam_pro     0.0000    0.0000    0.0000         2
          nam_pro_vehicle     0.0000    0.0000    0.0000         4
         nam_liv_habitant     0.5000    0.5714    0.5333         7
    nam_loc_hydronym_lake     1.0000    1.0000    1.0000         2
      nam_pro_media_radio     0.6000    1.0000    0.7500         3
         nam_fac_goe_stop     0.2000    0.2500    0.2222         4
   nam_loc_hydronym_ocean     1.0000    1.0000    1.0000         1
            nam_num_phone     0.0000    0.0000    0.0000         2
         nam_loc_hydronym     0.0000    0.0000    0.0000         1

                micro avg     0.7610    0.7872    0.7739      4398
                macro avg     0.7723    0.7872    0.7753      4398
```