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
mkdir models/roberta_base_fairseq -p
wget https://github.com/sdadas/polish-roberta/releases/download/models/roberta_base_fairseq.zip
unzip roberta_base_fairseq.zip -d models/roberta_base_fairseq
rm roberta_base_fairseq.zip
```

## Training and evaluating
The code expects the data directory passed to contain 3 dataset splits: `train.txt`, `valid.txt` and `test.txt`.

# Named Entity Recognition on KPWr n82

## Train the model from scratch

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

## Download pre-trained model

```bash
mkdir models -d
wget https://clarin-pl.eu/dspace/bitstream/handle/11321/743/kpwr_n82_polish_roberta_base.zip
unzip kpwr_n82_polish_roberta_base.zip -d models
rm kpwr_n82_polish_roberta_base.zip
```

## Eval model

```bash
time python main.py  \
      --data_dir=data/kpwr_n82/  \
      --task_name=ner \
      --output_dir=models/kpwr_n82_polish_roberta_base/   \
      --max_seq_length=128   \
      --do_eval \
      --pretrained_path models/roberta_base_fairseq \
      --eval_on test \
      --train_batch_size 32
```

### Summary 

Results on the test part of the KPWr n82 corpus.

| Model                 | Precision |	Recall |	F1 |  	 Time |	Memory usage | GPU memory | Embeddings size |
|-----------------------|----------:|---------:|------:|---------:|----------:|----------:|----------:|
| Polish RoBERTa large                                                   | 76.10 | 78.72 | 77.39 | ~ 0.9 m | 3.0 GB |  3.8 GB | 0.71 GB + 1.40 GB |
| Polish RoBERTa base                                                    | 74.79 | 76.42 | 75.60 | ~ 0.5 m | 3.0 GB |  2.0 GB | 0.25 GB + 0.50 GB |
| [PolDeepNer](https://github.com/CLARIN-PL/PolDeepNer) (n82-elmo-kgr10) | 73.97 | 75.49 | 74.72 | ~ 4.0 m | 4.5 GB |         | 0.4 GB |


### Detailed results for Polish RoBERTa base on KPWr n82 test

```bash
                           precision    recall  f1-score   support

nam_adj                       0.4833    0.5577    0.5179        52
nam_adj_city                  0.8409    0.8810    0.8605        42
nam_adj_country               0.6859    0.7892    0.7339       166
nam_adj_person                1.0000    0.3333    0.5000        18
nam_eve                       1.0000    0.8750    0.9333         8
nam_eve_human                 0.3472    0.3205    0.3333        78
nam_eve_human_cultural        0.2609    0.2727    0.2667        22
nam_eve_human_holiday         0.5714    0.4444    0.5000         9
nam_eve_human_sport           0.6712    0.8909    0.7656        55
nam_fac_bridge                0.5000    0.5000    0.5000         4
nam_fac_goe                   0.5179    0.4531    0.4833        64
nam_fac_goe_stop              0.0000    0.0000    0.0000         4
nam_fac_park                  0.8571    0.6000    0.7059        10
nam_fac_road                  0.7788    0.8526    0.8141        95
nam_fac_square                0.6667    0.3333    0.4444         6
nam_fac_system                0.6111    0.4231    0.5000        26
nam_liv_animal                0.0000    0.0000    0.0000        11
nam_liv_god                   0.9412    0.9143    0.9275        35
nam_liv_habitant              0.5000    0.2857    0.3636         7
nam_liv_person                0.8877    0.9319    0.9093       925
nam_loc                       0.0000    0.0000    0.0000         4
nam_loc_country_region        0.2000    0.5000    0.2857         4
nam_loc_gpe_admin1            0.8814    0.8125    0.8455        64
nam_loc_gpe_admin2            0.8286    0.8056    0.8169        36
nam_loc_gpe_admin3            0.8571    0.7660    0.8090        47
nam_loc_gpe_city              0.8069    0.8604    0.8328       437
nam_loc_gpe_country           0.9103    0.9384    0.9241       357
nam_loc_gpe_district          0.3077    0.2222    0.2581        18
nam_loc_gpe_subdivision       0.5714    0.4615    0.5106        26
nam_loc_historical_region     0.6190    0.5909    0.6047        22
nam_loc_hydronym              0.0000    0.0000    0.0000         1
nam_loc_hydronym_lake         1.0000    0.5000    0.6667         2
nam_loc_hydronym_ocean        0.5000    1.0000    0.6667         1
nam_loc_hydronym_river        0.9048    0.7451    0.8172        51
nam_loc_hydronym_sea          1.0000    0.6667    0.8000         3
nam_loc_land                  0.0000    0.0000    0.0000         2
nam_loc_land_continent        0.9667    0.9062    0.9355        32
nam_loc_land_island           0.8000    0.7273    0.7619        11
nam_loc_land_mountain         0.7143    0.5556    0.6250         9
nam_loc_land_region           0.5333    0.7273    0.6154        11
nam_num_house                 0.8462    1.0000    0.9167        11
nam_num_phone                 0.0000    0.0000    0.0000         2
nam_org_company               0.6923    0.5921    0.6383        76
nam_org_group                 0.2500    0.1667    0.2000        18
nam_org_group_band            0.5500    0.5789    0.5641        19
nam_org_group_team            0.8627    0.8919    0.8771       148
nam_org_institution           0.6622    0.7368    0.6975       266
nam_org_nation                0.8333    0.7407    0.7843        81
nam_org_organization          0.6939    0.6911    0.6925       246
nam_org_organization_sub      0.0000    0.0000    0.0000         3
nam_org_political_party       0.8615    0.9655    0.9106        58
nam_oth                       0.3929    0.5000    0.4400        22
nam_oth_currency              0.9388    0.9020    0.9200        51
nam_oth_data_format           0.0000    0.0000    0.0000        10
nam_oth_license               0.4118    0.6364    0.5000        11
nam_oth_position              0.4167    0.5000    0.4545        10
nam_oth_tech                  0.6981    0.6066    0.6491        61
nam_oth_www                   0.5000    0.1000    0.1667        20
nam_pro                       0.0000    0.0000    0.0000         2
nam_pro_award                 0.6316    0.5217    0.5714        23
nam_pro_brand                 0.5000    0.5000    0.5000        46
nam_pro_media                 0.2857    0.2500    0.2667         8
nam_pro_media_periodic        0.7857    0.8049    0.7952        82
nam_pro_media_radio           0.4000    0.6667    0.5000         3
nam_pro_media_tv              0.5000    0.8571    0.6316         7
nam_pro_media_web             0.3538    0.5750    0.4381        40
nam_pro_model_car             0.7778    0.8077    0.7925        26
nam_pro_software              0.7013    0.5567    0.6207        97
nam_pro_software_game         0.3333    0.3333    0.3333         3
nam_pro_title                 0.4146    0.4857    0.4474        35
nam_pro_title_album           0.5000    0.5714    0.5333         7
nam_pro_title_book            0.6667    0.5455    0.6000        11
nam_pro_title_document        0.5444    0.5904    0.5665        83
nam_pro_title_song            0.2000    0.2857    0.2353         7
nam_pro_title_treaty          0.0000    0.0000    0.0000         2
nam_pro_title_tv              0.4500    0.3750    0.4091        24
nam_pro_vehicle               0.0000    0.0000    0.0000         4

micro avg                     0.7479    0.7642    0.7560      4398
macro avg                     0.7491    0.7642    0.7529      4398
```

## Sample usage

Command:

```bash
time python sample.py
```

Expected output:

```bash
--------------------
Ala          B-nam_liv_person
z            O
Krakowa      B-nam_loc_gpe_city
jeździ       O
Audi         B-nam_pro_brand
--------------------
Marek        B-nam_liv_person
Nowak        I-nam_liv_person
z            O
Politechniki B-nam_org_organization
Wrocławskiej I-nam_org_organization
mieszka      O
przy         O
ul           O
.            O
Sądeckiej    B-nam_fac_road
--------------------

real	0m6.309s
user	0m9.360s
sys 	0m3.136s
```