# Deep Time-Aware Item Evolution Network for Click-Through Rate Prediction

CIKM 2020 Anonymous Submission #563.

## Introduction

Pipeline:

1. Prepare Data
2. Train Model

##  Running

We test our code on Python 2.7 and Tensorflow 1.4.


### 1. Prepare Data
    mkdir -p dataset/Amazon_Clothing_Shoes_and_Jewelry/
    cd dataset/Amazon_Clothing_Shoes_and_Jewelry/
    wget -c http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Clothing_Shoes_and_Jewelry_5.json.gz
    gzip -d reviews_Clothing_Shoes_and_Jewelry_5.json.gz
    wget -c http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Clothing_Shoes_and_Jewelry.json.gz
    gzip -d meta_Clothing_Shoes_and_Jewelry.json.gz
    
    cd ../..
    python script/TIEN/prepare_data/process_data_user_sort_by_time.py
    python script/TIEN/prepare_data/local_aggretor_by_time.py
    python script/TIEN/prepare_data/generate_voc.py

When you see the files below, you can do the next work.


    item-info
    reviews-info
    jointed-new-by-time
    local_all_sample_by_time
    local_train_by_time
    local_test_by_time
    uid_voc.pkl
    mid_voc.pkl
    cat_voc.pkl

### 1. Train

We introduce a novel TIEN method in our paper (Fig. 1). We have implemented multiple CTR prediction methods in our code (option `--model`).

    python script/TIEN/train.py --model SVDPP[DNN,PNN,GRU4REC,ATRANK,CASER,UBGRUA,DIEN] --dataset Amazon_Clothing_Shoes_and_Jewelry

For dual behavior model, the truncation length of item behaviors can be changed (option `--iblen`).

    python script/TIEN/train.py --model TIEN[TopoLSTM,DIB,IBGRUA] --dataset Amazon_Clothing_Shoes_and_Jewelry --iblen 5[10,20,30,40,50]

We also design ablation experiments to study how each component in TIEN contributes to the final performance.

    python script/TIEN/train.py --model TIEN_sumagg[TIEN_timeatt,TIEN_robust,TIEN_timeaware] --dataset Amazon_Clothing_Shoes_and_Jewelry

To verify the utility of evolutionary item dynamics proposed by TIEN, we select several models using user behaviors as base models, including GRU4Rc, ATRANK, CASER, and DIEN.

    python script/TIEN/train.py --model GRU4REC_TIEN[ATRANK_TIEN,ATRANK_TIEN,DIEN_TIEN] --dataset Amazon_Clothing_Shoes_and_Jewelry

Finally, we study the parameter sensitivity of TIEN (option `--hidden_units`, `-embedding`).

    python script/TIEN/train.py --model TIEN --dataset Amazon_Clothing_Shoes_and_Jewelry --hidden_units 1024,512,256,128,1
    python script/TIEN/train.py --model TIEN --dataset Amazon_Clothing_Shoes_and_Jewelry --embedding 128
    
    
## Acknowledgement

We build our code based on [DIEN](https://github.com/mouna99/dien). We'd like to thank their contribution to the research on the CTR prediction task.