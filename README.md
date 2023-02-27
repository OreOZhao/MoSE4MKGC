# MoSE4MKGC
Code for paper "MoSE: Modality Split and Ensemble for Multimodal Knowledge Graph Completion", EMNLP, 2022.

- [UPDATE 2023.2.27] We update the feature extractor code `src/img_encoder.py` and `src/text_encoder.py` for the pickle files.


## Dataset Download

You can download dataset FB15K-237, WN18 from [kbc](https://github.com/facebookresearch/kbc), and WN9 from [IKRL](https://github.com/thunlp/IKRL).

The descriptions of entities can be downloaded from [KG-BERT](https://github.com/yao8839836/kg-bert). 

The images of entities can be downloaded from [MMKB](https://github.com/mniepert/mmkb), [RSME](https://github.com/wangmengsd/RSME), or [MKGformer](https://github.com/zjunlp/MKGformer).
    
## Dataset Preprocess

For triples data, the data could be preprocessed by `src/process_datasets.py`.

For text data, we used bert-base-uncased model from [transformers](https://github.com/huggingface/transformers) as encoder.
We use the mean aggregation of token hidden states as text feature of entity.
The text encoder is in `src/text_encoder.py`.

For image data, we used pytorch_pretrained_vit model as encoder, which is the same as [RSME](https://github.com/wangmengsd/RSME).
The image encoder is in `src/img_encoder.py`.

For the missing descriptions or images, we impute the feature with `np.random.normal` vector.

After extracting features from fixed encoders, we save the text and image features of entities in a pickle file and save the file in `data/DATASET_NAME/`.

The expected project structure is:
```
MoSE4MKGC
 |-- src
 |-- src_data           # downloaded triples of dataset
 |    |-- DATASET_NAME
 |    |    |-- train
 |    |    |-- valid
 |    |    |-- test
 |    |    |-- ...      # other files
 |-- data
 |    |-- DATASET_NAME
 |    |    |-- ...      # files from process_datasets.py
 |    |    |-- YOUR_TEXT_FEATURE.pickle
 |    |    |-- YOUR_IMG_FEATURE.pickle
```

## How to run
After preprocessing, you could run `src/learner.py` to train the modality split KGs. 


```bash
cd src
CUDA_VISIBLE_DEVICES='1' python learn.py --model ComplExMDR --ckpt_dir YOUR_CKPT_DIR --dataset WN9 --early_stopping 10 --fusion_dscp True --fusion_img True --modality_split True --img_info PATH_TO_YOUR_IMG_FEATURE.pickle --dscp_info PATH_TO_YOUR_TEXT_FEATURE.pickle
```

**Note that the displayed metrics during training are more like an upper bound of modality split KGs performance, which is not the final prediction performance.**

Then you can run `src/boosting_inference.py` or `src/meta_learner.py` to ensemble the modality split predictions to get final predictions and performance.

MoSE-AI
```bash
cd src
python boosting_inference.py --model_path YOUR_MODEL_PATH --dataset DATASET_NAME --boosting False 
```
MoSE-BI
```bash
cd src
python boosting_inference.py --model_path YOUR_MODEL_PATH --dataset DATASET_NAME --boosting True 
```
MoSE-MI
```bash
cd src
python meta_learner.py --model_path YOUR_MODEL_PATH --dataset DATASET_NAME
```



## Acknowledgement
Thanks to the support of [RSME](https://github.com/wangmengsd/RSME) for the image acquisition and code open source.

## Citation
Please cite the following paper as reference if you find our work useful.

```bibtex

@inproceedings{zhao-etal-2022-mose,
    title = "{M}o{SE}: Modality Split and Ensemble for Multimodal Knowledge Graph Completion",
    author = "Zhao, Yu  and
      Cai, Xiangrui  and
      Wu, Yike  and
      Zhang, Haiwei  and
      Zhang, Ying  and
      Zhao, Guoqing  and
      Jiang, Ning",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-main.719",
    pages = "10527--10536"
}
```