## Quick Start:

Prepare dataset:
```bash
cd dataset
python prepare_dataset.py --annotations original/annotations/instance.json --labeled_dir original/images/ --unlabeled_dir original/normal/B/ --output_dir leddd --train_ratio 0.8
```

Train EfficientAD on custom dataset:
```bash
python efficientad.py --dataset custom --custom_dataset_path dataset/leddd --output_dir output/1 --model_size small --map_format jpg --train_steps 1000
```

## Expected Output:

```bash
(effad) ee303@ee303-Z790-AORUS-ELITE:~/Documents/agbld/GitHub/EfficientAD$ cd dataset/

(effad) ee303@ee303-Z790-AORUS-ELITE:~/Documents/agbld/GitHub/EfficientAD/dataset$ python prepare_dataset.py --annotations original/annotations/instance.json --labeled_dir original/images/ --unlabeled_dir original/normal/B/ --output_dir leddd --train_ratio 0.8
Namespace(annotations='original/annotations/instance.json', labeled_dir='original/images/', output_dir='leddd', train_ratio=0.8, unlabeled_dir='original/normal/B/')
prepare_dataset.py:91: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  normal_defect_img_df['full_image_path'] = normal_defect_img_df['image_path'].apply(lambda x: defect_img_dir / x)
prepare_dataset.py:150: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  defect_img_df['full_image_path'] = defect_img_df['image_path'].apply(lambda x: defect_img_dir / x)
Number of images in leddd/train/good: 773
Number of images in leddd/test/good: 194
Number of images in leddd/test/defect_type_2: 112
Number of images in leddd/test/defect_type_4: 134
Number of images in leddd/test/defect_type_3: 87
Number of images in leddd/test/defect_type_7: 10
Number of images in leddd/test/defect_type_6: 6

(effad) ee303@ee303-Z790-AORUS-ELITE:~/Documents/agbld/GitHub/EfficientAD/dataset$ cd ..

(effad) ee303@ee303-Z790-AORUS-ELITE:~/Documents/agbld/GitHub/EfficientAD$ python efficientad.py --dataset custom --custom_dataset_path dataset/leddd --output_dir output/1 --model_size small --train_steps 1000
Computing mean of features: 100%|████████████████████████████████████████████████████████████| 695/695 [00:01<00:00, 420.63it/s]
Computing std of features: 100%|█████████████████████████████████████████████████████████████| 695/695 [00:01<00:00, 618.44it/s]
Current loss: 5.1687  : 100%|███████████████████████████████████████████████████████████████| 1000/1000 [00:16<00:00, 60.22it/s]
Final map normalization: 100%|█████████████████████████████████████████████████████████████████| 78/78 [00:00<00:00, 117.57it/s]
Final inference: 100%|███████████████████████████████████████████████████████████████████████| 543/543 [00:03<00:00, 139.98it/s]
Final image auc: 95.1452
```

## TODO

- [ ] Read the paper
- [ ] Understand the metrics
- [ ] Familiarize with the code
- [ ] Integrate this repo with [agbld/led-defects-detection](https://github.com/agbld/led-defects-detection.git)

---

# EfficientAD
Unofficial implementation of paper https://arxiv.org/abs/2303.14535

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/efficientad-accurate-visual-anomaly-detection/anomaly-detection-on-mvtec-loco-ad)](https://paperswithcode.com/sota/anomaly-detection-on-mvtec-loco-ad?p=efficientad-accurate-visual-anomaly-detection)

## Results

| Model         | Dataset    | Official Paper | efficientad.py |
|---------------|------------|----------------|----------------|
| EfficientAD-M | Mvtec AD   | 99.1           | 99.1           |
| EfficientAD-M | VisA       | 98.1           | 98.2           |
| EfficientAD-M | Mvtec LOCO | 90.7           | 90.1           |
| EfficientAD-S | Mvtec AD   | 98.8           | 99.0           |
| EfficientAD-S | VisA       | 97.5           | 97.6           |
| EfficientAD-S | Mvtec LOCO | 90.0           | 89.5           |


## Benchmarks

| Model         | GPU   | Official Paper | benchmark.py |
|---------------|-------|----------------|--------------|
| EfficientAD-M | A6000 | 4.5 ms         | 4.4 ms       |
| EfficientAD-M | A100  | -              | 4.6 ms       |
| EfficientAD-M | A5000 | 5.3 ms         | 5.3 ms       |


## Setup

### Packages

```
Python==3.10
torch==1.13.0
torchvision==0.14.0
tifffile==2021.7.30
tqdm==4.56.0
scikit-learn==1.2.2
```

### Mvtec AD Dataset

For Mvtec evaluation code install:

```
numpy==1.18.5
Pillow==7.0.0
scipy==1.7.1
tabulate==0.8.7
tifffile==2021.7.30
tqdm==4.56.0
```

Download dataset (if you already have downloaded then set path to dataset (`--mvtec_ad_path`) when calling `efficientad.py`).

```
mkdir mvtec_anomaly_detection
cd mvtec_anomaly_detection
wget https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz
tar -xvf mvtec_anomaly_detection.tar.xz
cd ..
```

Download evaluation code:

```
wget https://www.mydrive.ch/shares/60736/698155e0e6d0467c4ff6203b16a31dc9/download/439517473-1665667812/mvtec_ad_evaluation.tar.xz
tar -xvf mvtec_ad_evaluation.tar.xz
rm mvtec_ad_evaluation.tar.xz
```

## efficientad.py

Training and inference:

```
python efficientad.py --dataset mvtec_ad --subdataset bottle
```

Evaluation with Mvtec evaluation code:

```
python mvtec_ad_evaluation/evaluate_experiment.py --dataset_base_dir './mvtec_anomaly_detection/' --anomaly_maps_dir './output/1/anomaly_maps/mvtec_ad/' --output_dir './output/1/metrics/mvtec_ad/' --evaluated_objects bottle
```

## Reproduce paper results

Reproducing results from paper requires ImageNet stored somewhere. Download ImageNet training images from https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data or set `--imagenet_train_path` of `efficientad.py` to other folder with general images in children folders for example downloaded https://drive.google.com/uc?id=1n6RF08sp7RDxzKYuUoMox4RM13hqB1Jo

Calls:

```
python efficientad.py --dataset mvtec_ad --subdataset bottle --model_size medium --weights models/teacher_medium.pth --imagenet_train_path ./ILSVRC/Data/CLS-LOC/train
python efficientad.py --dataset mvtec_ad --subdataset cable --model_size medium --weights models/teacher_medium.pth --imagenet_train_path ./ILSVRC/Data/CLS-LOC/train
python efficientad.py --dataset mvtec_ad --subdataset capsule --model_size medium --weights models/teacher_medium.pth --imagenet_train_path ./ILSVRC/Data/CLS-LOC/train
...

python efficientad.py --dataset mvtec_loco --subdataset breakfast_box --model_size medium --weights models/teacher_medium.pth --imagenet_train_path ./ILSVRC/Data/CLS-LOC/train
python efficientad.py --dataset mvtec_loco --subdataset juice_bottle --model_size medium --weights models/teacher_medium.pth --imagenet_train_path ./ILSVRC/Data/CLS-LOC/train
...
```

This produced the Mvtec AD results in `results/mvtec_ad_medium.json`.

## Mvtec LOCO Dataset

Download dataset:

```
mkdir mvtec_loco_anomaly_detection
cd mvtec_loco_anomaly_detection
wget https://www.mydrive.ch/shares/48237/1b9106ccdfbb09a0c414bd49fe44a14a/download/430647091-1646842701/mvtec_loco_anomaly_detection.tar.xz
tar -xf mvtec_loco_anomaly_detection.tar.xz
cd ..
```

Download evaluation code:

```
wget https://www.mydrive.ch/shares/48245/a4e9922c5efa93f57b6a0ff9f5c6b969/download/430648014-1646847095/mvtec_loco_ad_evaluation.tar.xz
tar -xvf mvtec_loco_ad_evaluation.tar.xz
rm mvtec_loco_ad_evaluation.tar.xz
```

Install same packages as for Mvtec AD evaluation code, see above.

Training and inference for LOCO sub-dataset:

```
python efficientad.py --dataset mvtec_loco --subdataset breakfast_box
```

Evaluation with LOCO evaluation code:

```
python mvtec_loco_ad_evaluation/evaluate_experiment.py --dataset_base_dir './mvtec_loco_anomaly_detection/' --anomaly_maps_dir './output/1/anomaly_maps/mvtec_loco/' --output_dir './output/1/metrics/mvtec_loco/' --object_name breakfast_box
```
