## Quick Start:

Prepare dataset:
```bash
cd dataset
python prepare_dataset.py --annotations original/annotations/instance.json --labeled_dir original/images/ --unlabeled_dir original/normal/B/ --output_dir leddd --train_ratio 0.8
```

Train EfficientAD on custom dataset:
```bash
python train.py --dataset custom --custom_dataset_path dataset/leddd --output_dir output/1 --model_size small --epochs 3 --batch_size 10
```

Evaluate EfficientAD on custom dataset:
```bash
python eval.py --dataset custom --custom_dataset_path dataset/leddd --output_dir output/1 --model_size small --map_format jpg --threshold 20 --weights_dir output/1/trainings/custom
```

### Expected Output

Following shows the metrics under different `threshold` values:

```bash
(effad) ee303@ee303-Z790-AORUS-ELITE:~/Documents/agbld/GitHub/EfficientAD$ python efficientad.py --dataset custom --custom_dataset_path dataset/leddd --output_dir output/1 --model_size small --map_format jpg --train_steps 1000 --threshold 20 --batch_size 4
Computing mean of features: 100%|██████████████████████████████████████████████████████████████████| 190/190 [00:01<00:00, 105.77it/s]
Computing std of features: 100%|███████████████████████████████████████████████████████████████████| 190/190 [00:01<00:00, 167.34it/s]
Current loss: 4.9925  : 100%|█████████████████████████████████████████████████████████████████████| 1000/1000 [00:52<00:00, 18.92it/s]
Final map normalization: 100%|████████████████████████████████████████████████████████████████████████| 22/22 [00:00<00:00, 48.38it/s]
Final inference: 100%|█████████████████████████████████████████████████████████████████████████████| 561/561 [00:05<00:00, 107.96it/s]

Class            Accuracy    Precision    Recall    Num Samples
-------------  ----------  -----------  --------  -------------
defect_type_2      0.9107       1.0000    0.9107            112
defect_type_3      0.9770       1.0000    0.9770             87
defect_type_4      1.0000       1.0000    1.0000            134
defect_type_6      1.0000       1.0000    1.0000              6
defect_type_7      1.0000       1.0000    1.0000             10
good               0.8538       0.0000  nan                 212

Class      Accuracy    Precision    Recall    Num Samples
-------  ----------  -----------  --------  -------------
Overall      0.9234       0.9158    0.9656            561
Final image auc: 97.6807
```

```bash
(effad) ee303@ee303-Z790-AORUS-ELITE:~/Documents/agbld/GitHub/EfficientAD$ python efficientad.py --dataset custom --custom_dataset_path dataset/leddd --output_dir output/1 --model_size small --map_format jpg --train_steps 1000 --threshold 15 --batch_size 4
Computing mean of features: 100%|██████████████████████████████████████████████████████████████████| 190/190 [00:01<00:00, 104.68it/s]
Computing std of features: 100%|███████████████████████████████████████████████████████████████████| 190/190 [00:01<00:00, 166.85it/s]
Current loss: 5.0027  : 100%|█████████████████████████████████████████████████████████████████████| 1000/1000 [00:52<00:00, 19.02it/s]
Final map normalization: 100%|████████████████████████████████████████████████████████████████████████| 22/22 [00:00<00:00, 73.24it/s]
Final inference: 100%|█████████████████████████████████████████████████████████████████████████████| 561/561 [00:05<00:00, 104.84it/s]

Class            Accuracy    Precision    Recall    Num Samples
-------------  ----------  -----------  --------  -------------
defect_type_2      0.9911       1.0000    0.9911            112
defect_type_3      1.0000       1.0000    1.0000             87
defect_type_4      1.0000       1.0000    1.0000            134
defect_type_6      1.0000       1.0000    1.0000              6
defect_type_7      1.0000       1.0000    1.0000             10
good               0.8160       0.0000  nan                 212

Class      Accuracy    Precision    Recall    Num Samples
-------  ----------  -----------  --------  -------------
Overall      0.9287       0.8992    0.9971            561
Final image auc: 97.5739
```

## Case Study

### Good Samples (Normal)

<table>
    <tr>
        <td><img src="assets/good_1.jpg" alt="good_1" /><br /><center>Good 1</center></td>
        <td><img src="assets/good_2.jpg" alt="good_2" /><br /><center>Good 2</center></td>
        <td><img src="assets/good_3.jpg" alt="good_3" /><br /><center>Good 3</center></td>
    </tr>
</table>

### Defect Samples (Anomalies)

<table>
    <tr>
        <td><img src="assets/Particle_Big_1.jpg" alt="Particle_Big_1" /><br /><center>Particle Big 1</center></td>
        <td><img src="assets/Particle_Big_2.jpg" alt="Particle_Big_2" /><br /><center>Particle Big 2</center></td>
    </tr>
</table>
<table>
    <tr>
        <td><img src="assets/particle_1.jpg" alt="particle_1" /><br /><center>Particle 1</center></td>
        <td><img src="assets/particle_2.jpg" alt="particle_2" /><br /><center>Particle 2</center></td>
    </tr>
</table>
<table>
    <tr>
        <td><img src="assets/led_ng_1.jpg" alt="led_ng_1" /><br /><center>LED NG 1</center></td>
        <td><img src="assets/led_ng_2.jpg" alt="led_ng_2" /><br /><center>LED NG 2</center></td>
    </tr>
</table>
<table>
    <td><img src="assets/flip_1.jpg" alt="flip_1" /><br /><center>Flip 1</center></td>
    <td><img src="assets/flip_2.jpg" alt="flip_2" /><br /><center>Flip 2</center></td>
    <td><img src="assets/flip_3.jpg" alt="flip_3" /><br /><center>Flip 3</center></td>
    </tr>
</table>
<table>
    <td><img src="assets/tilt_1.jpg" alt="tilt_1" /><br /><center>Tilt 1</center></td>
    <td><img src="assets/tilt_2.jpg" alt="tilt_2" /><br /><center>Tilt 2</center></td>
    <td><img src="assets/oh_no.jpg" alt="oh_no" /><br /><center>...</center></td>
    </tr>
</table>

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
