# Modeling Multimodal Aleatoric Uncertainty in Segmentation with Mixture of Stochastic Experts
Welcome to the GitHub repository for *Modeling Multimodal Aleatoric Uncertainty in Segmentation with Mixture of Stochastic Experts*. The complete code for this project will be released **before the end of April**. Please check back later for the code and instructions on how to use it. Thanks for your interest!

## Updates
- 03/01/2023 Upload evluation metrics.

<!--
## 0. Build up environment.
- cuda 10.1
- Follow the instructions to build up the environment setting.
```angular2html
conda env create -f env/environment.yaml
```
## 1. Prepare the data and set the path.
1. Download the LIDC dataset and the Cityscapes dataset follow the instruction in https://github.com/EliasKassapis/CARSSS.
2. Run data/cityscapes_data_loader.py and data/lidc_data_loader.py to preprocess the dataset.
3. Set the dataset path in the code. The default dataset path is '../data/lidc_npy' and  '../data/cityscape_npy_5'. You can modify it in the experiments config files. (See ./experiments/)

## 2. Train model.
Select one model to train in models/experiments.
```angular2html
python main.py experiments/lidc_proposed.py --parallel
```

## 3. Inference
```angular2html
python main.py experiments/lidc_proposed.py --parallel --demo test
```
-->
