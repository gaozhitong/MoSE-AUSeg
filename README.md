# Modeling Multimodal Aleatoric Uncertainty in Segmentation with Mixture of Stochastic Experts
[![Python 3.7](https://img.shields.io/badge/Python-3.8-3776AB.svg?logo=python)](https://www.python.org/) [![PyTorch 1.4](https://img.shields.io/badge/PyTorch-1.10-EE4C2C.svg?logo=pytorch)](https://pytorch.org/docs/1.4.0/) [![Apache](https://img.shields.io/badge/License-Apache-3DA639.svg?logo=open-source-initiative)](LICENSE)

This is the official code repository for [*Modeling Multimodal Aleatoric Uncertainty in Segmentation with Mixture of Stochastic Experts*](https://arxiv.org/pdf/2212.07328.pdf) by Zhitong Gao, Yucong Chen, Chuyu Zhang, Xuming He (ICLR 2023).
- **Additional resources:** [OpenReview](https://openreview.net/forum?id=KE_wJD2RK4)  | [Poster](https://gaozhitong.github.io/posters/poster-iclr.pdf) | [Video Presentation](https://www.youtube.com/watch?v=SVyqWKnR_pQ)
- **In a nutshell:** We propose a novel mixture of stochastic experts (MoSE) model training with a Wasserstein-like loss, which produces an efficient two-level representation for the multi-modal aleatoric uncertainty in semantic segmentation.

![avatar](imgs/overview.jpg)
# 1. Preparation
## Dataset
We evaluate our method on two public benchmarks, including the [LIDC-IDRI dataset](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI) 
and a synthetic multimodal version of the [Cityscapes dataset](https://www.cityscapes-dataset.com/).
Follow the instructions below to access and preprocess the data.
### LIDC-IDRI Dataset
1. Download the pre-processed 2D crops provided by [Probabilistic-Unet-Pytorch](https://github.com/stefanknegt/Probabilistic-Unet-Pytorch) in this [link](https://drive.google.com/drive/folders/1xKfKCQo8qa6SAr3u7qWNtQjIphIrvmd5?usp=sharing).
2. Process and organize the data directory by running [data/preprocess/lidc_data_loader.py](data/preprocess/lidc_data_loader.py) script. You may need to change the path 'data_root' to the directory where
you save the downloaded data.  Also, define the paths of the processed data. 
This will generate an HDF5 data package as well as a NumPy directory. Note that in our experiments, only the NumPy data directory is used. You can delete the others if you prefer.

### Cityscapes Dataset
1. Download the [images](https://www.cityscapes-dataset.com/file-handling/?packageID=3) and original [annotations](https://www.cityscapes-dataset.com/file-handling/?packageID=1) from the official Cityscapes dataset.
2. Set the data directories in the [data/preprocess/preprocessing_config.py](data/preprocess/preprocessing_config.py) file and run [data/preprocess/preprocessing.py](data/preprocess/preprocessing.py) script to rescale the data to a resolution of 256 x 512 and save as numpy arrays.
3. Download the black-box predictions provided [here](https://drive.google.com/file/d/1EkJD1PUe7J5f5oc_VvUj-7a7XTT-I-Gc/view) as in [CARSSS](https://drive.google.com/file/d/1EkJD1PUe7J5f5oc_VvUj-7a7XTT-I-Gc/view). 
4. Lastly, set up the dataset by running the [data/preprocess/cityscapes_data_loader.py](data/preprocess/cityscapes_data_loader.py) script.
This operation will split the dataset into training, validation, and test sets, construct multiple labels, and their corresponding probabilities. You may need to modify some data paths in the file to fit your specific situation.

## Environment
Our code requires CUDA 10.1 and some other dependencies. We provide an easy way to set up the environment using conda.
```python
# Create a new conda environment named 'MoSE'
conda env create -f env/environment.yaml

# Activate the environment
conda activate MoSE
```
# 2. Training the Model
After successfully preparing the datasets and setting up the environment, you are ready to train the model. 
The code allows you to run different experiments by using different experiment configuration files.

Here is an example of using the 'experiments/lidc_proposed.py' file to train the model with the proposed configuration for the LIDC dataset. Make sure to set the dataset path in the experiment configuration file. The default dataset path is '../data/lidc_npy' and '../data/cityscape_npy_5'.

```python
# Example command for running the model with different experiment configuration files
python main.py experiments/lidc_proposed.py
```
In the 'experiments/' directory, we provide three configuration files to reproduce the main results presented in our paper. In addition to 'lidc_proposed.py', we have 'lidc_proposed_1annot.py' for the LIDC dataset with one annotation per image and 'cityscapes_proposed.py' for the Cityscapes dataset.

### Other Training Options
You can explore and modify the experiment configuration files to adapt the training settings according to your requirements. 
For example, you can change the UNet backbone and the number of filters to [32, 64, 128, 192] to reproduce a lighter model as mentioned in the paper. 
Additionally, you can set the use_bbpred option to False in cityscapes_proposed.py to run experiments without the black box predictions.

Feel free to experiment with different configurations and options to achieve the desired results.

## 3. Testing the Model
After completing the training, you can assess the model's performance by executing the following command.
```python
python main.py experiments/lidc_proposed.py --demo test
```
This command will evaluate the model's performance by performing inference on the test dataset using the trained model associated 
with the experiment_name specified in the configuration file. The selected model for evaluation will be the one that achieved the highest validation GED score during the training process.

### Output Representation
By default, the command generate a number of samples in the **compact representation**. 
The sample number is determined by multiplying the number of experts by the 'sample_per_mode' variable.
To obtain results with a different number of samples in the compact representation, you can modify the value of the 'sample_per_mode' variable.

Alternatively, if you want to obtain results in the **standard representation**, you can change the value of the 'eval_sample_num' variable. 
For instance, setting eval_sample_num to 16 will produce inference results with 16 samples in the standard form.

### Pretrained Models
We provide pre-trained models, which can be downloaded using the following links. 
Please note that the UNet backbone design has been slightly modified based on [ProbUnet-Pytorch](https://github.com/stefanknegt/Probabilistic-Unet-Pytorch). 
Consequently, the resulting parameters are slightly smaller than those reported in the paper. However, the performance remains unchanged. \
(*Results are demonstrated with a sample number of 16 in the standard representation.)

| Dataset     | Annotation | Download                       | GED(16)  | M-IoU(16) | ECE(%)(16)  | # param.  |
|-------------|------------|--------------------------------|------|-------|------|------|
| LIDC        | Full       | [ckpt](https://drive.google.com/file/d/12JNF7JJ1gwQjrpIMBBZiA7lx_Q9UvYUE/view?usp=sharing) | 0.213 | 0.622   | 0.045 | 37.28 M|
| LIDC        | One        | [ckpt](https://drive.google.com/file/d/1UvcDHpi55NQhlzaeJZDDwocXOCxS8CT4/view?usp=sharing) | 0.243 | 0.599   | 0.096 | 37.28 M|
| Cityscapes  | Full       | [ckpt](https://drive.google.com/file/d/1L8_ED9TRswm1dy1zLerjQXlJdCmHju6j/view?usp=sharing) | 0.178 | 0.641   | 3.260 | 37.32 M|
 
After downloading the .ckpt files, please place them in the log directory and organize the files as follows:
```
.
├── log
│   ├── cityscapes
│   │   ├── cityscapes_pretrained
│   │   │   └── cityscapes_pretrained_best_ged.pth
│   │   └── ...
│   ├── lidc
│   │   ├── lidc_pretrained
│   │   │   └── lidc_pretrained_best_ged.pth
│   │   ├── lidc_pretrained_1annot
│   │   │   └── lidc_pretrained_1annot_best_ged.pth
│   │   └── ...
```
After arranging the files, follow the previous instructions for testing the model to perform inference. You should obtain results that are roughly similar to the ones presented in the table.

# Updates
Here we keep a record of all major updates to the code and model.
- 03/01/2023 Uploaded evluation metrics.
- 04/29/2023 Uploaded the complete code.  
- 06/27/2023 Uploaded pre-trained model files, refined some parts of the code and README.

# Contact
If you have any issues or questions related to the code, please feel free to open an issue on GitHub. 
For general discussions, suggestions, or further assistance, you can reach out to me (Zhitong Gao) via email at [gaozht@shanghaitech.edu.cn](mailto:gaozht@shanghaitech.edu.cn).


# Citation
If you find our work helpful for your research, please consider citing our paper:
```bibtex
@inproceedings{gao2023modeling,
    title={Modeling Multimodal Aleatoric Uncertainty in Segmentation with Mixture of Stochastic Experts},
    author={Zhitong Gao and Yucong Chen and Chuyu Zhang and Xuming He},
    booktitle={The Eleventh International Conference on Learning Representations },
    year={2023},
    url={https://openreview.net/forum?id=KE_wJD2RK4}
}
```