# On the Robustness of Deep Clustering Models: Adversarial Attacks and Defenses

The official code repository for the NeurIPS'2022 paper "[On the Robustness of Deep Clustering Models: Adversarial Attacks and Defenses](https://arxiv.org/abs/2210.01940)".

Note: We provide all the pre-trained saved models for the original Deep Clustering models that we consider for the experiments.  We also provide all the saved generator models to run our attack (Provided you place & train the original models on the original datasets). 
  
## Main Goal
- The main goal of this work is to show that deep clustering models are susceptible to adversarial attacks at inference time
- Other optimization approaches instead of the GAN can also be used
- We believe our attack loss function provides better results with fewer queries, but other (supervised) losses can also be used if desired
  
## Original Models/Other Links

1. Google Drive for Saved Models: https://drive.google.com/drive/folders/19aUz6zQFC_xlAW2PgMypXcLraKzozxTA?usp=sharing
2. CC: https://github.com/Yunfan-Li/Contrastive-Clustering
3. MICE: https://github.com/TsungWeiTsai/MiCE
4. NNM: https://github.com/ZhiyuanDang/NNM
5. SCAN: https://github.com/wvangansbeke/Unsupervised-Classification
6. RUC: https://github.com/deu30303/RUC
7. SPICE: https://github.com/niuchuangnn/SPICE
8. Face++ User Console: https://console.faceplusplus.com/login
9. Extended Yale Face Database B: http://vision.ucsd.edu/~leekc/ExtYaleDatabase/ExtYaleB.html


## Directory Overview

- **Anomaly_Detection_and_facepp:** Contains the experiments for Anomaly Detection and attack on Face++.
- **CC:** Contains all attacks for the CC model.
- **Generator_Models:** Contains the saved models for generators trained on all the deep clustering models to be used for transferability attacks.
- **MICE:** Contains all attacks for the MICE model.
- **NNM:** Contains all attacks for the NNM model.
- **RUC:** Contains all attacks for the RUC model.
- **SCAN:** Contains all attacks for the SCAN model.
- **SPICE:** Contains all attacks for the SPICE model.

## Main Attacks

All the main attacks for the models are situated in their own separate directories (CC/, MICE/, NNM/, SCAN/, RUC/, SPICE/). We provide all the files such that you can independently train the models by following their own README.md

For the attack, Inside the root directories of the models they have their own Jupyter Notebooks segregated by the dataset (Eg: `CC_Attack-CIFAR-10.ipynb`). Inside the notebooks the code logic flows in the following fashion:
1. Evaluates the performance of the original model on original dataset.
2. Trains GAN in a black-box fashion and saves the generator.
3. Evaluate the model on the generated adversarial samples. 
4. Save generated adversarial v.s. original samples with their predicted clusters.
5. Transferability attacks.
6. Generate additional results (perf w. norm).

**Note:**  To successfuly run the code, you will have to download/train and place the original deep clustering models, and update the paths accordingly. The code will automatically download all the datasets. 

**Note:** We utilized and modified the original AdvGAN implementation by mathcbc [here](https://github.com/mathcbc/advGAN_pytorch)


## Anomaly Detection

The code for Anomaly detection is present inside `Anomaly_Detection_and_facepp/SSD_CC`. You will have to train (or place the pre-trained models that we provide) the Anomaly detection for CIFAR-(10, 100) and STL-10 on ResNet18/34. Then run the individual Jupyter notebooks to get the results. 

## Face++

To Train and attack CC on Extended Yale Face Database B, From the root directory, you will have to first run `/CC/yale_face_CC.ipynb` (this will train CC on Extended Yale Face Database B). Next, `CC_Attack-Yale.ipynb` will attack the previously trained CC model and will save the generator for generating the adversarial samples. 

**Note:** You can skip the above step as we already provide the trained generator.

Code for attacking Face++ is present inside `anomaly_facepp`. 

**netG_cc_yale-attack_epoch_300.pth**: Is the saved generator after attacking CC on Extended Yale Face B dataset.

First run, `adv-gen.ipynb`, this will generate and save the adversarial samples along with original face images in `custom/`, and `custom_adv/` directories. Then `face++-Main.ipynb` will attack the face++ API using the previously saved adv samples. The rest of the files are additional experiements with the API. 

**Note:** You will have to generate the API key and secret by going into their user console (https://console.faceplusplus.com/login). 
