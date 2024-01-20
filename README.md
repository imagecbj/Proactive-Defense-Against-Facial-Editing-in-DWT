## Proactive Defense Against Deep Facial Attribute Editing via Non-Targeted Adversarial Perturbation Attack in the DWT domain

![Python 3.](https://img.shields.io/badge/python-3.8-green.svg?style=plastic)

![PyTorch 1.10.0](https://img.shields.io/badge/pytorch-1.10.0-green.svg?style=plastic)

> **Abstract:** *Proactive defense against face forgery disrupts the generative ability of forgery models by adding imperceptible perturbations to the faces to be protected. The recent latent space algorithm, i.e., Latent Adversarial Exploration (LAE), achieves better perturbation imperceptibility than the commonly-used image space algorithms but has two main drawbacks: (a) the forgery models can successfully edit the nullifying outputs after defense again; (b) the semantic information of defensed images is prone to be altered. Therefore, this paper proposes a proactive defense algorithm against deep facial attribute editing via non-targeted adversarial perturbation attack in the DWT domain. To address the former drawback, the nullifying attack is replaced by the non-targeted attack. Regarding the latter one, the perturbations are performed in the DWT domain. Furthermore, to speed user-concerned inference time, the generator-based approach is considered for generating frequency domain perturbations instead of the iterative approaches; to improve the visual quality of the defensed images, the perturbations are added in chrominance channels of YCbCr color space because the Human Visual System (HVS) is more sensitive to the perturbations in luminance channel. Numerous experimental results indicate that the proposed algorithm outperforms some existing algorithms, effectively disrupting the facial forgery system while achieving perturbation imperceptibility.*

## Datasets

Follow [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ) to download the dataset and specify the path in main.py.

## Pre-trained Models

Please download the pre-trained models from the following links and save them to `checkpoints/`

| SM                                                           | SA                                                           | PG                                                           |
| :----------------------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| Pretrained [FGAN](https://drive.google.com/file/d/1PQ5yZZ3lnyfN_gtShcdHdcDjCoHmwLXd/view?usp=sharing) Model. | [Saliency Detection Model](https://drive.google.com/file/d/1nwVloVzRLOGs7QL8QbBK0HA8ur9wPCTg/view?usp=sharing) | [Perturbation Generator](https://drive.google.com/file/d/17Lwzd_0NMW8_uE3ofJ53vC_d6-5Ac_9k/view?usp=sharing) |

## Train

<p align="justify"> Simply run the following command


```python
  python main.py
```

## Test

<p align="justify"> Run the following command to test one image, there are some test images in `test/test_data/`


```python
  python test_one_img.py
```

<p align="justify"> Run the following command to test on test dataset


```
  python test_dataset.py
```

## Acknowledgment

This repo is based on [Fixed-Point-GAN](https://github.com/mahfuzmohammad/Fixed-Point-GAN) 、 [Adversarial-visual-reconstruction](https://github.com/NiCE-X/Adversarial-visual-reconstruction) and [TAFIM](https://github.com/shivangi-aneja/TAFIM), thanks for their great work.

