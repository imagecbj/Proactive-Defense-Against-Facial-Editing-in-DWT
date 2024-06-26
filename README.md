# 抵抗第二次人脸属性编辑的不可感知主动防御算法

![Python 3.](https://img.shields.io/badge/python-3.8-green.svg?style=plastic)

![PyTorch 1.10.0](https://img.shields.io/badge/pytorch-1.10.0-green.svg?style=plastic)

> **摘要:** *人脸伪造的主动防御技术通过向待保护人脸添加难以察觉的扰动来破坏伪造模型的生成能力. 最近的潜在对抗性探索(Latent Adversarial Exploration, LAE)算法比常用的像素空间类算法实现了更好的扰动不可感知性, 但仍有两个不足: (a) 其防御人脸的语义信息容易被改变，这是一些人脸所有者无法接受的; (b) 其成功防御后的无效输出容易被伪造模型成功编辑. 因此，本文提出了一种抵抗第二次人脸属性编辑的不可感知主动防御算法. 针对前一缺点，替换LAE中非可逆的编码器-生成器结构为正交的离散小波变换(Discrete Wavelet Transform, DWT)，并在DWT域施加扰动. 关于后者, 替代LAE的无效攻击为无目标攻击. 此外, 为了提高防御人脸的视觉质量，利用人眼视觉系统对亮度通道扰动更为敏感的特点, 在YCbCr色彩空间的色度通道中加入扰动; 为了提升防御人脸的通用性, 采用权重动态更新的集成策略进行训练. 实验结果表明, 提出算法优于现有集成和非集成算法, 更好的平衡了扰动不可感知性和防御通用性.*

## 数据集

点击[CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ) 下载数据集并在`main_ensemble.py`修改数据集路径.

## 预训练模型

点击([Ensemble Models and Pre-trained models](https://drive.google.com/drive/folders/10WEoO6C6KkcFqtVb_iCciDEUqjLPJJEJ))下载5个属性编辑模型(StarGAN、AttentionGAN、FGAN、HiSD、AttGAN)的预训练权重以及显著性检测模型、本文提供的一个预训练好的权重. 下载到 `./checkpoints/`文件夹中. 注意： `PG.pth`是本文提供的预训练好的扰动生成器，可以直接用其在测试集上进行测试.

```xml
checkpoints/
├── attentiongan/
│   └── 200000-G.ckpt
├── AttGAN/
|   └── weights.199.pth
├── FAN/
|   └── best-model_epoch-204_mae-0.0505_loss-0.1370.pth
├── FGAN/
|   └── 200000-G.ckpt
├── HiSD/
|   └── gen_00600000.pt
├── stargan/
|   └── 200000-G.ckpt
└── PG.pth
```

## 训练

执行下列命令进行集成训练

```python
  python main_ensemble.py
```

## 测试

执行下列命令在测试集上进行测试，注意：`--model_choice`可以选择的参数如下：`fgan | attgan | hisd | stargan | attentiongan`

```python
  python test_dataset_ensemble.py --model_choice fgan
```



## 致谢

本工作基于 [Fixed-Point-GAN](https://github.com/mahfuzmohammad/Fixed-Point-GAN) 、 [Adversarial-visual-reconstruction](https://github.com/NiCE-X/Adversarial-visual-reconstruction) 和 [TAFIM](https://github.com/shivangi-aneja/TAFIM).
