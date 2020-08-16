# Tensorflow-MINST-CelebA-VAE-GAN

这是一个练手项目，主要用来实现VAE和GAN的一些基本模型，相对而言比较简单。

采用的数据集为`MINST`和`CelebA`

## OUTPUT

|CVAE|labeled CVAE|GAN|
|:---:|:---:|:---:|
|![](./imgs/minst_cvae.gif)|![](./imgs/minst_cvae_label.gif)|![](./imgs/minst_gan.gif)|
|||


## 训练GAN的一些trick

参考: [训练GANs一年我学到的10个教训](https://zhuanlan.zhihu.com/p/79959150)

### 标签平滑

### 多尺度梯度

### Two Time-Scale Update Rule

### 谱归一化

### critical loss