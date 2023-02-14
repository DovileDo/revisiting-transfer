# Revisiting Hidden Representations in Transfer Learning for Medical Imaging

This repository contains the code for the paper:

Transfer learning has become an increasingly popular approach in medical imaging, as it offers a solution to the challenge of training models with limited dataset sizes. Despite its widespread use, the precise effects of transfer learning on medical image classification are still heavily understudied. We set out to investigate this with a series of systematic experiments on the difference of representations learned from natural (ImageNet) and medical (RadImageNet) source datasets on a range of (seven) medical targets.

We use publicly available pre-trained ImageNet (Keras implementation of ResNet50) and RadImageNet (https://drive.google.com/drive/folders/1Es7cK1hv7zNHJoUW0tI0e6nLFVYTqPqK?usp=sharing) weights as source tasks in our transfer learning experiments.

We investigate transferability to seven medical target datasets:
1. [Chest X-rays](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
2. [PatchCamelyon](http://basveeling.nl/posts/pcam/)
3. [Curated Breast Imaging Subset of Digital Database for Screening Mammography (CBIS-DDSM)](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=22516629)
4. [ISIC2018 - Task 3 - the training set](https://challenge2018.isic-archive.com/task3/training/)
5. [Thyroid ultrasound](https://www.kaggle.com/datasets/dasmehdixtr/ddti-thyroid-ultrasound-images)
6. [Breast ultrasound](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)
7. [ACL and meniscus tear detection](https://stanfordmlgroup.github.io/competitions/mrnet/)
