# SF-TL54: Annotated Thermal Facial Landmark Dataset with Visual Pairs
<img src= "https://github.com/IS2AI/thermal-facial-landmarks-detection/blob/main/figures/demo.gif" width="464" height="348"> 

Download the repository:
```
git-clone https://github.com/IS2AI/thermal-facial-landmarks-detection.git
```
## Requirements
- imutils
- OpenCV
- NumPy
- Pandas
- dlib
- Tensorflow 2

To install the necessary packages properly, we ask you to walk through these two tutorials:
1. [How to install TensorFlow 2.0 on Ubuntu](https://www.pyimagesearch.com/2019/12/09/how-to-install-tensorflow-2-0-on-ubuntu/).
2. [Install dlib (the easy, complete guide)](https://www.pyimagesearch.com/2018/01/22/install-dlib-easy-complete-guide/).

## Data preparation
Download the dataset from [google drive](https://drive.google.com/drive/folders/1XLehM5DYqLqiAsteO_h1PYZnavcCNOcR?usp=sharing).

- Generate training, validation, and testing XML files for dlib shape predictor
```
python build_dlib_landmarks_xml.py --dataset dataset/ --color gray --set train
python build_dlib_landmarks_xml.py --dataset dataset/ --color gray --set val 
python build_dlib_landmarks_xml.py --dataset dataset/ --color gray --set test
```

- To generate training, validation, and testing ground-truth masks for U-net, open the `unet_generate_masks.ipynb` notebook and run cells.

## Training and testing dlib shape predictor
- To manually tune parameters of the model:
```
python train_dlib_predictor.py --training dataset/gray/train/dlib_landmarks_train.xml --validation dataset/gray/val/dlib_landmarks_val.xml
```
- To search optimal parameters via grid search:
```
python dlib_grid_search.py
```
- To optimize parameters via dlib's global optimizer:
```
python dlib_global_optimizer.py
```
- Testing the trained model:
```
python test_dlib_predictor.py --testing dataset/gray/test/dlib_landmarks_test.xml --model models/dlib_landmarks_predictor.dat
```

## Training and testing the U-net model
For training and testing the U-net model, open the `train_unet_predictor.ipynb` notebook and run cells.

## Pre-trained models
1. Download the models from [google drive](https://drive.google.com/drive/folders/1XLehM5DYqLqiAsteO_h1PYZnavcCNOcR?usp=sharing).
2. Put the pre-trained models inside `/thermal-facial-landmarks-detection/models` directory.
3. **dlib shape predictor**
- Make predictions on images:
```
python dlib_predict_image.py --images dataset/gray/test/images/ --models  models/ --upsample 1
```
- Make predictions on a video:
```
python dlib_predict_video.py --input video/2_0.avi --models  models/ --upsample 1 --output video/output.mp4
```
4. **U-net model**
```
python unet_predict_image.py --dataset dataset/gray/test --model  models/ 
```







