# SF-TL54: Annotated Thermal Facial Landmark Dataset with Visual Pairs
Download the repository:
```
git-clone https://github.com/IS2AI/thermal-facial-landmarks-detection.git
```
## Data preparation
Download the dataset from [google drive](https://drive.google.com/drive/folders/1XLehM5DYqLqiAsteO_h1PYZnavcCNOcR?usp=sharing).

### Generate training, validation, and testing XML files for dlib shape predictor
```
python build_dlib_landmarks_xml.py --dataset dataset/ --color gray --set train
python build_dlib_landmarks_xml.py --dataset dataset/ --color gray --set val 
python build_dlib_landmarks_xml.py --dataset dataset/ --color gray --set test
```

### Generate training, validation, and testing ground-truth masks for U-net

## dlib shape predictor
### Training
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
### Testing
```
python test_dlib_predictor.py --testing dataset/gray/test/dlib_landmarks_test.xml --model models/dlib_landmarks_predictor.dat
```

## U-net 

### Pre-trained models
1. Download the models from [google drive](https://drive.google.com/drive/folders/1XLehM5DYqLqiAsteO_h1PYZnavcCNOcR?usp=sharing).
2. Copy the models to the models folder of this repo.
### dlib shape predictor
- To make predictions on images:
```
python dlib_predict_image.py --images dataset/gray/test/images/ --models  models/ --upsample 1
```
- To make predictions on a video:
```
python dlib_predict_video.py --input video/2_0.avi --models  models/ --upsample 1 --output video/output.mp4
```
### U-net
```
python unet_predict_image.py --dataset dataset/gray/test --model  models/ 
```







