# SF-TL54: Thermal Facial Landmark Dataset with Visual Pairs
The dataset contains 2,556 thermal-visual image pairs of 142 subjects with manually annotated face bounding boxes and 54 facial landmarks. The dataset was constructed from our large-scale [SpeakingFaces dataset](https://github.com/IS2AI/SpeakingFaces).

<img src= "https://raw.githubusercontent.com/IS2AI/thermal-facial-landmarks-detection/main/figures/example.png"> 

## Link to the paper
[SF-TL54: A Thermal Facial Landmark Dataset with Visual Pairs](https://ieeexplore.ieee.org/abstract/document/9708901)

## The facial landmarks are ordered as follows:

<img src= "https://raw.githubusercontent.com/IS2AI/thermal-facial-landmarks-detection/main/figures/land_conf.png"> 

## Download the repository:
```
git clone https://github.com/IS2AI/thermal-facial-landmarks-detection.git
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
<img src= "https://raw.githubusercontent.com/IS2AI/thermal-facial-landmarks-detection/main/figures/demo.gif" width="464" height="348"> 

To detect faces, we trained the HOG+SVM based face detection model on our dataset. If you need more robust model then please check our [TFW: Annotated Thermal Faces in the Wild](https://github.com/IS2AI/TFW) project.

1. Download the models from [google drive](https://drive.google.com/drive/folders/1XLehM5DYqLqiAsteO_h1PYZnavcCNOcR?usp=sharing).
2. Put the pre-trained models inside `/thermal-facial-landmarks-detection/models` directory.
3. **dlib shape predictor**
- Make predictions on images:
```
python dlib_predict_image.py --images PATH_TO_IMAGES --models  models/ --upsample 1
```
- Make predictions on a video:
```
python dlib_predict_video.py --input PATH_TO_VIDEO --models  models/ --upsample 1 --output output.mp4
```

4. **U-net model**
```
python unet_predict_image.py --dataset dataset/gray/test --model  models/ 
```


## For dlib face detection model (HOG + SVM)
- Training the model:
```
python train_dlib_face_detector.py --training dataset/gray/train/dlib_landmarks_train.xml --validation dataset/gray/val/dlib_landmarks_val.xml
```
- Make predictions on images:
```
python dlib_face_detector.py --images dataset/gray/test/images --detector models/dlib_face_detector.svm
```

## To visualize dataset
- Thermal images with bounding boxes and landmarks:
```
python visualize_dataset.py --dataset dataset/ --color iron --set train
```
- Thermal-Visual pairs
```
python visualize_image_pairs.py --dataset dataset/ --color iron --set train

```

## Video tutorials
- [Face detection: Thermal domain.](https://www.youtube.com/watch?v=tzgGPVQwqq8&t=10s)
- [Facial landmark detection: Thermal domain.](https://www.youtube.com/watch?v=_7e3N3pMYDg&t=63s)

## If you use the dataset/source code/pre-trained models in your research, please cite our work:
```
@INPROCEEDINGS{9708901,
  author={Kuzdeuov, Askat and Koishigarina, Darina and Aubakirova, Dana and Abushakimova, Saniya and Varol, Huseyin Atakan}
  booktitle={2022 IEEE/SICE International Symposium on System Integration (SII)}, 
  title={SF-TL54: A Thermal Facial Landmark Dataset with Visual Pairs}, 
  year={2022},
  volume={},
  number={},
  pages={748-753},
  doi={10.1109/SII52469.2022.9708901}}
```

  ```
  @Article{s21103465,
 AUTHOR = {Abdrakhmanova, Madina and Kuzdeuov, Askat and Jarju, Sheikh and Khassanov, Yerbolat and Lewis, Michael and Varol, Huseyin Atakan},
 TITLE = {SpeakingFaces: A Large-Scale Multimodal Dataset of Voice Commands with Visual and Thermal Video Streams},
 JOURNAL = {Sensors},
 VOLUME = {21},
 YEAR = {2021},
 NUMBER = {10},
 ARTICLE-NUMBER = {3465},
 URL = {https://www.mdpi.com/1424-8220/21/10/3465},
 ISSN = {1424-8220},
 DOI = {10.3390/s21103465}
}
```


