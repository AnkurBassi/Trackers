# SEP788 Final Project: Visual Object Tracking

## Goal

The goal of this project is to test different visual object tracking methods, and then find the best model that efficiently and effectively tracks a diverse group of objects given any sequence of frames.

## OTB-2015 Dataset
The Visual Tracker Benchmark (OTB-2015 or OTB-100) is a publicly and freely available dataset that contains 98 sequences of image data. Source: http://cvlab.hanyang.ac.kr/tracker_benchmark/datasets.html

## Summary

Details on data preprocessing can be found in the file `p1_otb100_preprocessing.ipynb`.

### OpenCV Trackers
The pretrained models and all the image files are stored in Google Drive: https://drive.google.com/drive/folders/1qgGQRJDDN9GVaOx3iU4-4hp8T0PKYsE_?usp=sharing

Download and unzip `pretrained_models.zip`, `results.zip`, `sequences.zip` in order to run the following Python files.

`p2_1_opencv_tracking.py` contains a class called Tracking that can perform object tracking given an OpenCV tracker name and a sequence name. <br>

`p2_2_tracking_main.py` allows users to parse command line arguments. <br>
Run this file in the command line, choose an available OpenCV tracker, and see how it performs on a sequence available in the sequences folder.

```
python p2_2_tracking_main.py [TRACKER_NAME] [SEQUENCE_NAME] [--show_tracking=TRUE] [--save_frames=TRUE] 
```

`p2_3_track_all.py` applies the DaSiamRPN tracker on all available sequences.

### YOLO-v3 Model
`p2_4_Yolo_inference.ipynb` shows the tracking results of the YOLO-v3 model on 5 different sequences. The required files and the resulting videos can be found in another Google Drive folder:
https://drive.google.com/drive/folders/1BLC443oy5nGChvr-QzrbgQ6_6tgVaGCi?usp=sharing
