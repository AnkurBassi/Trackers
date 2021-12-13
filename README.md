# SEP788 Final Project: Visual Object Tracking

## Goal

The goal of this project is to test different visual object tracking methods, and then find the best model that efficiently and effectively tracks a diverse group of objects given any sequence of frames.

## OTB-2015 Dataset
The Visual Tracker Benchmark (OTB-2015 or OTB-100) is a publicly and freely available dataset that contains 98 sequences of image data. Source: http://cvlab.hanyang.ac.kr/tracker_benchmark/datasets.html

## Summary

Details on data preprocessing can be found in the file `p1_otb100_preprocessing.ipynb`.

### OpenCV Trackers
`p2_1_opencv_tracking.py` contains a class called Tracking that can perform object tracking given an OpenCV tracker name and a sequence name. <br>

`p2_2_tracking_main.py` allows users to parse command line arguments. <br>
Run this file in the command line, choose an available OpenCV tracker, and see how it performs on a sequence available in the sequences folder.
```
python p2_2_tracking_main.py [TRACKER_NAME] [SEQUENCE_NAME] [--show_tracking=TRUE] [--save_frames=TRUE] 
```
### YOLO-v3 Model
