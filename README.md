# STEPS TAKEN

## 1. Dataset Preparation

To imitate real detection models used on cars <br>
for training we chose following classes to detect:

- 'person',
- 'car',
- 'Green Light',
- 'Red Light',
- 'Speed Limit 10',
- 'Speed Limit 100',
- 'Speed Limit 110',
- 'Speed Limit 120',
- 'Speed Limit 20',
- 'Speed Limit 30',
- 'Speed Limit 40',
- 'Speed Limit 50',
- 'Speed Limit 60',
- 'Speed Limit 70',
- 'Speed Limit 80',
- 'Speed Limit 90',
- 'Stop'

So in the end we would be able to detect **pedestrians, cars and traffic signs.**

In order to train model, we have to merge two datasets.

### Official [**COCO dataset**](https://docs.ultralytics.com/ru/datasets/detect/coco/) <br>

already has enough training samples for cars and people, <br> so we used it as base, <br> but we have to filter it out beforehand in order to remove unneccessary classes feeding.

![coco_dataset](image_for_readme_1.png)
Sample of pictures from COCO, already filtered classes: car, people.

#

#

#

### [TRAFFIC SIGN DETECTION dataset](https://www.kaggle.com/datasets/pkdarabi/cardetection)

![alt text](image_for_readme_2.png)
Model with various samples of speed limit and stop signs.
<br>

For merging datasets we used [fiftyone](https://docs.voxel51.com/) library. <br>
It has exact functionality we need:

- Filter by classes and download COCO dataset
- Merge datasets
- Manage samples
- Export to Yolo dataset format
