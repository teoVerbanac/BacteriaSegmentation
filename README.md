# BacteriaSegmentation
This code is used for testing the differencesin results between traditional picture segmentation methods and SAM2. The metrics used for measurment are IoU and dice score.

Before running this code the following extensions should be installed:

- numpy
- opencv-python
- pandas
- scikit-learn
- scikit-image
- scipy
- torch
- torchvision
- peft
- if using GPU, pytorch needs to be configured to the right version of CUDA

Also SAM2 should be installed into SAM2 folder
### Usage
JPEGImages folder contains pictures that should be segmented, and SegmentationClass contains the masks that are hand drawn.
splitDataset.py is used for splitting data into training and val folders
LoRa_Sam2.py fine tunes the model
sam2_improved.py evalueates the whole model 
edge_segmentation.py returns scores when using edge segmentation
watershed.py return scores when using watershed segmentation
