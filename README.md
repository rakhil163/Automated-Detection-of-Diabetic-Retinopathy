# Diabetic Retinopathy

This project compares the performance of models like KNN, SVM , Dense CNN and pre trained CNN in automating detection of diabetic retinopathy.

## Dataset : [Kaggle Dataset](https://www.kaggle.com/c/diabetic-retinopathy-detection/data)

## Steps to run the files</br>
_To run the pre-trained CNN_</br>
* Run the move.py file to label the training images as healthy and non healthy retina </br>
* Download [Inception Dependency](http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz)
* python retrain.py --bottleneck_dir=bottlenecks --how_many_training_steps=300 --model_dir=inception --output_graph=retrained_graph.pb --output_labels=retrained_labels.txt --image_dir="C:\Copy of SD card\train1\train\500_selected" </br>

_To run SVM_</br>
* Run the move.py file to label the training images as healthy and non healthy retina </br>
* python diab.py

_To run KNN_</br>
* Run the move.py file to label the training images as healthy and non healthy retina </br>
* python AkhilSVM.py --dataset folderContainingTheTrainingImages

_To run Dense CNN_</br>
* Cell wise execution of the .ipynb file </br>
