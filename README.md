# Vids2 Model - Making ML Accessible
Vids2Model is a Python3 tool to automate most of the work in making an image classification system.
<br>Vids2Model takes two videos as input, and outputs a keras CNN model with two classifications.
## Installation
After downloading the files, install the python dependencies by executing `pip install -r requirements.txt` in Vids2Model's directory.
<br>Additionally, zip (standard on most devices) and FFmpeg are required for Vids2Model.
## Usage
There are 13 arguments that Vids2Model uses to make your desired image classification model. They are:
- Name of model
- Path of first video
- Path of second video
- Width of images (you can downscale)
- Height of images (you can downscale)
- Learning rate (0 = model doesn't learn, 1 = model forgets everything except last input, 0.001 recommended)
- Epochs (how many rounds of training the model will perform)
- isShufflingBetweenEpochs (1 or 0, whether dataset will be shuffled in between epochs)
- isChangingBrightness (1 or 0, whether brightness varies between images)
- isChangingSharpness (1 or 0, whether sharpness varies between images)
- isChangingContrast (1 or 0, whether contrast varies between images)
- isChangingRotation (1 or 0, whether random rotations are present)
- isChangingPosition (1 or 0, whether center of image is randomly moved)

Example: `python3 vids2model.py testModel 1.mp4 2.mp4 1920 1080 0.001 10 0 1 1 1 0 0`
<br>In this case, a new model named testModel will be created with 1.mp4 and 2.mp4 as inputs.
<br>The images will be 1080p, the learning rate will be 0.001, and 10 epochs will occur.
<br>No shuffling, rotations, or position changes will occur, but brightness, sharpness, and contrast will be randomized.
<br><br>The output of this will be a zip file containing the model, a config.txt containing the resolution, and two graphs.
<br> The first graph will contain a confusion matrix (a matrix showing true positives, false positives, false negatives, etc).
<br> The second graph will show the model accuracy over each epoch.