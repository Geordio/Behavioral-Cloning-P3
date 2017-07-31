# **Behavioral Cloning** 


---

## Behavioral Cloning Project

### This readme documents my solution to the Udacity Behavioural Cloning Project of the SDC Nanodegree.


The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

---

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md summarizing the results and fullfilling the role project writeup report(this file)
* run.mp4 a sample video showing the autonomous driving of track 1
* helpers.py methos to help with visualisation and investigations

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
The model should perform well on track 1


#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file includes a description of the pipeline I used for reading the dataset from the csv file, pre-processing images and training and validating the model.
The drive.py file was modified to include image preprocessing of the images coming from the feed from the simulator. If I had included the preprocessing in my model using keras this would not have been necessary.

I used a generator for loading data to the model. The generator also peforms random flips on the images.



###Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I implemented the CNN architecture based on the Nvidia end to end learning paper: https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/

Prior to this project, a college and I had purely coincidentally used this network implementation to drive an RC car round a track. For that task I had started with a implementation in tensorflow from Sully Chen, https://github.com/SullyChen/Autopilot-TensorFlow. Adapting this to allow it to interface to a RC car using raspberry pi and arduino gave very successful results. Again, this was done independent and prior to the behaviour cloning project but it gave me some experience of the architecture and some of the difficulties. 
I have not reused any of the code from the previous project in my implementation here, but the experience was beneficial.

With very little upfront work, the Nvidia architecture gave a good initial result, with the vehicle progressing round more than half the course without any issues. However, after reaching  a particularly tight left hand bend, with new features on the road edge, the car did not turn enough and crashed.

I had issues with Kera and Lambda layers, with a seemingly known bug preventing the model from being saved when using lambda layers on windows 10. (https://github.com/fchollet/keras/issues/7184)
I encountered this when using Lambda to normalise the image. 
I used the workaround https://stackoverflow.com/questions/41847376/keras-model-to-json-error-rawunicodeescape-codec-cant-decode-bytes-in-posi/41926741, i.e used BatchNormalization
It felt a little black box, something I wasnt happy with, and as I was having issues making improvements, I decided to change my preprocessing techniques by using a combination of cropping images by accessing them as arrays, and scipy.misc.imresize for resizing. I cropped the images by taking removing the top 66 and bottom 24 rows of the image, leaving a 66 by 320 image. Then I rszie to 66 x 200, matching the Nvidia architecture inputs.

The model includes RELU layers to introduce nonlinearity after each convolution.


#### 2. Attempts to reduce overfitting in the model

The model contains multiple dropout layers in order to prevent overfitting.
Dropouts were included after the 5 convolutions.
I trialled the use of further dropout layers following each of the fully connected layers but found that it had a very negative influence on the performance.
I set the dropout probability to 0.2. ie 20% of the units will be dropped.

In addition, I captured approximatly 6 laps of data from track 2 in order to prevent overfitting to track 1. Track 2 has a vastly different layout, and includes many hills, much tighter turns, and areas of shadow that will allow assist in preventing overfitting to track 1.

After still encountering problems, I implemented L2 regularization with L2 set to 0.001

#### 3. Model parameter tuning

I used the adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

I captured training data carefully as follows:
- 3 laps in each direction of track 1
- recovery actions from the edge of the track to the centre of the road
- mulitiple short recordings at areas I thought could be troublesome, such as the bridge and tighter turns
- 3 laps of track 2

At this point I was still having performance issues. The car would not make all the turns and would end up off course.
I believed that this was down to my data (however, afterwards I found a bug that could have been the cause)
In order to correct the problem with my data, I did further data capture:
- I captured more laps in both directions
- as further laps of track 2 in both directions.
This gave me a lot of data that I would be able to tune to suit my needs as required. In hindsight, capturing extra data was not really necessary, as I found a bug that would likely let my model perform well on a much smaller set of data.



### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I initially captured a sample of training data to allow initial training after implementing a model. To do this drove 3 laps anticlockwise round track 1.

As with David Silvers guide in the project material, I started with the most simpliest Network to just prove that the fundamental concepts worked and that the drive.py was able to connect to the simulator ok. The performance however was very poor, but the steering angle changed which proved that the model was able to interact with the simulator.

Following this I went straight to implementing the Nvidia architecture. As described previously, this was an architecture that I had some knowledge of previously.

I split the dataset into a training and validation set, using only centre images from the dataset within validation set, as the steering angles from the side cameras would be modified, and might not be a true representation.

I cropped the images to remove features above the horizon and also the car hood.
I wasn't a big fan of some of the Keras features, I didnt like that it was more difficult to view the preprocessing steps if they are done in Keras. Much prefer performing cropping and resizing in numpy or opencv and being able to visualise easily without building another model in order to do so. However, using Keras appeared to give a significant performance gain over open CV, which I assume is through performing tasks in parallel on the GPU.


#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers.

- 5 x Convolutional laters, each folled by a relu and a Dropout.
- 1 x Flatten
- 4 x Fully connected layers


Extracted from the model.py 

model.add(Convolution2D(24,5,5,subsample= (2,2), input_shape=(66, 200, 3),activation = 'relu', W_regularizer=l2(0.001)))
model.add(Dropout(0.2))
model.add(Convolution2D(36,5,5,subsample= (2,2), activation = 'relu', W_regularizer=l2(0.001)))
model.add(Dropout(0.2))
model.add(Convolution2D(48,5,5,subsample= (2,2), activation = 'relu', W_regularizer=l2(0.001)))
model.add(Dropout(0.2))
model.add(Convolution2D(64,3,3,subsample= (1,1), activation = 'relu', W_regularizer=l2(0.001)))
model.add(Dropout(0.2))
model.add(Convolution2D(64,3,3,subsample= (1,1), activation = 'relu',W_regularizer=l2(0.001)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(1164,W_regularizer=l2(0.001) ))
model.add(Dense(100, W_regularizer=l2(0.001)))
model.add(Dense(50, W_regularizer=l2(0.001)))
model.add(Dense(10, W_regularizer=l2(0.001)))
model.add(Dense(1, W_regularizer=l2(0.001)))


Below is the archiecture as shown in the Nvidia end to end learning papaer.
![dataset](https://github.com/Geordio/Behavioral-Cloning-P3/blob/master/report_images/nvidia_arch.png)




#### 3. Creation of the Training Set & Training Process

After my initial capture of 3 laps of data, I trained it for 5 epochs (unfortunately I didnt note the loses at this point as I was keen to let it drive).
The performance was pretty good as a starting effort, but the car performance was eratic, and would crash.

I decided to visualise the dataset at this stage.

![dataset](https://github.com/Geordio/Behavioral-Cloning-P3/blob/master/report_images/initial_dataset.png)

As can be seen, the vast majority of the images are in the straight ahead position. There are very few inputs greater than 0.5 (12.5 degrees) in either direction, which explains the tendency to understeer.
Note:
full left turn is logged as -1 but displayed as -25.
full right turn is logged as +1 but displayed as +25

Below are some sample images


![images](https://github.com/Geordio/Behavioral-Cloning-P3/blob/master/report_images/center_2017_07_15_15_12_55_604.jpg)
![images](https://github.com/Geordio/Behavioral-Cloning-P3/blob/master/report_images/center_2017_07_15_15_13_04_035.jpg)
![images](https://github.com/Geordio/Behavioral-Cloning-P3/blob/master/report_images/center_2017_07_15_15_24_51_894.jpg)
![images](https://github.com/Geordio/Behavioral-Cloning-P3/blob/master/report_images/shadow.jpg)

Note that although the images appear fairly consistent, the final image above shows a shadow from a tree that could cause issues if there were many occurances, and the texture and road edges of the bridge are unique.

At this point, and looking back at my experience with the Traffic Sign Classifier Project, I decided that data was key.
In order to expand the dataset, I took the following sets:
1. Capture training data from track 2. I found track 2 to be incredibly difficult to drive, but completed around 6 laps.
2. Performing some recovery actions by driving to the edge of the road and recording the action of steering back to the centre.
3. Added the side cameras to the training data with a modified steering input. Originally I tried to work out through maths how much to modify the angle by, but found I was making too many assumptions about separation of cameras and angles of cameras to the horizontal etc, so in the end I used a trial and error method to arrive at 0.25 as a value that gave reasonable performance. In addition, I also did some selective data capture, both doing recovery driving, recording the recovery from a starting point need the edge of the track, and also driving selected sections of the track.

Below are the left centre and right images of a given sample (Track 1)

![images](https://github.com/Geordio/Behavioral-Cloning-P3/blob/master/report_images/left_2017_07_15_15_24_51_894.jpg)
![images](https://github.com/Geordio/Behavioral-Cloning-P3/blob/master/report_images/center_2017_07_15_15_24_51_894.jpg)
![images](https://github.com/Geordio/Behavioral-Cloning-P3/blob/master/report_images/right_2017_07_15_15_24_51_894.jpg)

Below are some sample images from Track 2

![images](https://github.com/Geordio/Behavioral-Cloning-P3/blob/master/report_images/center_2017_07_26_22_55_38_215.jpg)
![images](https://github.com/Geordio/Behavioral-Cloning-P3/blob/master/report_images/center_2017_07_26_22_56_20_694.jpg)
![images](https://github.com/Geordio/Behavioral-Cloning-P3/blob/master/report_images/center_2017_07_26_22_56_28_446.jpg)

As you can see the track is very different, with a centre line, different road edges, and a very different roadside environment.

I visualised the dataset as a histogram again

![images](https://github.com/Geordio/Behavioral-Cloning-P3/blob/master/report_images/both_left_right_added.png)

However, still the dataset has a bias to straight ahead steering, but with additional peaks due to the side cameras.

In order to balance the distribution I performed the following actions:

1. extracted the left and right images from the cvs and adjusted the angles to suit (adding and subtracting an offset of 0.19)
2. Iterating through the angles and splitting into centre and non centre steering (less that 0.05 from centre considered straight). This provided the flexibility to decide how many samples from straight and turning that I would add to my training set
3. Balancing the dataset.
   - I discarded 2 out of 3 straight smaples with straight ahead steering 
   - Using the side cameras meant that I had a large number of samples around the angle values of my offset. I.e, because there was a large number of staright ahead samples, there is therefore a large number of samples centred around the offsets. I therefore discarded half of the samples around the offset value (offset = 0.19, discarded half values between 0.18 and 0.22 on each side.)
4. The generator then creates batches from the balanced data and feeds it the model

(Note: I originally created a generator that took the arrays of straight ahead and turning samples as an input, and then randomly selected samples from each array, inorder to try to create a balanced set, however, I discarded this in favour of the method that I finally used, as I found it easier to visualise and monitor how balanced the set was using by final method.)

The dataplot below shows the distribution of the final dataset. Even after discarding images I still have around 100,000 samples

![images](https://github.com/Geordio/Behavioral-Cloning-P3/blob/master/report_images/final.png)

As you can see the distribution is much more even.

I trained the model over 25 epochs, with the default learning rate using the Adam optimiser.

### Simulation Video
A video of the simulation is included in the repository.
![Video](https://github.com/Geordio/Behavioral-Cloning-P3/blob/master/run.mp4)
The car tends to oscilate around the track, which could be minised my tuning the PIController in drive.py