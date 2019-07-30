# **Behavioral Cloning**

## Writeup Template

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

The project submission includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* project_report.md containing the writeup for this project

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is based on the model described in this [NVIDIA's paper](https://arxiv.org/pdf/1604.07316.pdf) titled *End to End Learning for Self-Driving Cars*. It was chosen because  it has an objective that is similar to the objective of the current project (i.e., training a neural network on images captured from a car). The figure below provides the structure of the NVIDIA model as described [here](https://arxiv.org/pdf/1604.07316.pdf):

<p align="center">
<img src="report_images/nvidia_architecture.png" width="60%" alt>
</p>
<p align="center">
<em> NVIDIA network architecture ([source](https://arxiv.org/pdf/1604.07316.pdf))
</p>

The above model was modified for the present project, as described later in this report.


#### 2. Attempts to reduce overfitting in the model

The first model that was tested was similar to the NVIDIA model which used three 5x5 convolutional layers and two 3x3 convolutional layers. However, while the model
generated using this architecture was successful on the original track with resolution
and speed (fastest) setting, it did not work as well for the same track but an alternate
visualization setting.  The new visualization technique merely included better features
such as shadows of objects. The fact that the model did not work slight modifications in
the data, shows that it is overfitting to the original training data. Also, the model had too many parameters (95,471,419) and was perhaps too heavy for the problem it was trying to solve.

One way to reduce overfitting is to reduce the size of the neural network. As a first step,
the last convolutional layer was removed. The model obtained from this architecture also
cleared the track with the original visualization setting but failed at more advanced
visualization settings. Next, the last two convolutional layers were removed. With this
setting the model cleared all the visualization settings. This was chosen to be the final configuration since eliminating further convolutional layers made the model perform poorly
on the track. Compared to the original model (parameters), the final model had only 78,427,579 parameters (i.e., 18% reduction).

#### 3. Model parameter tuning

The model used an Adam optimizer, so the learning rate was not tuned manually. The default learning rate for the Adam optimizer is 0.001. The number of epochs was kept at 2 to
prevent overfitting.

#### 4. Appropriate training data

The training data used to train the model was wholly based on the dataset that was
provided as part of this project. Details of how this data was augmented is described
later in this report.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the [NVIDIA model](https://arxiv.org/pdf/1604.07316.pdf). This model was felt to be appropriate because because it solves a very similar problem. This model was adjusted for overfitting by
dropping convolutional layers till performance was affected.

The model was trained initially on the center camera data and did not perform poorly. Through pre-processing (cropping, normalization, etc.) and data augmentation (flipping adding additional camera views) and adjusting the neural network the model was improved till it
met the desired criteria.
At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes:

| Layer         		|     Description	        					|
|-----------------------|-----------------------------------------------|
| Input         		| 160x320x3 RGB image   							|
| Lambda      	| Normalize input as (image - 128.0)/128.0 	|
| Cropping      	| 70 pixels from top and 25 pixels from bottom , output (65,320,3)	|
| Convolution 5x5	    |  1x1 stride, valid padding, output = (61,316,24) |
| RELU					|												|
| Convolution 5x5	    |  1x1 stride, valid padding, output=(57,312,16) |
| RELU					|												|
| Convolution 5x5	    |  1x1 stride, valid padding, output=(53,308,48) |
| RELU					|												|
| Convolution 3x3	    |  1x1 stride, valid padding, output = (53,308,48) |
| RELU					|												|
| Convolution 3x3	    |  1x1 stride, valid padding, output |
| RELU					|												|
| Convolution 3x3	    |  1x1 stride, valid padding, output |
| RELU					|												|
| Flatten		| 53x308x48 = 783552 |
| Fully connected		| output = 100 |
| RELU					|												|
| Fully connected		| output = 50 |
| RELU					|												|
| Fully connected		| output = 10 |
| RELU					|												|
| Output		| 1 node        									|

Since this is a regression problem, the loss function was chosen to be mean square error.
Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)


<p align="center">
<img src="report_images/final_model_architecture_no_bg.png" width="100%" alt>
</p>
<p align="center">
<em> Sample image before (left) and after (right) adding noise
</p>

#### 3. Creation of the Training Set & Training Process

Without a joystick, manually generating new data by traveling through the track
using mouse/keyboard controls was challenging. Fortunately,
there was some training data provided as part of the project
and this was found to be adequate to train the model to
successfully meet the criteria.

The original data set had totally 24108 images. Because the track has a an anti-clockwise bias, all the images were flipped about the vertical axis to remove
the bias. The bash command for flipping an image is provided below:
```
# Note: The ImageMagick package must be installed for this command to work
convert -flop input_image.jpg flipped_image.jpg
```

This increased the size of the dataset by a factor of two. The steering measurement
for the flipped image is obtained by changing the sign of the steering value for the
original image. Shown below are sample camera images before and after flipping:

<p align="center">
<img src="report_images/left_example.jpg" width="32%" alt>
<img src="report_images/center_example.jpg" width="32%" alt>
<img src="report_images/right_example.jpg" width="32%" alt>
</p>
<p align="center">
<em> Left, center and right images before flipping
</p>

<p align="center">
<img src="report_images/left_flip.jpg" width="32%" alt>
<img src="report_images/center_flip.jpg" width="32%" alt>
<img src="report_images/right_flip.jpg" width="32%" alt>
</p>
<p align="center">
<em> Left, center and right images after flipping
</p>

For the model to predict the correct steering angle, only some portions of the image are relevant. For the most part, the top part of the image typically contains background information such as trees, sky, mountain, etc. This is not critical for making steering
control decisions. Also, the bottom part of the image contains part of the dashboard which
again, is irrelevant for making steering decisions. The dashboard adds noise to the image,
especially because it makes the location of the dashboard is different for different cameras (left, right and center). Therefore, it makes sense to crop out a part of the lower pixels
in the image as well. As part of the pre-processing step, 70 pixels were removed from
the top and 25 pixels were removed from the bottom.

The bash command for cropping an image (70 pixels at the top and 25 pixels at the bottom  is provided below:
```
```

<p align="center">
<img src="report_images/left_example.jpg" width="32%" alt>
<img src="report_images/center_example.jpg" width="32%" alt>
<img src="report_images/right_example.jpg" width="32%" alt>
</p>
<p align="center">
<em> Left, center and right images before cropping
</p>

<p align="center">
<img src="report_images/left_cropped.jpg" width="32%" alt>
<img src="report_images/center_cropped.jpg" width="32%" alt>
<img src="report_images/right_cropped.jpg" width="32%" alt>
</p>
<p align="center">
<em> Left, center and right images after cropping
</p>

It is seen that before cropping, it is easy to distinguish between the left, center
and right cameras whereas after cropping this is not easy because the dashboard has
been cropped off.

Since the steering angle provided with the pre-existing dataset corresponds to the
center camera, the first attempt at training the model used only the data from the
center camera. However, the model trained only on the center camera performed poorly.
The figure below shows the distribution training samples across different steering
angles:
<p align="center">
<img src="report_images/histogram_center_camera_only.png" width="75%" alt>
</p>
<p align="center">
<em> Histogram of center camera data alone
</p>

It can be seen that most of the data is available only for low steering angles.
This may bias the model to prefer low steering angles even when higher steering angles
may be required. To remove this bias, an attempt was made to augment the center camera
data by adding more images. In this approach, images that had low steering angles (< 0.2)
were not augmented while images with large steering angles were augmented proportionate
the magnitude of the steering angle. The function for creating an augmented image by
adding noise is provided below:

```python
def add_noise(input_image, mean=0, var=10):
    sigma = var ** 0.5
    gaussian = np.random.normal(mean, sigma, input_image.shape)
    noisy_image = np.zeros(input_image.shape, np.float32)
    noisy_image[:, :, :] = input_image[:, :, :] + gaussian
    cv2.normalize(noisy_image, noisy_image, 0, 255, cv2.NORM_MINMAX, dtype=-1)
    noisy_image = noisy_image.astype(np.uint8)

    return noisy_image
    ```
The histogram obtained after augmenting the center camera data is provided below:
<p align="center">
<img src="report_images/histogram_center_camera_augmentation.png" width="75%" alt>
</p>
<p align="center">
<em> Histogram of center camera data after augmentation
</p>

However, despite the augmentation, the newly trained model still performed poorly, especially
at sharp turns. It was felt that the center camera data alone was not enough to train
the model to work successfully and that use of the images from the left and right cameras was critical
for the model to work. Some reasons why the left and right images were necessary
are listed below:

* The left and right cameras provide a lot more training data without artificial image
augmentation.

* The center camera only captures views of the road when driver is driving optimally. If
the car veers to the edge of the road, it does not contain sufficient data on how
to recover from such situations. The left and right cameras on the other hand, contain
perspectives of the road that would be seen from the center camera if the car veers of
to the edges of the road. Therefore, by including images from the left and right cameras
and attributing suitable steering values for these images, we provide training on how
to recover from mistakes and edge behavior.

* Since the steering angles for the left and right cameras are obtained by adding a correction
factor to the center camera steering angle, the distribution of samples across steering
angles is improved by increasing the number samples with higher steering angles.

However, one challenge when including left and right camera images is that when the model is tested, the data provided to the model
is only the center camera image. Therefore, the steering angles used for the left and right camera images during training must be adjusted to make it correspond to that of a center camera. A steering correction factor is used to assign a suitable value for the left
and right camera images based on the value for the corresponding center camera image. The
steering correction factor is applied as shown in the code snippet below:

```python
steering_left = steering_center + steering_correction
steering_right = steering_center - steering_correction
```

The steering correction factor is a hyper-parameter that needs to be tuned. For this project,
a value of 0.1 was found to work well. Provided below is an example image of left, center and
right lane driving:

<p align="center">
<img src="report_images/left_example.jpg" width="32%" alt>
<img src="report_images/center_example.jpg" width="32%" alt>
<img src="report_images/right_example.jpg" width="32%" alt>
</p>
<p align="center">
<em> Sample image before (left) and after (right) adding noise
</p>

Including the left and right camera images and including flipping, increased the size of
the dataset to 48,324. This was randomly shuffled and 80% was used as training data and 20% was used as validation data.
