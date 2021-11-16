# FREGAN-Frame-Rate-Enhancement-using-GANs 

## Overview
<p align="justify">
A GAN based model that attempts to improve the frame rate by generating new frame between two frames. FREGAN (Frame Rate Enhancement Generative Adversarial Network) which predicts future frames of a video sequence based on a sequence of past frames.
</p>
<p align="justify">
An approach towards frame rate enhancement is Frame interpolation, it is the action of generating a frame for a video, given the immediate frames occurring sequentially before and after. This allows a video to have its frame rate enhanced, which is a process known as upsampling. In general upsampling, we can’t assume access to the ground truth for the interpolated frames. But the problem with this method is blurry interpolated images do not fit with the overall style of the videos, and are easily detected by the human eye. Multiple methods have proven that Generative Adversarial Networks(GANs) perform relatively better in the study of frame representation learning
</p>

<!-------Network architecture of FREGAN-------->
## Network architecture of FREGAN ##


### Generator
<p align="justify">
The generator model used in the proposed approach takes 2 subsequent frames and predicts the intermediate frame based on these inputs. The generator consists of an
encoding block and decoder block which make it able to predict X<sub>n+1</sub><sup>th</sup> frame of size 256,256 pixels from an array of 2 frames [X<sub>n</sub>, X<sub>n+2</sub>], both of size 256,256 pixels.The encoder block consists of sets of downsampling layers as shown, which include in the order, a 2D convolution operation, a batch normalization operation, and a LeakyReLU activation.The decoder block consists of sets of upsampling layers which use 2D convolutional transpose for upsampling operation.

![Generator Architecture](https://github.com/naveenrb19/FREGAN-Frame-Rate-Enhancement-using-GANs/blob/main/Images/FREGAN_gen.PNG)
</p>

### Discriminator
<p align="justify">
The discriminator model uses a multilayer convolutional neural network to predict whether the input image is fake or real.

  Note:In the research paper batchnorm layer has not been used but I added as it provides better stability for training GANs. Also I have used patch GAN instead of single output value model as shown above. For patch GAN check pix2pix model.

![Discriminator Architecture](https://github.com/naveenrb19/FREGAN-Frame-Rate-Enhancement-using-GANs/blob/main/Images/FREGAN_disc.PNG)
</p>

<!--------------->
## Overall flow of the network 
<p align="justify">
The generator is accepted as input [X<sub>n</sub>, X<sub>n+2</sub>] where X<sub>n</sub> and X<sub>n+2</sub> are the nth and n+2nd
frames as the source input array and the discriminator takes X<sub>n+1</sub> and X'<sub>n+1</sub> as the
testing image, where Xn+1 is the real image and the X’n+1 is G(X<sub>n</sub>, X<sub>n+2</sub>) the
discriminator outputs the log loss, D(X) for real image and D(X') for fake image, due to
the log loss the score lies between 0 and 1, with 0 being real and 1 being fake.

Note:The paper uses log loss for discriminator and Huber loss for generator. But since SmoothL1Loss is able to behave as both huber loss and L1 loss(pix2pix) I have used SmoothL1Loss with beta=0.5.
![ Overall flow of the network](https://github.com/naveenrb19/FREGAN-Frame-Rate-Enhancement-using-GANs/blob/main/Images/FREGAN_overall_flow.PNG)
</p>

<!-------------------->
## Training process

### Dataset
<p align="justify">
  The dataset I used to train the model is entirely different from the dataset used in the paper. I downloaded 2 minute clips of various movies of all genres including animated ones. I used opencv to conver those videos into images in seperate folder. I sampled 999 images from each video 
</p>

### Training
<p align="justify">
 In training loop all images from one folder are ingested split into three lists i.e list1 for every x frame, list2 for every x+1 frames and list3 for every x+2 frames. Batch of 9 images from list1 and list2 are sampled and concatenated to be used as input for the generator that generates fake batch. The generated fake batch and list2 are used to train the discriminator to identify real and fake images. I also implemented label smoothing so that real and fake labels lie between (0.7,1.1) and (0,0.3) respectively since it is said to help train GANs.
  
  
Note: I have trained upto 100 epochs which means architecture looped through 80 folders each with 999 images 100 times. Results are visible but still there is huge improvement required.  
 </p>


<!------------------------->
## Results

### Normal video result

| Frame no:1 | Frame no:2(Generated) | Frame no:3 |
|------------|-----------------------|------------|
|![1](https://github.com/naveenrb19/FREGAN-Frame-Rate-Enhancement-using-GANs/blob/main/Images/1.jpg)|![2](https://github.com/naveenrb19/FREGAN-Frame-Rate-Enhancement-using-GANs/blob/main/Images/2.jpg)|![3](https://github.com/naveenrb19/FREGAN-Frame-Rate-Enhancement-using-GANs/blob/main/Images/3.jpg)|

### Normal video result (in between different shots)

| Frame no:1 | Frame no:2(Generated) | Frame no:3 |
|------------|-----------------------|------------|
|![4](https://github.com/naveenrb19/FREGAN-Frame-Rate-Enhancement-using-GANs/blob/main/Images/4.jpg)|![5](https://github.com/naveenrb19/FREGAN-Frame-Rate-Enhancement-using-GANs/blob/main/Images/5.jpg)|![6](https://github.com/naveenrb19/FREGAN-Frame-Rate-Enhancement-using-GANs/blob/main/Images/6.jpg)|

### Animated movie result

| Frame no:1 | Frame no:2(Generated) | Frame no:3 |
|------------|-----------------------|------------|
|![10](https://github.com/naveenrb19/FREGAN-Frame-Rate-Enhancement-using-GANs/blob/main/Images/10.jpg)|![11](https://github.com/naveenrb19/FREGAN-Frame-Rate-Enhancement-using-GANs/blob/main/Images/11.jpg)|![12](https://github.com/naveenrb19/FREGAN-Frame-Rate-Enhancement-using-GANs/blob/main/Images/12.jpg)|

### Normal video result (Dark shots)

| Frame no:1 | Frame no:2(Generated) | Frame no:3 |
|------------|-----------------------|------------|
|![13](https://github.com/naveenrb19/FREGAN-Frame-Rate-Enhancement-using-GANs/blob/main/Images/13.jpg)|![14](https://github.com/naveenrb19/FREGAN-Frame-Rate-Enhancement-using-GANs/blob/main/Images/14.jpg)|![15](https://github.com/naveenrb19/FREGAN-Frame-Rate-Enhancement-using-GANs/blob/main/Images/15.jpg)|


<!--------->
## Contact

[LinkedIn](https://www.linkedin.com/in/naveenrb)

[Facebook](https://www.facebook.com/naveenraj0398)

<p align="justify">
  Let me know if any changes could be made to improve the model. 
 </p>



