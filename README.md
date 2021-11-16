# FREGAN-Frame-Rate-Enhancement-using-GANs 
<p align="justify">
A GAN based model that attempts to improve the frame rate by generating new frame between two frames. FREGAN (Frame Rate Enhancement Generative Adversarial Network) which predicts future frames of a video sequence based on a sequence of past frames.
</p>
<p align="justify">
An approach towards frame rate enhancement is Frame interpolation, it is the action of generating a frame for a video, given the immediate frames occurring sequentially before and after. This allows a video to have its frame rate enhanced, which is a process known as upsampling. In general upsampling, we can’t assume access to the ground truth for the interpolated frames. But the problem with this method is blurry interpolated images do not fit with the overall style of the videos, and are easily detected by the human eye. Multiple methods have proven that Generative Adversarial Networks(GANs)[2] perform relatively better in the study of frame representation learning
</p>

<!-------Network architecture of FREGAN-------->
## Network architecture of FREGAN ##

<!-------------Generator-------------->
### Generator
<p align="justify">
The generator model used in the proposed approach takes 2 subsequent frames and predicts the intermediate frame based on these inputs. The generator consists of an
encoding block and decoder block which make it able to predict Xn+1th frame of size 256,256 pixels from an array of 2 frames [Xn, Xn+2], both of size 256,256 pixels.The encoder block consists of sets of downsampling layers as shown, which include in the order, a 2D convolution operation, a batch normalization operation, and a LeakyReLU activation.The decoder block consists of sets of upsampling layers which use 2D convolutional transpose for upsampling operation.

![Generator Architecture](FREGAN-Frame-Rate-Enhancement-using-GANs/Images/FREGAN_gen.PNG)
</p>
