# Program
The training program is used to predict and segment teeth. The file yolov8n.pt.pt is a pre-trained model. In the training code, model is used to load the pre-trained model, and results specifies the source of the image for segmentation. source indicates the image to be segmented, project specifies the name of the folder to save the results, and name defines the name of the cropped images after segmentation.

Image Processing - Divided into two parts, pre-process and process. In pre-process, images were inhanced by multiple tasks such as  bilateral filter, binarization, masking, grayscale, histogram equalization, Gaussian filters, and Canny Edge Detection. In process, inhanced images undergo masking, pixel value adjust, close operation, color filling, preservation of maximum connectivity region, and superimpose onto the original image.

CNN - Contains five convolutional neural network models and calculates the precision, recall, and F1 score for each class after training.  
Params - Includes the pretrained parameters for the five models as well as the test data obtained after training.  
Testing - A program used to classify user-provided images, returning the most likely class and its probability.  


Few example results:




