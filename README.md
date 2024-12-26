# Program
Program Usage Flow:  
![image](https://github.com/jojowang234/bioengineering3342859/blob/main/Program%20Usage%20Flowchart.png)  
First, use the pre-trained YOLO model to predict and segment tooth images. In the folder named "YOLO Model," you'll find the program files for tooth segmentation. The file yolov8_model.pt is the trained model file, while the predict file is the script for tooth segmentation. The example folder contains sample images, and the after folder contains the segmented sample images.  
After running the tooth segmentation script from the yolov8_model, the images are then fed into the image_inhancement.m file in the “Image Processing” folder for further image processing.  
Next, the images enhanced through image augmentation are trained using five convolutional neural network models located in the “CNN” folder, and the precision, recall, and F1 score for each class are obtained after training. And “Params” folder includes the pretrained parameters for the five models as well as the test data obtained after training.To verify the accuracy of the training, we then place the trained model into the Testing program in the “Testing” folder for validation using an unseen dataset, ultimately obtaining the validation accuracy.  

Result:  
The training program is used to predict and segment teeth. The file yolov8n.pt.pt is a pre-trained model. In the training code, model is used to load the pre-trained model, and results specifies the source of the image for segmentation. source indicates the image to be segmented, project specifies the name of the folder to save the results, and name defines the name of the cropped images after segmentation. The segmented images were not pre-processed to a uniform size; instead, their proportions were constrained at a later stage.

Before segmentation:  
![image](https://github.com/jojowang234/bioengineering3342859/blob/main/yolo%20model/example.jpg)  

After segmentation:  
![image](https://github.com/jojowang234/bioengineering3342859/blob/main/yolo%20model/afer.jpg)  
Image Processing - Divided into two parts, pre-process and process. In pre-process, images were inhanced by multiple tasks such as  bilateral filter, binarization, masking, grayscale, histogram equalization, Gaussian filters, and Canny Edge Detection. In process, inhanced images undergo masking, pixel value adjust, close operation, color filling, preservation of maximum connectivity region, and superimpose onto the original image.

Median Filter:  
![image](https://github.com/jojowang234/bioengineering3342859/blob/main/Image%20Processing/Medain%20Filter.png)  
Bilateral Filter:  
![image](https://github.com/jojowang234/bioengineering3342859/blob/main/Image%20Processing/Bilateral%20Filter.jpg)  
Histogram equalization:  
![image](https://github.com/jojowang234/bioengineering3342859/blob/main/Image%20Processing/Histogram%20equalization.png)  
Enhancement of Masked:  
![image](https://github.com/jojowang234/bioengineering3342859/blob/main/Image%20Processing/Enhancement%20of%20Masked%20Image.png)  
Close Operation:  
![image](https://github.com/jojowang234/bioengineering3342859/blob/main/Image%20Processing/Close%20Operation.png)  
Canny edge detection:  
![image](https://github.com/jojowang234/bioengineering3342859/blob/main/Image%20Processing/Canny%20edge%20detection.png)  
CNN - Contains five convolutional neural network models and calculates the precision, recall, and F1 score for each class after training.  
Params - Includes the pretrained parameters for the five models as well as the test data obtained after training.  
Testing - A program used to classify user-provided images, returning the most likely class and its probability.  

Validation accuracy:  
![image](https://github.com/jojowang234/bioengineering3342859/blob/main/CNN/Validation%20accuracy.jpg)  
Validation loss function:  
![image](https://github.com/jojowang234/bioengineering3342859/blob/main/CNN/Validation%20loss%20function.jpg)  


