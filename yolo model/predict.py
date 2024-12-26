from ultralytics import YOLO
import multiprocessing



# Load a trained model
# model = YOLO(model datapath)
model = YOLO('models/yolov8n.pt.pt')

# Specify the image source and enable cropping
results = model.predict(
    source = 'data/1.jpg',  # Sample image data path
    save_crop = True,       # Whether to enable cropping
    project = 'teeth_1',    # The name of the folder where the results are saved
    name = 'predict_crops'  # The name of the cropped result
)