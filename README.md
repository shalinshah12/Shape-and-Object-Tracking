# Shape-and-Object-Tracking
In this project we are tracking the object and shape using deep learning and OpenCV. For this first we detect the object. For detection of the object we use the pre-trained mobilenet_ssd model. Pre-trained model file (caffemodel) and the configuration file for the model (prototxt) are in the mobilenet_ssd directory. This model can detect many objects (list is mentioned in the code), but for now we are only detecting the bottle. 

For detecting the shape we use the Contours functionality provided by the openCV. Firstly we find the outline representing the shape (star image in shape directory) using "findcontours()" function of cv2. Then whenever we keep looking for contours in our frame which matches the contour of our star image. This way we detect the star shape. 

At the center of the frame we drew a line. Whenever the center of the detected object or shape crosses the line, it shows an CROSS alert. This simple application might be very useful to alert the entry in certain areas.

Pre-trained model and its prototxt file is in the "mobilenet_ssd" directory. 
Image for the shape to be detected is in the "shape" directory (you can replace the star shape image with your desired shape image).  
"output_vid" directory stores the video which shows all the detections and alerts.
obj_shape_line.py is the actual code file.


Command to execute:

python objdetection.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt --model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --output output_vid/demo.mp4

