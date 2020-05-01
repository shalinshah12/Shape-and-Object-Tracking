"""
Importing the required libraries
"""
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
"""
When running from the terminal following arguments have to be specified
"""
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
    help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
    help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
    help="minimum probability to filter weak detections")
ap.add_argument("-i", "--shape", type=str, required=True,
    help="path to the image that containes shape")
ap.add_argument("-o", "--output", type=str,
	help="path to optional output video file")
args = vars(ap.parse_args())
"""
Classes here characterizes the objects that this particular code can recognize
"""
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
"""
Loading the prototext and caffemodel
"""
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
"""
initialize the video stream, allow the camera sensor to warmup,
and initialize the FPS counter
"""
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

def nothing(x):
    # any operation
    pass

"""
Defining the trackers for the frame so as to manually change on which color to detect
"""
cap = cv2.VideoCapture(0)
cv2.namedWindow("Trackbars")
cv2.createTrackbar("L-H", "Trackbars", 0, 180, nothing)
cv2.createTrackbar("L-S", "Trackbars", 66, 255, nothing)
cv2.createTrackbar("L-V", "Trackbars", 134, 255, nothing)
cv2.createTrackbar("U-H", "Trackbars", 180, 180, nothing)
cv2.createTrackbar("U-S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U-V", "Trackbars", 243, 255, nothing)
"""
Taking as an input the shape to be detected.
Also finding it contours.
"""
path = args["shape"]
img = cv2.imread(path,0)
ret, thresh = cv2.threshold(img, 127,255,0)
contours, _ = cv2.findContours(thresh, 2,1)
cnt1 = contours[0]
"""
initializing the writer object
"""
writer = None
while True:
    """
    getting the frame from the webcam/ipcam and resizing it to a specific size
    """
    frame = vs.read()
    frame = imutils.resize(frame, width=1000)
    """
    grab the frame dimensions and convert it to a blob
    """
    (h, w) = frame.shape[:2]
    """
    specifying the video format
    """
    if args["output"] is not None and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MP4V")
        writer = cv2.VideoWriter(args["output"], fourcc, 30,
            (w, h), True)
    """
    coverting the fram from BGR to HSV
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    """
    getting the pointer position from the trackbar
    """
    l_h = cv2.getTrackbarPos("L-H", "Trackbars")
    l_s = cv2.getTrackbarPos("L-S", "Trackbars")
    l_v = cv2.getTrackbarPos("L-V", "Trackbars")
    u_h = cv2.getTrackbarPos("U-H", "Trackbars")
    u_s = cv2.getTrackbarPos("U-S", "Trackbars")
    u_v = cv2.getTrackbarPos("U-V", "Trackbars")
    """
    specifying the range of yellow color to be detected
    """
    lower_red = np.array([20, 110, 110])
    upper_red = np.array([40, 255, 255])
    """
    specifying the range of the color dynamically
    """
    lower = np.array([l_h,l_s,l_v])
    upper = np.array([u_h,u_s,u_v])
    """
    the following snippet will show a blacked out frame.
    This frame highlights the portion of the frame where color is detected.
    """
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel)
    font = cv2.FONT_HERSHEY_COMPLEX
    """
    drawing the line at the center
    """
    cv2.line(frame, (0, h//2), (w, h//2), (0, 255, 0), 2)
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
        0.007843, (300, 300), 127.5)
    """
    pass the blob through the network and obtain the detections and predictions
    """
    net.setInput(blob)
    detections = net.forward()
    """
    loop over the detections
    """
    for i in np.arange(0, detections.shape[2]):
        """
        Extracting the confidence
        """
        confidence = detections[0, 0, i, 2]
        """
        removing the unnecesaary detections based on the confidence
        """
        if confidence > args["confidence"]:
            """
            extract the index of the class label from the
            detections`, then compute the (x, y)-coordinates of
            the bounding box for the object
            """
            idx = int(detections[0, 0, i, 1])
            if CLASSES[idx] != "bottle":
                continue
            """
            drawing the detection on the fram using a rectangle border.
            tracking this border to check whether the bottle moves
            beyond the specified limits
            """
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")            
            label = "{}: {:.2f}%".format(CLASSES[idx],
                confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                COLORS[idx], 2)            
            center = startY+endY/2
            if center > h//2:
                cv2.putText(frame, 'BOTTLE CROSSED', (500, h-40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2) 
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)    

    """
    checkng the cv2 version installed in the system
    and getting the contours from the blacked out frame
    """
    if int(cv2.__version__[0]) > 3:
        contours, _ = cv2.findContours(mask, 2, 1)
    else:
        _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    """
    looping over the contours
    """
    for cnt in contours:
        """
        calculating the area
        approximating the points to identify the shapes
        """
        area = cv2.contourArea(cnt)
        approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
        x = approx.ravel()[0]
        y = approx.ravel()[1]
        """
        the shape detected is then compared with the base shape 'star'
        """
        dist = cv2.matchShapes(cnt, cnt1, 1, 0.0)
        """
        if the shape is matched, it will draw the borders around the shape 
        using cv2 and a rectangle box keeping the shape in the center. 
        Also shape tracking is done to check whether it moves 
        beyond a specified limit or not
        """
        if area > 400 and dist < 0.001:
            cv2.drawContours(frame, [approx], 0, (0, 255, 0), 2)
            (a,b,c,d) = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (a,b), (a+c,b+d),(255,0,0),1)
            cv2.putText(frame, "MATCHED", (x, y), font, 1, (255, 0, 0))
            center = (b + b + d)/2
            if center > h//2:
                cv2.putText(frame, 'SHAPE CROSSED', (500, h-40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
    """
    if the writer object is true, it will write the entire tracking into 
    a video file
    """
    if writer is not None:
        writer.write(frame)
    """
    displaying the frame
    """
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    fps.update()
    
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
