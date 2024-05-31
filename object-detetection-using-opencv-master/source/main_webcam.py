import cv2
import numpy as np
import os

# Set thresholds for detection
thres = 0.45  # Confidence threshold to detect objects
nms_threshold = 0.5  # Non-maximum suppression threshold

# Initialize video capture (webcam)
cap = cv2.VideoCapture(0)
cap.set(3, 1280) #set width
cap.set(4, 720)  #set height
cap.set(10, 150) #set brigthness of video

# Import class names from coco.names
classNames = []
classFile = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'config_files', 'coco.names'))

# Read object classes
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Import the config and weights files
configPath = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'config_files', 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'))
weightsPath = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'config_files', 'frozen_inference_graph.pb'))

# Set up the network
net = cv2.dnn_DetectionModel(weightsPath, configPath)

# Set network parameters
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    # Start webcam
    success, image = cap.read()
    if not success:
        print("Failed to grab frame")
        break

    # Detect objects
    classIds, confs, bbox = net.detect(image, confThreshold=thres)

    if len(classIds) != 0:
        # Prepare bounding box and confidence list for NMS
        bbox = list(bbox)  # NMS function requires bbox as a list, not a tuple
        confs = list(np.array(confs).reshape(1, -1)[0])  # Reshape confidences to be a flat list
        confs = list(map(float, confs))  # Ensure confidences are float

        # Apply non-maximum suppression
        indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)
        
        # Convert indices to a flat list if it's not already
        indices = indices.flatten().tolist() if len(indices) > 0 else []

        # Draw bounding boxes and labels
        for i in indices:
            box = bbox[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            classId = int(classIds[i]) if isinstance(classIds[i], (list, np.ndarray)) else classIds[i]
            cv2.rectangle(image, (x, y), (x + w, y + h), color=(0, 255, 200), thickness=2)
            cv2.putText(image, classNames[classId - 1], (box[0] + 10, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 10), 2)

    # Show output
    cv2.imshow("Output", image)

    # Exit on pressing 'q'
    if cv2.waitKey(3) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()
cv2.destroyAllWindows()
