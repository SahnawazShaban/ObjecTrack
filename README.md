# Object-Detection using OpenCV
Object Detection using OpenCV
OpenCV offers methods for object detection using various models, providing confidence thresholds and bounding box coordinates. It can also apply Non-maximum Suppression (NMS) to avoid multiple detections of the same object, keeping the detection with the highest confidence.

Example detections <br>
![Test](https://github.com/SahnawazShaban/ObjecTrack/blob/main/object-detetection-using-opencv-master/sample/Sample_Detection_image.png)


## Credit to Murtazahassan for his tutorial on Object Detection 
- https://www.murtazahassan.com/courses/opencv-projects/lesson/code-and-files/.
- https://www.youtube.com/watch?v=HXDD7-EnGBY
<br>
<br>
<br>

<u>Components</u></br>
coco.names: Contains names of all objects the model can detect.</br>
frozen_inference_graph.pb: The weights file storing the trained model parameters.</br>
ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt: Configuration file with parameters for the SSD MobileNet model.</br>
