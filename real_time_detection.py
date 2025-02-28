import cv2
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = "/Users/macbookpro/Desktop/detectron2/models/model_final.pth"  
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6  

cfg.MODEL.DEVICE = "cpu" 
predictor = DefaultPredictor(cfg)

MetadataCatalog.get("face_dataset").set(thing_classes=["face1", "face2", "face3", "face4", "face5", "face6"])  
metadata = MetadataCatalog.get("face_dataset")

cap = cv2.VideoCapture(0)  
print("Webcam initialized.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = frame[:, :, ::-1]

    outputs = predictor(frame_rgb)

    print("Predicted classes:", outputs["instances"].pred_classes)

    v = Visualizer(frame_rgb, metadata, scale=1.0)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    output_frame = v.get_image()[:, :, ::-1]

    cv2.imshow("Real-Time Face Detection", output_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

