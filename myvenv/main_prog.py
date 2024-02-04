from ultralytics import YOLO
import numpy as np
import cv2
from ultralytics import FastSAM
from ultralytics.models.fastsam import FastSAMPrompt
import collections as coll

detecion_model = YOLO('yolov8n.pt')
segmentation_model = FastSAM('FastSAM-x.pt')

cap = cv2.VideoCapture(0)

frames = coll.deque()

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        continue

    detection = detecion_model(source = frame, conf = 0.4, show = False, save = False)
    
    x1, y1, x2, y2 = list(map(float, detection[0].boxes[0].xyxy.cpu().numpy()[0]))

    everything_results = segmentation_model(source = frame, device= 'cuda:0', retina_masks=True, imgsz=640, conf=0.4, iou=0.9)

    prompt_process = FastSAMPrompt(frame, everything_results, device='cuda:0')

    ann = prompt_process.box_prompt(bbox=[x1, y1, x2, y2])

    curr_frame_with_mask = np.asarray((np.array(ann[0].masks[0].data[0]).reshape(480 * 640, 1) * np.reshape(frame, (480*640, 3))).reshape(480, 640, 3), dtype=np.uint8)

    frames.append(curr_frame_with_mask)

    if len(frames) > 5:
        frames.popleft()
        difference = cv2.absdiff(frames[4], frames[0])
        threshold = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)[1]
        cv2.imshow("cam1", threshold)

        if threshold.sum() > 10**6:
            print("Movement")
    
    
    cv2.imshow("cam2", curr_frame_with_mask)
    
    if cv2.waitKey(10) == 27:
        cap.release()

cv2.destroyAllWindows