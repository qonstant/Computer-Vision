# !python3 -m pip install --upgrade pip
# !pip3 install ultralytics

from ultralytics import YOLO
import cv2
import math 
import torch

# Avaiability of mps
print(torch.backends.mps.is_available())

# Start webcam
cap = cv2.VideoCapture(0)

# Model
model = YOLO("best.pt")

# Custom object classes
classNames = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

while True:
    success, img = cap.read()
    
    # Perform inference
    results = model.predict(img, device="mps")
    # results = model.predict(img, show=True, stream=True, device="mps")

    # Process detections
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # Bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # Convert to int values

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # Confidence
            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence --->", confidence)

            # Class index
            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])

            # Display class name
            text = f"{classNames[cls]}: {confidence}"
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2
            # cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)
            cv2.putText(img, text, org, font, fontScale, color, thickness)

    # Display the webcam feed with detections
    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

# Release webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
