from ultralytics import YOLO
import cv2
import json
import time

timestamp = int(time.time())


# for output file 
output_image_path = 'Data/output/img/det_img_{timestamp}.png'
output_detect_data = 'Data/output/data/det_data_{timestamp}.json'



# download model
model = YOLO("yolo11n.pt")

results = model("Data/img/man-2920911_960_720.jpg")

img = cv2.imread("Data/img/man-2920911_960_720.jpg")

# probability
confidence = 0.0

# list of detection data for procesing
detection = []

# processing the result
for result in results:
    boxes = result.boxes
    for box in boxes:
        if int(box.cls[0]) == 0:
            # print(f"Detect person: {box.xyxy}, probability: {box.conf}")

            # extrat frame coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0] # probability

            detection.append({
                "class_id": int(box.cls[0]),
                "confidence": float(box.conf[0]),
                "bbox": [x1, y1, x2, y2]
            })

            # draw a frame around a person
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f'Person: {confidence:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# save image
cv2.imwrite(output_image_path, img)

# save data
with open(output_detect_data, 'w') as f:
    json.dump(detection, f, indent = 4)
          
cv2.imshow('Detect image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
