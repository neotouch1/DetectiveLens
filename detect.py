from ultralytics import YOLO
import cv2

# download model
model = YOLO("yolo11n.pt")

results = model("img/standup-paddleboarding-7210815_1280.jpg")

img = cv2.imread("img/standup-paddleboarding-7210815_1280.jpg")

confidence = 0.0

# processing the result
for result in results:
    boxes = result.boxes
    for box in boxes:
        if int(box.cls[0]) == 0:
            # print(f"Detect person: {box.xyxy}, probability: {box.conf}")

            # extrat frame coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0] # probability

            # draw a frame around a person
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f'Person: {confidence:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
cv2.imshow('Detect image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()