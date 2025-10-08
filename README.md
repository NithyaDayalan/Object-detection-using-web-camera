# WORKSHOP-2-Object-detection-using-web-camera

### AIM :
To perform real-time object detection using a trained YOLO v4 model through your laptop camera.

### PROGRAM :
```
import cv2, numpy as np, os
w, c, n = "yolov4.weights", "yolov4.cfg", "coco.names"
for f in [w, c, n]:
    if not os.path.exists(f): raise FileNotFoundError(f"Missing: {f}")
net = cv2.dnn.readNet(w, c)
layers = [net.getLayerNames()[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
classes = open(n).read().splitlines()
cap = cv2.VideoCapture(0)
while cap.isOpened():
    r, f = cap.read()
    if not r: break
    h, wd = f.shape[:2]
    blob = cv2.dnn.blobFromImage(f, 0.00392, (416, 416), (0, 0, 0), True)
    net.setInput(blob)
    for o in net.forward(layers):
        for d in o:
            s = d[5:]; i = np.argmax(s)
            if s[i] > 0.5:
                cx, cy, w_, h_ = (d[:4] * [wd, h, wd, h]).astype(int)
                x, y = cx - w_//2, cy - h_//2
                cv2.rectangle(f, (x, y), (x+w_, y+h_), (0,255,0), 2)
                cv2.putText(f, f"{classes[i]} {s[i]:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    cv2.imshow("YOLOv4", f)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
cap.release(); cv2.destroyAllWindows()

```

### OUTPUT :

<img width="809" height="607" alt="Screenshot 2025-10-07 155052" src="https://github.com/user-attachments/assets/5b09ded7-2854-43ef-93a4-dace1f6df26a" />

### RESULT :
The real-time object detection using a trained YOLO v4 model through your laptop camera is executed and performed successfully.
