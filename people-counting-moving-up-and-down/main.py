import cv2
import numpy as np
from collections import OrderedDict
from ultralytics import YOLO

# ========================
# Centroid Tracker
# ========================
class CentroidTracker:
    def __init__(self, max_disappeared=50):
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.labels = {}  # track label for each object_id
        self.max_disappeared = max_disappeared

    def register(self, centroid, label):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.labels[self.next_object_id] = label
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]
        del self.labels[object_id]

    def update(self, input_centroids, input_labels):
        if len(input_centroids) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects, self.labels

        if len(self.objects) == 0:
            for centroid, label in zip(input_centroids, input_labels):
                self.register(centroid, label)
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids, axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.labels[object_id] = input_labels[col]
                self.disappeared[object_id] = 0
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(D.shape[0])) - used_rows
            unused_cols = set(range(D.shape[1])) - used_cols

            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            for col in unused_cols:
                self.register(input_centroids[col], input_labels[col])

        return self.objects, self.labels



model = YOLO("kdandadult-model.pt")

CLASS_NAMES = ["Kid", "Adult"]


cap = cv2.VideoCapture(0)  # webcam

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 20

out = cv2.VideoWriter('people_counting_output.mp4',
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      fps,
                      (frame_width, frame_height))


counts = {
    "Kid_in": 0,
    "Kid_out": 0,
    "Adult_in": 0,
    "Adult_out": 0
}

ct = CentroidTracker()
previous_x = {}
line_position = frame_width // 2  # vertical middle line

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame, conf=0.5)
    centroids = []
    labels = []

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        for box, conf, cls in zip(boxes, confs, classes):
            x1, y1, x2, y2 = map(int, box)
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            centroids.append((cx, cy))
            labels.append(int(cls))

            label_name = CLASS_NAMES[int(cls)]
            display_text = f"{label_name} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, display_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Update tracker with centroids + labels
    objects, tracked_labels = ct.update(centroids, labels)

    # Count logic per class
    for (object_id, centroid) in objects.items():
        current_x = centroid[0]
        label = CLASS_NAMES[tracked_labels[object_id]]
        if object_id in previous_x:
            prev_x = previous_x[object_id]
            if prev_x > line_position and current_x < line_position:
                counts[f"{label}_out"] += 1
                print(f"{label} {object_id} moved LEFT (keluar).")
            elif prev_x < line_position and current_x > line_position:
                counts[f"{label}_in"] += 1
                print(f"{label} {object_id} moved RIGHT (masuk).")
        previous_x[object_id] = current_x

    # Draw middle line
    cv2.line(frame, (line_position, 0), (line_position, frame_height), (0, 0, 255), 2)

    # Show counts on screen
    cv2.putText(frame, f"Kids In: {counts['Kid_in']}  Out: {counts['Kid_out']}", 
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, f"Adults In: {counts['Adult_in']}  Out: {counts['Adult_out']}", 
                (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Kids & Adults Counting", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Final Counts:", counts)

