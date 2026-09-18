import cv2
from ultralytics import YOLO

def expand_box(box, factor):
    (x1, y1, x2, y2) = box

    # Find width of the original box and multiply it by factor, then divide by two
    # to get half, because we are expanding from the centre
    half_width, half_height = ((x2 - x1) * factor) / 2, ((y2 - y1) * factor) / 2
    # Calculate horizontal centre and vertical centre
    centre_x, centre_y = (x1 + x2) / 2, (y1 + y2) / 2

    # Subtract expanded half width and half height from the centre to get the left edge
    x1, y1 = int(centre_x - half_width), int(centre_y - half_height)
    # ... and add the expanded half width and half height from the centre to get the right edge
    x2, y2 = int (centre_x + half_width), int(centre_y + half_height)

    return x1, y1, x2, y2

def pixelate_boxes(boxes, frame, block_size=8):
    frame_height, frame_width = frame.shape[:2]
    blurred_frame = frame.copy()
    
    for box in boxes:
        # The face detection bounding box is exact to the edges of the persons face,
        # we more or less want to blur their whole head to obscure more identifiable features
        (x1, y1, x2, y2) = expand_box(box, 1.5)

        # Clamp coordinates that go off frame (the object detection bounding
        # box can be at the edge of the screen and return negative values)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame_width, x2), min(frame_height, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        roi = frame[y1:y2, x1:x2]
        height, width = roi.shape[:2]
        down_sized = cv2.resize(roi, (block_size, block_size))
        pixelated_box = cv2.resize(down_sized, (width, height))

        blurred_frame[y1:y2, x1:x2] = pixelated_box

    return blurred_frame

def pixelate_results(frame, results):
    boxes = []

    for result in results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            boxes.append((x1, y1, x2, y2))
    
    return pixelate_boxes(boxes, frame)

model = YOLO("yolov11n-face.pt")

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()

    if not success:
        break

    results = model(frame)
    frame = pixelate_results(frame, results)
    
    cv2.imshow("Face Blur", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
