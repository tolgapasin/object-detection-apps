import cv2
from ultralytics import YOLO
from datetime import datetime
from collections import deque

# 90 frames, should be about 30 seconds of footage but it depends on how long the YOLO object
# detection takes on each frame so its hard to be exact
frame_buffer_size = 90
frame_buffer = deque(maxlen=frame_buffer_size)

model = YOLO("yolo11n.pt")
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()

    if not success:
        break

    results = model(frame, verbose=False, classes=[0])
    annotated_frame = results[0].plot()

    # Store the last frame, because its a deque with a max size, it will automatically drop older frames
    # when it reaches the max length
    frame_buffer.append(annotated_frame)

    if len(results) > 0:
        time_stamp = datetime.now().isoformat()
        # Windows doesn't allow ':' in file names
        time_stamp = time_stamp.replace(":", ".")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        writer = cv2.VideoWriter(
            f"security-camera/footage/security-footage-{time_stamp}.mp4",
            fourcc, # video format
            10.0, # frame rate (fps)
            (annotated_frame.shape[1], annotated_frame.shape[0]) # width and height
        )

        # TODO: this is writing a new video on every frame, need to figure out a way to until theres more
        # footage before writing
        for frame in frame_buffer:
            writer.write(frame)

        writer.release()

    cv2.imshow("Security Camera", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
