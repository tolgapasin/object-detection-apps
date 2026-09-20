import cv2
import time
from ultralytics import YOLO
from datetime import datetime
from pathlib import Path
#from collections import deque

# 90 frames, should be about 30 seconds of footage but it depends on how long the YOLO object
# detection takes on each frame so its hard to be exact
frame_buffer_size = 90
frame_buffer = []
Path("security-camera/footage").mkdir(parents=True, exist_ok=True)

model = YOLO("yolo11n.pt")
cap = cv2.VideoCapture(0)

total_person_count = recent_person_count = recordings_saved = 0

start = time.time()

while cap.isOpened():
    success, frame = cap.read()
    time_buffer = time.time()

    if not success:
        break

    results = model(frame, verbose=False, classes=[0])
    annotated_frame = results[0].plot()

    # Store the last frame, because its a deque with a max size, it will automatically drop older frames
    # when it reaches the max length
    frame_buffer.append(annotated_frame)

    num_people_detected = len(results)
    if num_people_detected == 0: recent_person_count = 0
    if num_people_detected > total_person_count: total_person_count = num_people_detected
    person_count_increased = num_people_detected > recent_person_count

    # TODO: its only recording for like a second, fix this
    # TODO: it says 1 people detected when it starts up
    print(time_buffer)
    if person_count_increased and time_buffer > 30:
        time_stamp = datetime.now().isoformat()
        # Windows doesn't allow ':' in file names
        time_stamp_formatted = time_stamp.replace(":", ".")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        writer = cv2.VideoWriter(
            f"security-camera/footage/security-footage-{time_stamp_formatted}.mp4",
            fourcc, # video format
            10.0, # frame rate (fps)
            (annotated_frame.shape[1], annotated_frame.shape[0]) # width and height
        )

        for frame in frame_buffer:
            writer.write(frame)

        writer.release()
        recordings_saved += 1 
        print(f"{num_people_detected} people detected at {time_stamp}")
        time_buffer = 0

    cv2.imshow("Security Camera", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

end = time.time()
total_time = end - start
print(f"Ran for {total_time}, {total_person_count} people detected, {recordings_saved} recordings saved.")

cap.release()
cv2.destroyAllWindows()
