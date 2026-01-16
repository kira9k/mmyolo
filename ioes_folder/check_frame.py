import cv2

cap = cv2.VideoCapture('/home/kira9k/ieos/test_human_datasets/videos/mp4/Georg_0.mp4')
frame_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_counter += 1

cap.release()
print(f"Количество кадров: {frame_counter}")
