import threading
import cv2
import time
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import torch


visualize_data = {"image": np.array([]), "points": []}
visualize_data_lock = threading.Lock()
visualize_started = False


class StreamIterator:

    def __init__(self, video_source):
        self.start_time = time.time()
        self.cap = cv2.VideoCapture(video_source)
        if not self.cap.isOpened():
            raise RuntimeError("Get video failed.")

    def __next__(self, retry=10):
        global visualize_data, visualize_data_lock
        ret, frame = self.cap.read()
        if not ret and retry > 0:
            return self.__next__(retry=retry-1)
        elif retry <= 0:  # failed
            raise RuntimeError("Video stopped.")
        time_stamp = time.time() - self.start_time
        with visualize_data_lock:
            visualize_data["image"] = np.copy(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame), time_stamp

    def __iter__(self):
        return self


def visualize():
    global visualize_data, visualize_data_lock
    def resize_image_by_ratio(image, scale_percent):
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        return resized_image
    # main circle
    while True:
        with visualize_data_lock:
            data = visualize_data.copy()
        points = data["points"]
        image = data["image"]
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        for index, point in enumerate(points):
            *center, state, score = point
            cv2.putText(image, "{}: {:.2f}".format(index, score), (center[0]+4, center[1]-6), cv2.FONT_HERSHEY_COMPLEX,
                        1, (0, 0, 255) if score < 0.5 else (0, 255, 0), 1.5, cv2.LINE_AA)
            if state == 0:  # head down
                cv2.circle(image, center, 5, (0, 0, 255), -1)
            else:  # state == 1:  # head up
                cv2.circle(image, center, 5, (0, 255, 0), -1)
        image = resize_image_by_ratio(image, 50)
        cv2.imshow("Video", image)
        cv2.waitKey(1)


def start_visualize():
    global visualize_started
    if not visualize_started:
        threading.Thread(target=visualize).start()
        visualize_started = True


def crop_image(image: Image, boxes: torch.Tensor) -> torch.Tensor:
    boxes = boxes.float()
    cropped_images = []
    for box in boxes:
        x, y, w, h = box.tolist()
        cropped_img = image.crop((x-w/2, y-h/2, x+w/2, y+h/2))
        resize_transform = transforms.Resize((192, 192))
        resized_cropped_img = resize_transform(cropped_img)
        cropped_images.append(transforms.ToTensor()(resized_cropped_img))
    return torch.stack(cropped_images)


if __name__ == "__main__":
    video_source = 'rtmp://10.198.246.135:1935/live/8888'
    iterator = StreamIterator(video_source)
    for frame, time_stamp in iterator:
        cv2.imshow('Video Stream', np.array(frame))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
