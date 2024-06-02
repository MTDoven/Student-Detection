import cv2
import time
import numpy as np
from PIL import Image


class StreamIterator:

    def __init__(self, video_source):
        self.start_time = time.time()
        self.cap = cv2.VideoCapture(video_source)
        if not self.cap.isOpened():
            raise RuntimeError("Get video failed.")

    def __next__(self, retry=10):
        ret, frame = self.cap.read()
        if not ret and retry > 0:
            return self.__next__(retry=retry-1)
        elif retry <= 0:  # failed
            raise RuntimeError("Video stopped.")
        time_stamp = time.time() - self.start_time
        return Image.fromarray(frame), time_stamp

    def __iter__(self):
        return self


if __name__ == "__main__":
    video_source = 'rtmp://10.198.246.135:1935/live/8888'
    iterator = StreamIterator(video_source)
    for frame, time_stamp in iterator:
        cv2.imshow('Video Stream', np.array(frame))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
