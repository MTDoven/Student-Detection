from typing import Iterable
from torch import Tensor

from ultralytics import YOLO
from Cluster import k_means
from Logger import Logger
from PIL import Image


config = {
    "id_confirm_steps": 10,
    "reset_id_steps": 10000,
    "log_history_length": 100,
}


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
detection = YOLO("./detection_yolov9c.yaml")
detection.load("./checkpoints/detection_result/weights/best.pt")
detection = detection.to(device)
classifier = YOLO("./classifier_yolov8m.yaml")
classifier.load("./checkpoints/classifier_result/weights/best.pt")
classifier = classifier.to(device)
logger = Logger(log_history_length=config["log_history_length"])




# TODO: !!! We need a iterator for extract video stream !!!
stream: Iterable[Image] = ...
stream_iterator = iter(stream)

if True: # for debug
    image = next(stream_iterator)
    image.show()
# TODO: !!! We need a iterator for extract video stream !!!


def crop_image(image: Tensor, boxes: list[tuple]) -> Tensor:
    # TODO: crop images and resize to (batch * 1 * 192 * 192)
    ...


def confirm_id():
    global logger
    global stream_iterator
    for _ in range(config["id_confirm_steps"]):
        image = next(stream_iterator)
        detection_result = detection(image)
        detection_result = detection_result[0].boxes
    # TODO: to log these boxes for time series analysis and confirm ID


while True:
    try:
        image = next(stream_iterator)
    except StopIteration:
        print("Stream turned off...")
        break
    detection_result = detection(image)
    detection_result = detection_result[0].boxes
    # TODO: to get which ID it belongs to
    cropped_image = crop_image(image, ...)
    predict_label = classifier(cropped_image)
    # TODO: analysis logged information

    if logger.total_number % config["reset_id_steps"] == 0:
        confirm_id()





