from ultralytics import YOLO
from RtmpStream import StreamIterator, visualize_data_lock, visualize_data, start_visualize, crop_image
from IdentityLogger import IdentityLogger
import torch
import time


config = {
    "id_confirm_steps": 100 * 5,
    "log_history_length": 14400 * 5,
    "process_interval": 5,
}

source_url = 'rtmp://10.198.246.135:1935/live/8888' #r"C:\Users\t1526\Desktop\WeChat_20240602195533.mp4"#
stream_iterator = StreamIterator(source_url)
next(stream_iterator)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
detection = YOLO("./detection_yolov9c.yaml")
detection.load("./checkpoints/detection_result/weights/best.pt")
detection = detection.to(device)
classifier = YOLO("./classifier_yolov8m.yaml")
classifier.load("./checkpoints/classifier_result/weights/best.pt")
classifier = classifier.to(device)
logger = IdentityLogger(log_history_length=config["log_history_length"])


start_visualize()
total_number = 0
while True:
    total_number += 1

    # get images and time stamp
    image, time_stamp = next(stream_iterator)
    if total_number % config["process_interval"] != 0:
        continue

    # predict boxes
    detection_result = detection.predict(image, verbose=False)
    boxes = detection_result[0].boxes.xywh

    # pre-confirm centers
    if total_number < config["id_confirm_steps"]:
        logger.pre_log(boxes)
    elif total_number == config["id_confirm_steps"]:
        logger.pre_log(boxes)
        centers = logger.get_centers()
        center_points = [(*tuple(map(int, point + 0.5)), 1, 0.5) for point in logger.centers]
        with visualize_data_lock:
            visualize_data["points"] = center_points.copy()
        print("finished confirming centers.")

    # main part
    else:  # total_number > config["id_confirm_steps"]
        # confirm state
        cropped_heads = crop_image(image=image, boxes=boxes)
        results = classifier.predict(cropped_heads, verbose=False)
        is_head_up = torch.tensor([result.probs.top1 for result in results])

        # cal ids and score
        logger.log(boxes, is_head_up, time_stamp)
        ids = logger.confirm_ids(boxes)
        time_now = time.time() - stream_iterator.start_time
        scores = logger.get_scores(start_time=time_now - 60, end_time=time_now)

        # log and visualize
        with visualize_data_lock:
            points = visualize_data["points"]
            for index, i in enumerate(ids):
                points[i] = (round((boxes[index][0]+center_points[i][0]).item()/2),
                             round((boxes[index][1]+center_points[i][1]).item()/2),
                             is_head_up[index].item(),
                             scores[i])


