from ultralytics import YOLO


if __name__ == '__main__':

    model = YOLO("./detection_yolov9c.yaml")
    model.load("./checkpoints/origin_yolov9c.pt")

    model.train(data="detection_data.yaml",
                epochs=100,
                workers=6,
                batch=32,
                imgsz=640,
                save=True,
                save_period=10,
                cache="ram",
                cos_lr=True,
                close_mosaic=10,
                resume=False,)
