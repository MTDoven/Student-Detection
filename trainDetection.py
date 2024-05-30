from ultralytics import YOLO


if __name__ == '__main__':

    model = YOLO("./detection_yolov9c.yaml")
    model.load("./runs/detect/train18/weights/best.pt")

    model.train(data="detection_data.yaml",
                device="cuda:0",
                epochs=100,
                workers=8,
                batch=16,
                imgsz=640,
                save=True,
                save_period=5,
                cos_lr=True,
                close_mosaic=8,
                warmup_epochs=0,
                resume=False,)
