from ultralytics import YOLO


if __name__ == '__main__':

    model = YOLO("./classifier_yolov8m.yaml")
    model.load("./checkpoints/origin_yolov8m.pt")

    model.train(data=r"..\dataset\ClassifierData",
                device="cuda:0",
                epochs=20,
                workers=8,
                batch=128,
                imgsz=192,
                save=True,
                save_period=4,
                cos_lr=True,
                close_mosaic=4,
                warmup_epochs=0,
                resume=False,)