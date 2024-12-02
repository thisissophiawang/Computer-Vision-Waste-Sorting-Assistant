import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model=YOLO('yolo11m.pt')
    model = YOLO(r'E:\an\ultralytics-main\ultralytics\cfg\models\11\yolo11.yaml')  
    model.train(data=r'E:/an/ultralytics-main/rubbish.yml',
                cache=False,
                epochs=400,
                single_cls=False,  # 是否是单类别检测 whether single-class object detection
                batch=32,
                close_mosaic=10,
                workers=0,
                device='0',
                optimizer='SGD',
                amp=True,
                project='runs/train',
                name='exp',
                )
