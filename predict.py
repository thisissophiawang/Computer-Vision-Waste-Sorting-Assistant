import warnings
import csv
import os
from ultralytics import YOLO

warnings.filterwarnings('ignore')

def save_predictions_to_csv(predictions, csv_file):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # write header
        writer.writerow(['ID', 'image_id', 'class_id', 'x_min', 'y_min', 'width', 'height'])
        
        # write each prediction
        id_counter = 1
        for pred in predictions:
            print(f"Processing image: {pred.path}")  # 调试信息 debug info
            image_name = os.path.splitext(os.path.basename(pred.path))[0]  # 获取图像名 get image name
            boxes = pred.boxes
            if boxes is not None:
                for box in boxes:
                    xyxy = box.xyxy[0]  # 获取坐标 get coordinates
                    cls = box.cls[0]  # 获取类别 get class

                    # 互换 class_id 0 和 1 swap class_id 0 and 1
                    if cls.item() == 0:
                        cls = 1
                    elif cls.item() == 1:
                        cls = 0

                    writer.writerow([
                        id_counter,  # ID
                        image_name,  # image_id
                        int(cls),  # class_id
                        int(xyxy[0].item()),  # x_min
                        int(xyxy[1].item()),  # y_min
                        int(xyxy[2].item() - xyxy[0].item()),  # width
                        int(xyxy[3].item() - xyxy[1].item())  # height
                    ])
                    id_counter += 1

        # 填充ID至4999 fill ID to 4999
        while id_counter <= 4999:
            writer.writerow([
                id_counter,  # ID
                9999,        # image_id
                9,           # class_id
                0,           # x_min
                0,           # y_min
                0,           # width
                0            # height
            ])
            id_counter += 1

if __name__ == '__main__':
    model = YOLO('best.pt')
    predictions = model.predict(source='E:/项目/杂草识别/test/images',
                                imgsz=580,
                                device='0',
                                save=True)
    
    # 保存预测结果到CSV文件 save predictions to CSV file
    save_predictions_to_csv(predictions, 'submission.csv')