from ultralytics import YOLO
import multiprocessing



# 加載已訓練的模型
model = YOLO('models/yolov8n.pt.pt')

# 指定圖片來源，並啟用裁剪功能
results = model.predict(
    source = 'data/1.jpg',  # 單張圖片或目錄
    save_crop = True,       # 啟用裁剪，保存裁剪的結果
    project = 'teeth_1',    # 保存結果的專案名稱
    name = 'predict_crops'  # 結果名稱
)