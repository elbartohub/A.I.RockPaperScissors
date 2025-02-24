import cv2
import numpy as np
import os

# 建立儲存資料的目錄
def create_directories():
    for label in ['rock', 'paper', 'scissors']:
        if not os.path.exists(f'data/{label}'):
            os.makedirs(f'data/{label}')

# 收集訓練資料的主要函數
def collect_data():
    # 初始化攝像頭
    cap = cv2.VideoCapture(0)
    labels = ['rock', 'paper', 'scissors']
    current_label = 0
    count = 0
    
    create_directories()

    while True:
        # 讀取攝像頭畫面
        ret, frame = cap.read()
        if not ret:
            continue

        # 水平翻轉畫面，使其如鏡像般顯示
        frame = cv2.flip(frame, 1)

        # 獲取畫面尺寸
        height, width = frame.shape[:2]
        
        # 計算中心區域（ROI）的座標
        roi_size = 600  # 確保捕獲區域為 600 像素
        roi_x = (width - roi_size) // 2
        roi_y = (height - roi_size) // 2
        
        # 定義並擷取中心區域
        roi = frame[roi_y:roi_y+roi_size, roi_x:roi_x+roi_size]
        roi_resized = cv2.resize(roi, (300, 300))  # 調整大小以便儲存
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x+roi_size, roi_y+roi_size), (0, 255, 0), 2)

        # 在畫面上顯示操作指示
        cv2.putText(frame, f"Collecting {labels[current_label]}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Press 'c' to capture, 'n' for next gesture", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 顯示攝像頭畫面
        cv2.imshow("Data Collection", frame)
        key = cv2.waitKey(1)

        # 處理按鍵輸入
        if key == ord('c'):
            # 儲存圖像
            count += 1
            cv2.imwrite(f'data/{labels[current_label]}/{count}.jpg', roi_resized)
        elif key == ord('n'):
            # 切換到下一個手勢類別
            current_label = (current_label + 1) % 3
            count = 0
        elif key == ord('q'):
            # 按下 'q' 退出程式
            break

    # 釋放資源並關閉視窗
    cap.release()
    cv2.destroyAllWindows()

# 程式入口點
if __name__ == "__main__":
    collect_data()