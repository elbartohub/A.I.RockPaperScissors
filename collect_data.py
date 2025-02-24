import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont

# 新增函數：用於將 PIL Image 轉換為 OpenCV 格式
def pil_to_cv2(pil_image):
    numpy_image = np.array(pil_image)
    return cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

# 新增函數：在圖片上添加中文文字
def put_chinese_text(img, text, position, font_size=32, color=(0, 255, 0)):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # Modified font loading for cross-platform compatibility
    font_paths = [
        "/System/Library/Fonts/PingFang.ttc",  # MacOS
        "/System/Library/Fonts/Arial Unicode.ttf",  # MacOS
        "C:/Windows/Fonts/msyh.ttc",  # Windows (Microsoft YaHei)
        "C:/Windows/Fonts/simsun.ttc",  # Windows (SimSun)
        "C:/Windows/Fonts/arial.ttf"  # Windows (Arial)
    ]
    
    font = None
    for path in font_paths:
        try:
            font = ImageFont.truetype(path, font_size)
            break
        except:
            continue
    
    if font is None:
        raise ValueError("No suitable font found. Please install a Chinese font.")
    
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

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
    labels_hk = ['揼', '包', '剪']
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

        # 使用新的中文文字渲染函數
        text = f"正在收集: {labels[current_label]} ({labels_hk[current_label]})"
        frame = put_chinese_text(frame, text, (10, 50), font_size=32)
        frame = put_chinese_text(frame, "按 'c' 拍攝, 按 'n' 切換下一個手勢", (10, 100), font_size=24)

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
