import cv2
import numpy as np
import tensorflow as tf
import random
import time
from PIL import Image, ImageDraw, ImageFont

# Add the Chinese text rendering function
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

# 取得電腦的隨機選擇（石頭、布或剪刀）
def get_computer_choice():
    return random.choice(['rock', 'paper', 'scissors'])

# 判斷遊戲勝負
def determine_winner(player_choice, computer_choice):
    if player_choice == computer_choice:
        return "Tie!"
    
    # 定義勝利組合：鍵是贏家，值是輸家
    winning_combinations = {
        'rock': 'scissors',
        'paper': 'rock',
        'scissors': 'paper'
    }
    
    if winning_combinations[player_choice] == computer_choice:
        return "You win!"
    return "Computer wins!"

# 遊戲主程式
def play_game():
    model = tf.keras.models.load_model('model/rock_paper_scissors_model.keras')
    labels = ['rock', 'paper', 'scissors']
    labels_hk = ['揼', '包', '剪']
    
    cap = cv2.VideoCapture(0)
    last_choice_time = time.time()
    last_prediction_time = time.time()  # Add prediction timing
    countdown = 5
    computer_choice = None
    computer_choice_hk = None
    player_choice = None  # Initialize player choice
    
    while True:
        # 讀取攝像頭畫面
        ret, frame = cap.read()
        if not ret:
            continue

        # 水平翻轉畫面，使其如鏡像般顯示
        frame = cv2.flip(frame, 1)
        current_time = time.time()

        # 設定畫面中央的擷取區域（ROI）
        height, width = frame.shape[:2]
        roi_size = 600
        roi_x = (width - roi_size) // 2
        roi_y = (height - roi_size) // 2
        
        roi = frame[roi_y:roi_y+roi_size, roi_x:roi_x+roi_size]
        roi_resized = cv2.resize(roi, (300, 300))
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x+roi_size, roi_y+roi_size), (0, 255, 0), 2)

        # 使用模型進行手勢預測
        # Update prediction every 100ms instead of every frame
        current_time = time.time()
        if current_time - last_prediction_time >= 0.1:  # 100ms interval
            img = roi_resized / 255.0
            img = np.expand_dims(img, axis=0)
            prediction = model.predict(img, verbose=0)
            player_choice = labels[np.argmax(prediction)]
            last_prediction_time = current_time

        # 每五秒更新一次電腦的選擇
        if current_time - last_choice_time >= 5:  # Changed from 3 to 5
            computer_choice = get_computer_choice()
            last_choice_time = current_time
            countdown = 5  # Changed from 3 to 5
        
        # 計算倒數計時
        remaining_time = int(5 - (current_time - last_choice_time))  # Changed from 3 to 5
        
        # 在畫面上顯示遊戲資訊
        # Update the text display sections
        frame = put_chinese_text(frame, f"下一回合倒數: {remaining_time}", (10, 30), font_size=32)
        frame = put_chinese_text(frame, f"你的選擇: {labels[labels.index(player_choice)]} ({labels_hk[labels.index(player_choice)]})", 
                               (10, 70), font_size=32)
        
        if computer_choice:
            computer_choice_hk = labels_hk[labels.index(computer_choice)]
            frame = put_chinese_text(frame, f"電腦的選擇: {computer_choice} ({computer_choice_hk})", 
                                   (10, 110), font_size=32)
            result = "平手！" if determine_winner(player_choice, computer_choice) == "Tie!" else \
                    "你贏了！" if determine_winner(player_choice, computer_choice) == "You win!" else "電腦贏了！"
            frame = put_chinese_text(frame, result, (10, 150), font_size=32)

        # 顯示遊戲畫面
        cv2.imshow("Rock Paper Scissors", frame)

        # 按下 'q' 鍵退出遊戲
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 釋放資源並關閉視窗
    cap.release()
    cv2.destroyAllWindows()

# 程式入口點
if __name__ == "__main__":
    play_game()
