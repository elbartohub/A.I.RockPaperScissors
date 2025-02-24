import cv2
import numpy as np
import tensorflow as tf
import random
import time

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
    # 載入訓練好的模型
    model = tf.keras.models.load_model('rps_model.h5')
    labels = ['rock', 'paper', 'scissors']
    
    # 初始化攝像頭和遊戲變數
    cap = cv2.VideoCapture(0)
    last_choice_time = time.time()
    countdown = 3
    computer_choice = None
    
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
        img = roi_resized / 255.0
        img = np.expand_dims(img, axis=0)
        prediction = model.predict(img, verbose=0)
        player_choice = labels[np.argmax(prediction)]

        # 每三秒更新一次電腦的選擇
        if current_time - last_choice_time >= 3:
            computer_choice = get_computer_choice()
            last_choice_time = current_time
            countdown = 3
        
        # 計算倒數計時
        remaining_time = int(3 - (current_time - last_choice_time))
        
        # 在畫面上顯示遊戲資訊
        cv2.putText(frame, f"Next round in: {remaining_time}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Your choice: {player_choice}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 顯示電腦選擇和遊戲結果
        if computer_choice:
            cv2.putText(frame, f"Computer's choice: {computer_choice}", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, determine_winner(player_choice, computer_choice), (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

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