# 包剪揼 AI 遊戲

一個使用電腦視覺和深度學習的即時包剪揼遊戲。

## 系統需求

本專案需要以下 Python 函式庫：
- TensorFlow (2.x)
- OpenCV (cv2)
- NumPy
- Python 3.10.16

## 環境設置

### 使用 Conda

1. 建立新的 conda 環境：<br>
conda create -n aigame python=3.10.16<br>
conda activate aigame<br>
pip install -r requirements.txt<br>

## 專案結構
- collect_data.py ：使用網絡攝像頭收集訓練數據的腳本
- train_model.py ：神經網絡模型訓練腳本
- play_game.py ：包剪揼遊戲主程式

## 使用方法
1. 首先，收集訓練數據：
python collect_data.py

2. 訓練模型：
python train_model.py

3. 開始遊戲:
python play_game.py

## 操作說明
- 按 'c' 鍵進行數據收集時拍攝圖像
- 按 'n' 鍵在數據收集時切換手勢
- 按 'q' 鍵退出任何程式

## 數據收集
數據將自動組織為以下結構：
data/
    rock/     # 揼
    paper/    # 包
    scissors/ # 剪
    
每個資料夾將包含用於訓練的相應手勢圖像。
