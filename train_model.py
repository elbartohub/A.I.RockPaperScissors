import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Rescaling

def create_model():
    # 建立模型的輸入層，設置輸入圖像大小為 300x300，3個顏色通道
    inputs = tf.keras.Input(shape=(300, 300, 3))
    # 將像素值正規化到 0-1 之間
    x = tf.keras.layers.Rescaling(1./255)(inputs)
    
    # 第一個卷積區塊
    # 使用32個3x3的卷積核進行特徵提取
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    # 批次正規化，用於加速訓練和防止過度擬合
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    # 最大池化層，減少特徵圖大小
    x = tf.keras.layers.MaxPooling2D(2, 2)(x)
    # Dropout層，防止過度擬合
    x = tf.keras.layers.Dropout(0.25)(x)
    
    # 第二個卷積區塊，特徵圖數量增加到64
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(2, 2)(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    
    # 第三個卷積區塊，特徵圖數量增加到128
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(2, 2)(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    
    # 全連接層
    # 將特徵圖展平
    x = tf.keras.layers.Flatten()(x)
    # 添加具有512個神經元的密集層
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    # 輸出層，3個類別（石頭、剪刀、布）
    outputs = tf.keras.layers.Dense(3, activation='softmax')(x)

    # 建構模型
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # 編譯模型，設置優化器、損失函數和評估指標
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_model():
    try:
        # 設置數據增強，增加訓練數據的多樣性
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomRotation(0.2),    # 隨機旋轉
            tf.keras.layers.RandomZoom(0.2),        # 隨機縮放
            tf.keras.layers.RandomBrightness(0.2),  # 隨機調整亮度
            tf.keras.layers.RandomContrast(0.2),    # 隨機調整對比度
        ])

        # 載入訓練數據，設置80%用於訓練，20%用於驗證
        train_ds = tf.keras.utils.image_dataset_from_directory(
            'data',
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(300, 300),
            batch_size=32,
            label_mode='categorical'
        )

        # 載入驗證數據集
        val_ds = tf.keras.utils.image_dataset_from_directory(
            'data',
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(300, 300),
            batch_size=32,
            label_mode='categorical'
        )

        # 對訓練數據應用數據增強
        train_ds = train_ds.map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        # 優化數據集效能
        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

        # 建立模型
        model = create_model()
        
        # 添加早停機制，防止過度擬合
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True
        )
        
        # 訓練模型
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=50,  # 訓練輪數
            callbacks=[early_stopping]
        )
        
        # 儲存模型
        model.save('model/rock_paper_scissors_model.keras')
        print("Model training completed and saved successfully!")
        
    except Exception as e:
        print(f"An error occurred during training: {str(e)}")

if __name__ == "__main__":
    train_model()
