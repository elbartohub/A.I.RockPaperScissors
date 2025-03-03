import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard

def create_transfer_learning_model():
    """創建基於MobileNetV2的遷移學習模型"""
    # 使用 MobileNetV2 作為基礎模型
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # 凍結大部分預訓練層，只解凍最後幾層
    base_model.trainable = True
    for layer in base_model.layers[:-30]:  # 凍結更多層
        layer.trainable = False

    # 建立新的模型
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    
    # 使用預訓練模型提取特徵
    x = base_model(x, training=True)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    # 增加更多正則化
    x = tf.keras.layers.Dense(256, activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    x = tf.keras.layers.Dense(128, activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    outputs = tf.keras.layers.Dense(3, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # 使用較小的學習率
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_high_res_model(input_size=(512, 512, 3)):
    """創建基於MobileNetV2的高解析度模型，用於512x512輸入"""
    # 使用 MobileNetV2 作為基礎模型
    base_model = MobileNetV2(
        input_shape=input_size,
        include_top=False,
        weights='imagenet'
    )
    
    # 凍結大部分預訓練層，只解凍最後幾層
    base_model.trainable = True
    for layer in base_model.layers[:-30]:  # 凍結更多層
        layer.trainable = False

    # 建立新的模型
    inputs = tf.keras.Input(shape=input_size)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    
    # 使用預訓練模型提取特徵
    x = base_model(x, training=True)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    # 增加更多正則化
    x = tf.keras.layers.Dense(512, activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    x = tf.keras.layers.Dense(256, activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    # 修改輸出層，確保類別順序與數據集一致
    # 修改輸出層，調整類別順序為 Paper, Rock, Scissors
    outputs = tf.keras.layers.Dense(3, activation='softmax', name='predictions')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_high_res_model():
    """使用高解析度圖像訓練石頭剪刀布模型"""
    try:
        # 確保模型保存目錄存在
        os.makedirs('model', exist_ok=True)
        
        # 設置適度的數據增強
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomTranslation(0.1, 0.1),
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomBrightness(0.2),
            tf.keras.layers.RandomContrast(0.2),
        ])
        
        # 載入訓練數據，使用512x512的圖像尺寸
        train_ds = tf.keras.utils.image_dataset_from_directory(
            'data',
            validation_split=0.3,
            subset="training",
            seed=42,
            image_size=(512, 512),  # 使用高解析度
            batch_size=16,          # 減小batch size以適應更大的圖像
            shuffle=True,
            label_mode='categorical'
        )

        val_ds = tf.keras.utils.image_dataset_from_directory(
            'data',
            validation_split=0.3,
            subset="validation",
            seed=42,
            image_size=(512, 512),  # 使用高解析度
            batch_size=16,
            shuffle=True,
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

        # 建立高解析度模型
        model = create_high_res_model()
        
        # 添加早停機制
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            min_delta=0.001
        )
        
        # 學習率調整
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=0.00001,
            verbose=1
        )
        
        # 添加 TensorBoard 回調
        tensorboard_callback = TensorBoard(
            log_dir='logs/high_res_model',
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        )
        
        # 訓練模型
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=50,
            callbacks=[early_stopping, reduce_lr, tensorboard_callback],
            verbose=1
        )
        
        # 添加訓練過程可視化
        from visualize_training import plot_training_history
        plot_training_history(history)
        print("訓練過程可視化已保存為 'training_history_high_res.png'")
        
        # 可視化一些預測結果
        for images, labels in val_ds.take(1):
            from visualize_training import visualize_model_predictions
            visualize_model_predictions(model, images[:8], labels[:8], num_images=8)
            print("預測結果可視化已保存為 'prediction_samples_high_res.png'")
            break
        
        # 打印最終的驗證準確率
        final_accuracy = max(history.history['val_accuracy'])
        print(f"高解析度模型最佳驗證準確率: {final_accuracy:.4f}")

        # 儲存模型
        model.save('model/rock_paper_scissors_high_res_model.keras')
        print("高解析度模型訓練完成並成功保存！")
        
    except Exception as e:
        print(f"高解析度模型訓練過程中發生錯誤: {str(e)}")

# 在主函數中添加選項來訓練高解析度模型
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='訓練石頭剪刀布識別模型')
    parser.add_argument('--high-res', action='store_true', help='使用高解析度(512x512)訓練模型')
    
    args = parser.parse_args()
    
    if args.high_res:
        print("使用高解析度(512x512)訓練模型...")
        train_high_res_model()
    else:
        print("使用標準解析度(224x224)訓練模型...")
        train_with_transfer_learning()