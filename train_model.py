import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Rescaling

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255, input_shape=(300, 300, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_model():
    try:
        # Load and prepare the data
        train_ds = tf.keras.utils.image_dataset_from_directory(
            'data',
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(300, 300),
            batch_size=32,
            label_mode='categorical'  # Add this line to get one-hot encoded labels
        )

        val_ds = tf.keras.utils.image_dataset_from_directory(
            'data',
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(300, 300),
            batch_size=32,
            label_mode='categorical'  # Add this line to get one-hot encoded labels
        )

        # Configure dataset for performance
        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

        # Create and train the model
        model = create_model()
        
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=20
        )
        
        # Save the model
        model.save('rps_model.h5')
        print("Model training completed and saved successfully!")
        
    except Exception as e:
        print(f"An error occurred during training: {str(e)}")

if __name__ == "__main__":
    train_model()