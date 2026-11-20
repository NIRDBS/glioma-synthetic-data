import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

def build_1d_cnn(input_shape=(1738, 1), num_classes=6, learning_rate=0.001, dropout_rate=0.4,
                  loss='sparse_categorical_crossentropy'):
    """
    1D-CNN Baseline for Raman Spectra.
    Detects spectral shapes rather than just peak heights.

    Compiles exactly once, here, with every setting the caller actually wants;
    callers must not compile() again
    afterward, or dropout_rate/learning_rate/loss silently stop being the ones
    this model actually trains with.
    """
    inputs = layers.Input(shape=input_shape)
    
    # Reshape (Batch, 1738) -> (Batch, 1738, 1) if needed
    if len(input_shape) == 1:
        x = layers.Reshape((input_shape[0], 1))(inputs)
    else:
        x = inputs

    # Block 1: Broad Features (Kernel 11)
    x = layers.Conv1D(32, kernel_size=11, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    # Block 2: Medium Features (Kernel 5)
    x = layers.Conv1D(64, kernel_size=5, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    # Block 3: Fine Features (Kernel 3)
    x = layers.Conv1D(128, kernel_size=3, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    # Global Features
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Output
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="Raman_CNN")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=['accuracy']
    )
    return model