from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Reshape, LSTM, Dense, Dropout, TimeDistributed, Flatten, UpSampling1D

def build_ocr_model(input_shape, vocab_size, max_sequence_length=256):
    inputs = Input(shape=input_shape)
    
    # Convolutional layers for feature extraction
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)  # 256x256
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)  # 128x128
    
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)  # 64x64
    
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)  # 32x32
    
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)  # 16x16
    
    # Reshape to sequence
    x = Reshape((16, 8192))(x)
    
    # Reduce feature dimension
    x = Dense(512, activation='relu')(x)
    
    # LSTM layers
    x = LSTM(256, return_sequences=True)(x)
    x = Dropout(0.25)(x)
    
    # Upsample to target sequence length
    x = UpSampling1D(size=16)(x)  # Now we have (None, 256, 256)
    
    # Final output layer
    outputs = Dense(vocab_size + 1, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    # Print model summary for debugging
    model.summary()
    
    return model