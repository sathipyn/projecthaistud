import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from processingdata import load_data, encode_labels
from modelocr import build_ocr_model

def train():
    image_folder = "data/output_images/"
    label_file = "data/labeling_template.csv"
    
    # Load and preprocess data
    print("Loading data...")
    images, texts = load_data(image_folder, label_file)
    
    # Define char_map (keep your existing char_map)
    char_map = {c: i for i, c in enumerate("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?:\\()[]+-*/=<>^_%|~\x85\n²\x96°³"
            " .,!?:()[]+-*/=<>^_%|~²³°"
            "∑∏∆∇∫∮∝∞∈∉∋∌∩∪⊂⊃⊆⊇≈≠≡≤≥"
            "αβγδεζηθικλμνξοπρστυφχψω"
            "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ"
            "…""''„‚«»‹›")}
    
    print("Encoding labels...")
    labels = encode_labels(texts, char_map)
    
    # Constants
    max_sequence_length = 256
    vocab_size = len(char_map)
    
    print("Padding sequences...")
    labels_padded = pad_sequences(labels, maxlen=max_sequence_length, padding='post', truncating='post')
    
    print("Splitting data...")
    x_train, x_val, y_train, y_val = train_test_split(
        images, labels_padded, test_size=0.2, random_state=42
    )
    
    print("Converting to one-hot encoding...")
    y_train_one_hot = to_categorical(y_train, num_classes=vocab_size + 1)
    y_val_one_hot = to_categorical(y_val, num_classes=vocab_size + 1)
    
    print(f"Input shape: {images.shape}")
    print(f"Target shape: {y_train_one_hot.shape}")
    
    # Build model
    print("Building model...")
    input_shape = (512, 512, 1)
    model = build_ocr_model(input_shape, vocab_size, max_sequence_length)
    
    # Compile model
    print("Compiling model...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-7
    )
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Train model
    print("Starting training...")
    model.fit(
        x_train,
        y_train_one_hot,
        batch_size=8,  # Reduced batch size due to model size
        epochs=100,
        validation_data=(x_val, y_val_one_hot),
        callbacks=[reduce_lr, early_stopping]
    )
    
    # Save model
    model.save("models/ocr_model.h5")
    print("Model saved to models/ocr_model.h5")

if __name__ == "__main__":
    train()