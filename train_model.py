import tensorflow as tf
import json
import numpy as np
from random import shuffle, seed
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


class MaliciousDocumentDetectionModel(tf.keras.Model):
    def __init__(self):
        super(MaliciousDocumentDetectionModel, self).__init__()
        self.embedding_dim = 100
        self.lstm_units = 64

        self.embedding = tf.keras.layers.Embedding(input_dim=10000, output_dim=self.embedding_dim, input_length=100)
        self.bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.lstm_units))
        self.dense1 = tf.keras.layers.Dense(16, activation='relu')
        self.out = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=False):
        x = self.embedding(inputs)
        x = self.bilstm(x)
        x = self.dense1(x)
        return self.out(x)

    def evaluate(self, x, y):
        predictions = self.call(x)
        loss = tf.keras.losses.binary_crossentropy(y, predictions)
        accuracy = tf.keras.metrics.binary_accuracy(y, predictions)
        return tf.reduce_mean(loss), tf.reduce_mean(accuracy)


def collect_data_file(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data


def embed(texts, tokenizer, max_length=100):
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
    return padded_sequences


if __name__ == "__main__":
    # Load and shuffle data
    file_path = 'data/cat_rag_dataset.json'
    collected_data = collect_data_file(file_path)
    seed(42)
    shuffle(collected_data)

    # Split
    split_index = int(len(collected_data) * 0.8)
    train_data = collected_data[:split_index]
    test_data = collected_data[split_index:]

    # Prepare text and labels
    train_x = [item['content'] for item in train_data]
    train_y = np.array([item['label'] for item in train_data])
    test_x = [item['content'] for item in test_data]
    test_y = np.array([item['label'] for item in test_data])


    train_y = np.where(train_y == 'malicious', 1, 0)
    test_y = np.where(test_y == 'malicious', 1, 0)


    # Tokenize and embed
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(train_x)
    train_x_embedded = np.array(embed(train_x, tokenizer), dtype=np.float32)
    test_x_embedded = np.array(embed(test_x, tokenizer), dtype=np.float32)


    print("Type:", type(train_x_embedded))
    print("Dtype:", train_x_embedded.dtype)
    print("Shape:", train_x_embedded.shape)


    # Build and train the model
    model = MaliciousDocumentDetectionModel()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_x_embedded, train_y, epochs=25, batch_size=4)

    # Evaluate
    loss, accuracy = model.evaluate(test_x_embedded, test_y.reshape(-1, 1))
    print("Test Loss:", loss.numpy())
    print("Test Accuracy:", accuracy.numpy())


    # query loop for testing
    while True:
        query = input("Enter a document to classify (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            print("Exiting the loop.")
            break
        
        # Embed the query
        query_embedded = embed([query], tokenizer)
        prediction = model.predict(query_embedded)
        
        if prediction[0] > 0.5:
            print("The document is classified as malicious.")
        else:
            print("The document is classified as benign.")  

    # save model
    model.save('malicious_document_detection_model.keras')