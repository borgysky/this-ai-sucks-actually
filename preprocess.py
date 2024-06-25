import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

def load_and_preprocess_data(num_words=10000, max_length=500):
    # Загрузка датасета IMDB
    (train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.imdb.load_data(num_words=num_words)

    # Предобработка данных (дополнение последовательностей до фиксированной длины)
    train_data = pad_sequences(train_data, maxlen=max_length)
    test_data = pad_sequences(test_data, maxlen=max_length)

    return (train_data, train_labels), (test_data, test_labels)

def preprocess_user_text(text, tokenizer, max_length=500):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length)
    return padded_sequence

if __name__ == '__main__':
    (train_data, train_labels), (test_data, test_labels) = load_and_preprocess_data()
    print(f'Train data shape: {train_data.shape}')
    print(f'Test data shape: {test_data.shape}')
