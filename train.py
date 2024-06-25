import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from preprocess import load_and_preprocess_data

# Параметры
num_words = 10000
max_length = 500
embedding_dim = 128

# Загрузка и предобработка данных
(train_data, train_labels), (test_data, test_labels) = load_and_preprocess_data(num_words, max_length)

# Создание модели
model = Sequential()
model.add(Embedding(input_dim=num_words, output_dim=embedding_dim, input_length=max_length))
model.add(LSTM(units=128, return_sequences=False))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Обучение модели
model.fit(train_data, train_labels, epochs=100, batch_size=64, validation_split=0.2)

# Сохранение модели
model.save('sentiment_model.h5')

# Оценка модели на тестовых данных
loss, accuracy = model.evaluate(test_data, test_labels)
print(f'Точность тестирования: {accuracy:.2f}')
