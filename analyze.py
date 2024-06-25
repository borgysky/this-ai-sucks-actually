import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model
from preprocess import preprocess_user_text
import tkinter as tk

# Параметры
num_words = 10000
max_length = 500

# Загрузка модели
model = load_model('sentiment_model.h5')

# Инициализация Tokenizer
word_index = imdb.get_word_index()
tokenizer = Tokenizer(num_words=num_words)
tokenizer.word_index = {k: (v+3) for k, v in word_index.items()}
tokenizer.word_index['<PAD>'] = 0
tokenizer.word_index['<START>'] = 1
tokenizer.word_index['<UNK>'] = 2
tokenizer.word_index['<UNUSED>'] = 3

def evaluate_sentiment():
    # Ввод пользовательского текста из текстового поля
    user_text = text_entry.get("1.0", tk.END).strip()

    if user_text:
        # Предобработка пользовательского текста
        processed_text = preprocess_user_text(user_text, tokenizer, max_length)

        # Оценка тональности пользовательского текста
        prediction = model.predict(processed_text)
        sentiment = 'Положительный' if prediction[0][0] > 0.5 else 'Отрицательный'
        confidence = prediction[0][0] if sentiment == 'Положительный' else 1 - prediction[0][0]

        # Отображение результата в метке
        result_label.config(text=f'Тональность: {sentiment} (Уверенность: {confidence:.2f})')

# Создание главного окна
root = tk.Tk()
root.title("Анализ тональности")

# Создание и размещение элементов управления
text_entry = tk.Text(root, height=10, width=50)
text_entry.pack(pady=10)

evaluate_button = tk.Button(root, text="Оценить тональность", command=evaluate_sentiment)
evaluate_button.pack(pady=10)

result_label = tk.Label(root, text="Тональность: ", font=("Helvetica", 14))
result_label.pack(pady=10)

# Запуск главного цикла обработки событий
root.mainloop()
