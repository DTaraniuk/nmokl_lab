import pandas
import tensorflow as tf
import matplotlib.pyplot as plt

books = pandas.read_csv("new_books.csv", header=None, names=["category", "description"])
books.dropna()

words = 10000
D = 4

data = books['description']
tok = tf.keras.preprocessing.text.Tokenizer(num_words=words)
tok.fit_on_texts(data)
seqs = tok.texts_to_sequences(data)

train_args = tf.keras.preprocessing.sequence.pad_sequences(seqs)
train_res = tf.keras.utils.to_categorical(books['category'], D)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(words, 32))
model.add(tf.keras.layers.Conv1D(250, 5, padding='valid', activation='relu'))
model.add(tf.keras.layers.GlobalMaxPooling1D())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(D, activation='softmax'))

model.compile(loss='binary_crossentropy', metrics=['accuracy'])

h = model.fit(train_args, train_res, epochs=10, batch_size=128, validation_split=0.1)

plt.plot(h.history['accuracy'],
         label='Доля вірних відповідей на тренувальному сеті')
plt.plot(h.history['val_accuracy'],
         label='Доля вірних відповідей на тестувальному сеті')
plt.xlabel('Епоха навчання')
plt.ylabel('Доля вірних відповідей')
plt.legend()
plt.show()