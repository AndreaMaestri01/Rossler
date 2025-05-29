import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import visualkeras

# Carica i dati
positions = np.load('RND1.npy')

# Normalizzazione dei dati
scaler = StandardScaler()
positions = scaler.fit_transform(positions)

# Creazione delle sequenze
def create_sequences(data, sequence_length):
    sequences = []
    labels = []
    for i in range(len(data) - sequence_length):
        seq = data[i:i + sequence_length]
        label = data[i + sequence_length]
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)

Lunghezza_Sequenza = 10
Sequenze, Etichette = create_sequences(positions, Lunghezza_Sequenza)

# Divisione dei dati
split = int(0.8 * len(Sequenze))
Seq_train, Seq_test = Sequenze[:split], Sequenze[split:]
Eti_train, Eti_test = Etichette[:split], Etichette[split:]

# Creazione del modello
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(Seq_train.shape[1], Seq_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(3))

model.compile(optimizer='adam', loss='mean_squared_error')

# Addestramento del modello con Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
history = model.fit(Seq_train, Eti_train, epochs=10, batch_size=32, validation_data=(Seq_test, Eti_test), callbacks=[early_stopping])

# Visualizzazione Modello senza font
visualkeras.layered_view(model, legend=True) 

# Previsioni
predictions_unscale = model.predict(Seq_test)
predictions = scaler.inverse_transform(predictions_unscale)

# Calcolo delle metriche
mse = mean_squared_error(Eti_test, predictions_unscale)
r2 = r2_score(Eti_test, predictions_unscale)
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

np.save('ILSTM_RND1.npy', predictions)

