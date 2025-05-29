import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import visualkeras

positions = np.load('RND1.npy')

#Creiamo (con una lunghezza decisa) delle sequenze, ovvere una lista di valori noti che vengono usati per prevedere le etichette (valore successivo)
def create_sequences(data, sequence_length):
    sequences = []
    labels = []
    for i in range(len(data) - sequence_length):
        seq = data[i:i + sequence_length]
        label = data[i + sequence_length]
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)

# Parametri
Lunghezza_Sequenza = 10  # Lunghezza della finestra temporale

# Creazione delle sequenze
Sequenze, Etichette = create_sequences(positions, Lunghezza_Sequenza)
if len(Sequenze) != len(Etichette):
    print("Errore nelle lunghezze delle Sequenze e Eichette")

# Divisione dati in addestramento e test
split = int(0.8 * len(Sequenze))
Seq_train, Seq_test = Sequenze[:split], Sequenze[split:]
Eti_train, Eti_test = Etichette[:split], Etichette[split:]


#----Creazione del modello LSTM (Utile per previsioni temporali)----

#Crea modello sequenziale (aggiungerò layer per layer)
model = Sequential() 
#Aggiungo primo strato a 50 neuroni, e chiedo di dare in output tutta la sequenza (Utile per previsioni temporali) e inoltre specifico la dimensione del layer
model.add(LSTM(50, return_sequences=True, input_shape=(Seq_train.shape[1], Seq_train.shape[2])))
#Aggiungo secondo strato a 50 neuroni, e chiedo di dare in output solo ultima sequenza
model.add(LSTM(50))
#Aggiunge uno strato denso che elabora output in un vettore x,y,x
model.add(Dense(3))  

#Prepara modello al train specificando aggiornamento pesi, uso ADAM come ottimizzatore per la funzione MSE
model.compile(optimizer='adam', loss='mean_squared_error')
#Stampa un riassunto del modello
model.summary()


#----Addestramento del Modello LSTM----

#Addestro per 10 epoche, processo 32 dati per volta (un batch) prima di aggiornare i dati
model.fit(Seq_train, Eti_train, epochs=10, batch_size=32, validation_data=(Seq_test, Eti_test))

#---Visualizzazione Modello---
visualkeras.layered_view(model, legend=True)


#----Previsioni del Modello LSTM----
predictions = model.predict(Seq_test)

# Calcolo delle metriche
mse = mean_squared_error(Eti_test, predictions)
r2 = r2_score(Eti_test, predictions)
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")


# ----Visualizzazione dei risultati
np.save('LSTM_RND1.npy', predictions)