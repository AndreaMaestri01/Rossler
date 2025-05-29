import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Carica i dati
prevision_LSTM_uncut= np.load('LSTM_RND1.npy')
prevision_ILSTM_uncut= np.load('ILSTM_RND1.npy')
cut_index=min(len(prevision_LSTM_uncut),len(prevision_ILSTM_uncut))
prevision_LSTM=prevision_LSTM_uncut[-cut_index:,:]
prevision_ILSTM=prevision_ILSTM_uncut[-cut_index:,:]

data_uncut = np.load('RND1.npy')
data_final = data_uncut[:cut_index, :]
data_initial = data_uncut[cut_index:, :]
data_gif = data_initial[-int( (len(data_initial)) * 0.25):] #salva ultime 25% di righe (ultimi due giri per gif prima LSTM)


def plot_X(data,prevision_LSTM,prevision_ILSTM):
    # Crea i grafici
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'hspace': 0})
    fig.suptitle('')  # Titolo globale per la figura
    fig.subplots_adjust(hspace=0)

    # Secondo sotto-grafico: y(t)
    axes[0].plot(data[:, 0], color='blue', alpha=1, label='RK4')
    axes[0].plot(prevision_LSTM[:, 0], color='red', linestyle='--', alpha=1, label='LSTM')
    axes[0].plot(prevision_ILSTM[:, 0], color='green', linestyle='--', alpha=1, label='LSTM Migliorato')
    axes[0].set_ylabel('x')
    axes[0].legend()

    # Primo sotto-grafico: differenze
    axes[1].plot(data[:, 0] - prevision_LSTM[:, 0], color='red', label='Errore LSTM')
    axes[1].plot(data[:, 0] - prevision_ILSTM[:, 0], color='green', label='Errore LSTM Migliorato')
    axes[1].set_ylabel(r'$\Delta x$')
    axes[1].set_xlabel('Tempo')
    axes[1].legend()

    plt.tight_layout()  # Spazio per il titolo globale
    plt.show()  # Mostra il grafico

def plot_Y(data,prevision_LSTM,prevision_ILSTM):
    # Crea i grafici
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'hspace': 0})
    fig.suptitle('')  # Titolo globale per la figura
    fig.subplots_adjust(hspace=0)

    # Secondo sotto-grafico: y(t)
    axes[0].plot(data[:, 1], color='blue', alpha=1, label='RK4')
    axes[0].plot(prevision_LSTM[:, 1], color='red', linestyle='--', alpha=1, label='LSTM')
    axes[0].plot(prevision_ILSTM[:, 1], color='green', linestyle='--', alpha=1, label='LSTM Migliorato')
    axes[0].set_ylabel('y')
    axes[0].legend()

    # Primo sotto-grafico: differenze
    axes[1].plot(data[:, 1] - prevision_LSTM[:, 1], color='red', label='Errore LSTM')
    axes[1].plot(data[:, 1] - prevision_ILSTM[:, 1], color='green', label='Errore LSTM Migliorato')
    axes[1].set_ylabel(r'$\Delta y$')
    axes[1].set_xlabel('Tempo')
    axes[1].legend()

    plt.tight_layout()  # Spazio per il titolo globale
    plt.show()  # Mostra il grafico

def plot_Z(data,prevision_LSTM,prevision_ILSTM):
    # Crea i grafici
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'hspace': 0})
    fig.suptitle('')  # Titolo globale per la figura
    fig.subplots_adjust(hspace=0)

    # Secondo sotto-grafico: y(t)
    axes[0].plot(data[:, 2], color='blue', alpha=1, label='RK4')
    axes[0].plot(prevision_LSTM[:, 2], color='red', linestyle='--', alpha=1, label='LSTM')
    axes[0].plot(prevision_ILSTM[:, 2], color='green', linestyle='--', alpha=1, label='LSTM Migliorato')
    axes[0].set_ylabel('z')
    axes[0].legend()

    # Primo sotto-grafico: differenze
    axes[1].plot(data[:, 2] - prevision_LSTM[:, 2], color='red', label='Errore LSTM')
    axes[1].plot(data[:, 2] - prevision_ILSTM[:, 2], color='green', label='Errore LSTM Migliorato')
    axes[1].set_ylabel(r'$\Delta z$')
    axes[1].set_xlabel('Tempo')
    axes[1].legend()

    plt.tight_layout()  # Spazio per il titolo globale
    plt.show()  # Mostra il grafico

def gif_LSTM(data, prevision):
    fig = plt.figure(dpi=100)
    ax = fig.add_subplot(projection='3d')

    # Numero di punti da aggiungere per ogni frame
    step_size = 10
    # Funzione di aggiornamento per l'animazione
    def update(t):
        ax.cla()  # Pulisce l'asse

        # Imposta i limiti degli assi
        ax.set_xlim(data[:,0].min(), data[:,0].max())
        ax.set_ylim(data[:,1].min(), data[:,1].max())
        ax.set_zlim(data[:,2].min(), data[:,2].max())

        # Disegna i punti blu fino all'inizio di data2
        if t * step_size < len(data): #sto leggendo i dati
            ax.scatter(data[:t*step_size, 0], data[:t*step_size, 1], data[:t*step_size, 2], s=1, color='blue', marker='o')
            ax.set_title("Dati Simulati con RK4", color='blue')
        # Disegna il puntino rosso (posizione attuale)
            ax.scatter(data[t * step_size, 0], data[t * step_size, 1], data[t * step_size, 2], s=20, color='red', marker='o')
        else:
            ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=0.2, color='blue', marker='o',alpha=0.5)
            ax.scatter(prevision[:(t*step_size)-len(data), 0], data[:(t*step_size)-len(data), 1], data[:(t*step_size)-len(data), 2], s=1, color='red', marker='o')
            ax.scatter(data[(t*step_size)-len(data), 0], data[(t*step_size)-len(data), 1], data[(t*step_size)-len(data), 2], s=20, color='red', marker='o')
            ax.set_title("Dati Modello LSTM", color='red')
    # Crea l'animazione
    ani = FuncAnimation(fig, update, frames=((len(data)+len(prevision)) // step_size) + 1, interval=1)  # Aumentato l'intervallo per renderlo più lento

    # Mostra l'animazione
    #plt.show()

    # Salva l'animazione come GIF (opzionale)
    ani.save('LSTM_RND1.gif', writer='pillow',fps=30)

def gif_ILSTM(data, prevision):
    fig = plt.figure(dpi=100)
    ax = fig.add_subplot(projection='3d')

    # Numero di punti da aggiungere per ogni frame
    step_size = 10
    # Funzione di aggiornamento per l'animazione
    def update(t):
        ax.cla()  # Pulisce l'asse

        # Imposta i limiti degli assi
        ax.set_xlim(data[:,0].min(), data[:,0].max())
        ax.set_ylim(data[:,1].min(), data[:,1].max())
        ax.set_zlim(data[:,2].min(), data[:,2].max())

        # Disegna i punti blu fino all'inizio di data2
        if t * step_size < len(data): #sto leggendo i dati
            ax.scatter(data[:t*step_size, 0], data[:t*step_size, 1], data[:t*step_size, 2], s=1, color='blue', marker='o')
            ax.set_title("Dati Simulati con RK4", color='blue')
        # Disegna il puntino rosso (posizione attuale)
            ax.scatter(data[t * step_size, 0], data[t * step_size, 1], data[t * step_size, 2], s=20, color='red', marker='o')
        else:
            ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=0.2, color='blue', marker='o',alpha=0.5)
            ax.scatter(prevision[:(t*step_size)-len(data), 0], data[:(t*step_size)-len(data), 1], data[:(t*step_size)-len(data), 2], s=1, color='red', marker='o')
            ax.scatter(data[(t*step_size)-len(data), 0], data[(t*step_size)-len(data), 1], data[(t*step_size)-len(data), 2], s=20, color='red', marker='o')
            ax.set_title("Dati Modello LSTM Migliorato", color='red')
    # Crea l'animazione
    ani = FuncAnimation(fig, update, frames=((len(data)+len(prevision)) // step_size) + 1, interval=1)  # Aumentato l'intervallo per renderlo più lento

    # Mostra l'animazione
    #plt.show()

    # Salva l'animazione come GIF (opzionale)
    ani.save('ILSTM_RND1.gif', writer='pillow',fps=30)
#plot_X(data_final,prevision_LSTM,prevision_ILSTM)
#plot_Y(data_final,prevision_LSTM,prevision_ILSTM)
#plot_Z(data_final,prevision_LSTM,prevision_ILSTM)

gif_LSTM(data_gif,prevision_LSTM)
gif_ILSTM(data_gif,prevision_ILSTM)