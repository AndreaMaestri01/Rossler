import numpy as np
import matplotlib.pyplot as plt

# Definisci le funzioni del sistema di Rössler
def f(x, a):
    xd = np.zeros(len(x))
    xd[0] = -x[1] - x[2]
    xd[1] = x[0] + a[0] * x[1]
    xd[2] = a[1] + x[2] * (x[0] - a[2])
    return xd
# Schema di integrazione di Runge-Kutta di 4° ordine
def RK4_step(f, x, a, h):    
    k1 = f(x, a)
    k2 = f(x + h/2 * k1, a)
    k3 = f(x + h/2 * k2, a)
    k4 = f(x + h * k3, a)
    xp = x + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    return xp
# Schema di integrazione di Eulero
def Euler_step(f,x,a,h):
    xp = x + h*f(x,a)
    return xp
#Grafico 3D
def plot_3D(xs, P1, P2, a, b, c, T):
    plt.figure(figsize=(6, 6))  # Crea una finestra per il grafico
    ax = plt.axes(projection='3d')  # Crea un grafico 3D
    ax.plot(xs[:, 0], xs[:, 1], xs[:, 2])  # Linea 3D
    #ax.set_title(f"Attrattore di Rössler, a={a}, b={b}, c={c} (T={T:.2f})")  # Titolo
    ax.set_xlabel('x')  # Etichetta asse x
    ax.set_ylabel('y')  # Etichetta asse y
    ax.set_zlabel('z')  # Etichetta asse z
    ax.plot(xs[:, 0], xs[:, 1], xs[:, 2], color='green') 
    # Limiti sugli assi
    ax.set_xlim(xs[:, 0].min(), xs[:, 0].max())    
    ax.set_ylim(xs[:, 1].min(), xs[:, 1].max())
    ax.set_zlim(xs[:, 2].min(), xs[:, 2].max())

    # Punti di equilibrio
    ax.scatter(P1[0], P1[1], P1[2], color='black', marker='o', s=10, label='Punto di equilibrio 1')
    #ax.scatter(P2[0], P2[1], P2[2], color='green', marker='o', s=10, label='Punto di equilibrio 2')

    #ax.legend()  # Aggiunge la legenda
    plt.tight_layout()  # Aggiusta il layout per evitare sovrapposizioni
    plt.show()  # Mostra il grafico
#Grafico XY
def plot_XY(xs,P1,P2):
    plt.figure(figsize=(6, 6))  # Crea una nuova finestra per il secondo grafico
    plt.plot(xs[:, 0], xs[:, 1])
    plt.title('')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(P1[0], P1[1], color='red', marker='o', s=10,  label='Punto di equilibrio 1')
    #plt.scatter(P2[0], P2[1], color='green', marker='o', s=10,label='Punto di equilibrio 2')
    plt.tight_layout()  # Aggiusta il layout
    plt.show()  # Mostra il secondo grafico
#Grafico T
def plot_T(xs, ts):
    # Crea una figura con tre sottotrame
    fig, axes = plt.subplots(3, 1, figsize=(10, 8),sharex=True, gridspec_kw={'hspace': 0})
    fig.suptitle('')  # Titolo globale per la figura
    fig.subplots_adjust(hspace=0)
    # Primo sotto-grafico: x(t)
    axes[0].plot(ts, xs[:, 0], color='blue')
    axes[0].set_ylabel('x')
    
    # Secondo sotto-grafico: y(t)
    axes[1].plot(ts, xs[:, 1], color='green')
    axes[1].set_ylabel('y')
  
    # Terzo sotto-grafico: z(t)
    axes[2].plot(ts, xs[:, 2], color='orange')
    axes[2].set_xlabel('Tempo')
    axes[2].set_ylabel('z')

    # Aggiusta il layout per evitare sovrapposizioni
    plt.tight_layout()  # Spazio per il titolo globale
    plt.show()  # Mostra il grafico
#Calcolo punti equilibrio
def Punti_Equilibrio(a,b,c):
#Calcolo punti di Equilibrio
    if c**2 < 4*a*b:
        print('Punti Equilibrio Complessi')
    else:
        delta= np.sqrt(c**2 - (4*a*b))
        x_1 = (c-delta)/(2)
        y_1 = (-c+delta)/(2*a)
        z_1 = (c-delta)/(2*a)

        x_2 = (c+delta)/(2)
        y_2 = (-c-delta)/(2*a)
        z_2 = (c+delta)/(2*a)

        P1=np.array([x_1,y_1,z_1])
        P2=np.array([x_2,y_2,z_2])
    return P1,P2


# Definisci i parametri del sistema di Rössler
a = 0.2
b = 0.2
c = 5.7
T = 100
# Definisci le condizioni iniziali
x0 = 5.0
y0 = 5.0
z0 = 1.0

parametri = np.array([a, b, c])
CI = np.array([x0, y0, z0])

# Definisci il passo temporale e il numero di passi
h = 0.01
N = int((T / h))
print(f"Evoluzione Osservata Per: {h*N} unità temporali")

# Inizializza gli array per memorizzare i risultati 
ts = np.arange(0, (h * N) + h, h)
xs = np.zeros((N+1, 3))  # Array per memorizzare [x, y, z] ad ogni passo

# Imposta le condizioni iniziali
xs[0, :] = CI

# Applica il metodo RK4
for i in range(N):
    xs[i+1, :] = RK4_step(f, xs[i, :], parametri, h)


P1,P2=Punti_Equilibrio(a,b,c)
plot_3D(xs,P1,P2,a,b,c,T)
#plot_XY(xs,P1,P2)
#plot_T(xs,ts)
#np.save('T4_X10.npy', xs)
#np.save('Time',ts)

print(f"Lunghezza Matrice posizioni: {len(xs)}")
print(f"Lunghezza Vettore Tempi: {len(ts)}")
