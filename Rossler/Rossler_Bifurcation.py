import numpy as np
import matplotlib.pyplot as plt



def f(x, a):
    xd = np.zeros(len(x))
    xd[0] = -x[1] - x[2]
    xd[1] = x[0] + a[0] * x[1]
    xd[2] = a[1] + x[2] * (x[0] - a[2])
    return xd

# Schema di integrazione di Runge-Kutta di 4° ordine
def RK4_step(f, x,y,z, a, h):    
    X=np.array([x,y,z])
    k1 = f(X, a)
    k2 = f(X + h/2 * k1, a)
    k3 = f(X + h/2 * k2, a)
    k4 = f(X + h * k3, a)
    xp = X + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    return xp[0], xp[1], xp[2]

# Parametri del sistema di Rössler
a = 0.15
b = 0.2
c_s = np.arange(1, 20, 0.01)

# Condizioni iniziali
x0 = -4.9
y0 = -0.6
z0 = 0.0

# Parametri di integrazione
h = 0.01
T = 2500
N = int(T / h)

# Inizializzo vettori
X_list = []
C_list = []

for c in c_s:
    print(f'{(c/np.max(c_s))*100}%') #caricamento
    parameters=np.array([a, b, c])
    x, y, z = x0, y0, z0
    counter = 10  # Contatore per eliminare i primi 10 punti transitori
    
    for i in range(N):
        x, y, z = RK4_step(f,x, y, z, parameters, h)
        
        if 0 < x < 70 and np.abs(y) < 1e-2:
            if counter == 0:
                X_list.append(x)
                C_list.append(c)
            else:
                counter -= 1

# Grafico
ax = plt.figure().add_subplot(111)
plt.xlim([0, 45])
plt.ylim([0, 70])
plt.xlabel('c')
plt.ylabel('X')
#plt.title(f'Diagramma di Biforcazione per Rossler (a={a}, b={b}), RK4 (h={h}), T={T}s' )
plt.scatter(C_list, X_list, .1, color='green')
plt.show()
