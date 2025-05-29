import numpy as np
from scipy.optimize import leastsq
import matplotlib.pyplot as plt

# Definisci le funzioni del sistema di Rössler
def f(x,a,T):
    xd = np.zeros(len(x))
    xd[0] = T*(-x[1] - x[2])
    xd[1] = T*(x[0] + a[0] * x[1])
    xd[2] = T*(a[1] + x[2] * (x[0] - a[2]))
    return xd


# Schema di integrazione di Runge-Kutta di 5° ordine
def RK4_step(f, x, a, h, T):
    k1 = f(x, a, T)
    k2 = f((x+h/2*k1),a,T)
    k3 = f((x+h/2*k2),a,T)
    k4 = f((x+h*k3),a,T)
    xp = x + h/6 *(k1 + 2*k2 + 2*k3 + k4)
    return xp

# Calcola il vettore di errore residuo
def Res(v, x, a, f, h, p):
    j = int(2.0 / h)
    vv = np.zeros((j, len(x))) #matrice per minimizzazione
    
    # Imposta la condizione iniziale
    vv[0, :2] = v[:2]
    vv[0, 2] = x[2]
    T = v[2]
    
    # Integrazione per calcolare la traiettoria
    i = 0
    while i < j / 2 + p:
        t = i * h
        vv[i+1, :] = RK4_step(f, vv[i, :],a,h,T)
        i += 1
    
    # Calcola il vettore di errore residuo
    er = vv[j//2, :] - vv[0, :]
    for i in range(1, p):
        er = np.concatenate((er, vv[j//2 + i, :] - vv[i, :]))
    return er

# Funzione principale per l'ottimizzazione delle condizioni iniziali
def main():
    #Parametri di interesse
    a = 0.15
    b = 0.2
    c = 10.5

    # Guesse  delle condizioni iniziali e del Periodo
    x0 = -4
    y0 = -0.6
    z0 = 0
    T = 5.92030065*4

    parametri = np.array([a, b, c])
    guess = np.array([x0, y0 ,z0])
    
    v0 = np.zeros(3) #v0 contiene i parametri da ottimizzare (non viene ottimizzato z)
    v0[:2] = guess[:2]
    v0[2] = T  # Stima iniziale del periodo
    
    p = 2  # Lunghezza del residuo
    h = 0.001  # Passo di integrazione rk
    
    v, _ = leastsq(Res, v0, args=(guess, parametri,f,h, p), ftol=1e-12, maxfev=200)
        
    print('Stima x0:', v[0])
    print('Stima y0:', v[1])
    print('Stima z0:', z0)
    print('Stima T:', v[2])
# Esecuzione della funzione principale
main()