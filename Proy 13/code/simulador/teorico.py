import math

def P0(lam, mu, c):
    rho = lam / (c * mu)
    suma = sum((lam/mu)**n / math.factorial(n) for n in range(c))
    termino = ((lam/mu)**c / math.factorial(c)) * (1 / (1 - rho))
    return 1 / (suma + termino)

def P_esperar(lam, mu, c):
    rho = lam / (c * mu)
    P0_val = P0(lam, mu, c)
    numerador = ((lam/mu)**c / math.factorial(c)) * (1 / (1 - rho))
    return numerador * P0_val

def Wq_teorico(lam, mu, c):
    return P_esperar(lam, mu, c) / (c * mu - lam)

def W_teorico(lam, mu, c):
    return Wq_teorico(lam, mu, c) + 1/mu
