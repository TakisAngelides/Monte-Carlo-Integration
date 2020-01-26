# Exercise 1 Core Task 2: Fresnel integrals

# This program calculates the Fresnel integrals using scipy's quad() and then plots the cornu spiral.
# It also creates a pandas dataframe and prints from that the first and last 10 points of the cornu locus.

from scipy.integrate import quad
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
import pandas as pd

U_SAMPLES = 1000
LOWER_U = -15
UPPER_U = 15
U_VALUES = np.linspace(LOWER_U, UPPER_U, num = U_SAMPLES)

def cos_integrand(x):
    """
    :param x: (float) Dummy variable of integration handled by the quad() function
    :return: (float) The cos integrand of the Fresnel integrals
    """
    return np.cos((pi*x**2)/2)

def sin_integrand(x):
    """
    :param x: (float) Dummy variable of integration handled by the quad() function
    :return: (float) The sin integrand of the Fresnel integrals
    """
    return np.sin((pi*x**2)/2)

def get_C_value(u):
    """
    :param u: (float) The upper limit of integration
    :return: (float) The value of the cos Fresnel integral
    """
    return quad(cos_integrand, 0, u)[0]

def get_S_value(u):
    """
    :param u: (float) The upper limit of integration
    :return: (float) The value of the sin Fresnel integral
    """
    return quad(sin_integrand, 0, u)[0]

def get_C_list():
    """
    :return: (list) Returns many values of C at different values of u as a list
    """
    C_list = []
    for u in U_VALUES:
        C_value = get_C_value(u)
        C_list.append(C_value)
    return C_list

def get_S_list():
    """
    :return: (list) Returns many values S at different values of u as a list
    """
    S_list = []
    for u in U_VALUES:
        S_value = get_S_value(u)
        S_list.append(S_value)
    return S_list

#Generate C and S values in lists
C = get_C_list()
S = get_S_list()

#Create dataframes to print the values of some points in the spiral locus
df_C = pd.DataFrame(C, columns = ['C'])
df_S = pd.DataFrame(S, columns = ['S'])
df = pd.concat([df_C, df_S], axis = 1)
df.index.name = 'Point'
print(df.head(5))
print(df.tail(5))

#Plot the Cornu spiral
plt.plot(C, S, color = 'k')
plt.grid(True)
plt.xlabel('C(u)')
plt.ylabel('S(U)')
plt.title('Cornu Spiral')
plt.axhline(y = 0, lw = 1, color = 'k')
plt.axvline(x = 0, lw = 1, color = 'k')
plt.savefig('cornu_spiral.png', bbox_inches = "tight")
plt.show()