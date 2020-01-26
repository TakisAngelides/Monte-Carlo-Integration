# Exercise 1 Supplementary Task 2: Fresnel slit diffraction

from numpy import pi
import numpy as np
from scipy.integrate import quad
import cmath
import matplotlib.pyplot as plt

# Wavelength of illuminating light in metres
WAVELENGTH = 0.01
# Aperture to screen distance 0.3, 0.5, 1 metres
D_ARRAY = [0.3, 0.5, 1]
# Slit width in metres
d = 0.3
# Generate list of points to evaluate the pattern, the value in points is the lower limit x_0
POINTS = np.linspace(-1, d+1, num = 10001)

def cos_integrand(x, D):
    """
    :param x: (float) Dummy variable of integration handled by the quad() function
    :param D: (float) Aperture to screen distance in metres
    :return: (float) The integrand cos in the complex amplitude integral
    """
    return np.cos((pi*x**2)/(WAVELENGTH*D))

def sin_integrand(x, D):
    """
    :param x: (float) Dummy variable of integration handled by the quad() function
    :param D: (float) Aperture to screen distance in metres
    :return: (float) The integrand sin in the complex amplitude integral
    """
    return np.sin((pi*x**2)/(WAVELENGTH*D))

def get_x_value(lower, upper, D):
    """
    :param lower: (float) Lower limit of integration -> -(distance from left slit edge in metres)
    :param upper: (float) Upper limit of integration -> +(distance from right slit edge in metres)
    :param D: (float) Aperture to screen distance in metres
    :return: (float) Integration value of the cos part forming the real part of the complex amplitude
    """
    return quad(cos_integrand, lower, upper, args = (D,))[0]

def get_y_value(lower, upper, D):
    """
    :param lower: (float) Lower limit of integration -> -(distance from left slit edge in metres)
    :param upper: (float) Upper limit of integration -> +(distance from right slit edge in metres)
    :param D: (float) Aperture to screen distance in metres
    :return: (float) Integration value of the sin part forming the imaginary part of the complex amplitude
    """
    return quad(sin_integrand, lower, upper, args = (D,))[0]

def get_magnitude(real_x, imaginary_y):
    """
    :param real_x: (float) Real part of a complex number
    :param imaginary_y: (float) Imaginary part of a complex number
    :return: (float) Magnitude of a complex number
    """
    z = complex(real_x, imaginary_y)
    return abs(z)

def get_phase(real_x, imaginary_y):
    """
    :param real_x: (float) Real part of a complex number
    :param imaginary_y: (float) Imaginary part of a complex number
    :return: (float) Phase of a complex number in radians
    """
    z = complex(real_x, imaginary_y)
    return cmath.phase(z)

def generate_data():
    """
    :return: (dictionary) Keys are the different D values (aperture to screen distance)
                          and values are arrays of arrays each containing position, magnitude, phase
    """
    # For every axis starting from lower = 0 to lower = -d we generate a complex number z
    # With z we get magnitude and phase for each point on the screen. In the plot we will have
    # the x-axis as the distance from the left edge of the slit with lower = 0 and this will go up to d.
    # The data dictionary will hold for each D value key data for each point on the screen in the form
    # position, magnitude, phase.
    data = {}

    for D_value in D_ARRAY:
        data[D_value] = []
        for point in POINTS:
            x = get_x_value(-point, d - point, D_value)
            y = get_y_value(-point, d - point, D_value)
            mag = get_magnitude(x, y)
            phs = get_phase(x, y)
            data[D_value].append([point, mag, phs])

    return data

# Create the dictionary to hold all the data
data_dict = generate_data()

# Gather magnitude data
magnitude_list_D30 = [elements[1] for elements in data_dict[0.3]]
magnitude_list_D50 = [elements[1] for elements in data_dict[0.5]]
magnitude_list_D100 = [elements[1] for elements in data_dict[1]]

# Gather phase data
phase_list_D30 = [elements[2] for elements in data_dict[0.3]]
phase_list_D50 = [elements[2] for elements in data_dict[0.5]]
phase_list_D100 = [elements[2] for elements in data_dict[1]]

# Plot magnitude
plt.plot(POINTS, magnitude_list_D50)
#plt.plot(POINTS, magnitude_list_D30)
#plt.plot(POINTS, magnitude_list_D100)
plt.legend(['D = 50 cm', 'D = 50 cm', 'D = 100 cm'])
plt.xlabel('Distance from the left slit edge (m)')
plt.ylabel('Relative Magnitude')
plt.title('Magnitude vs Position on the screen')
plt.grid(True)
plt.savefig('magnitude_vs_position.png')
plt.show()

# Plot phase
#plt.plot(POINTS, phase_list_D30)
plt.plot(POINTS, phase_list_D50)
#plt.plot(POINTS, phase_list_D100)
plt.legend(['D = 50 cm', 'D = 50 cm', 'D = 100 cm'])
plt.xlabel('Distance from the left slit edge (m)')
plt.ylabel('Relative Phase (Radians)')
plt.title('Phase vs Position on the screen')
plt.savefig('phase_vs_position.png')
plt.grid(True)
plt.show()
