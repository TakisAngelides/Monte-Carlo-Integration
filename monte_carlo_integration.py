# Exercise 1 Core Task 1, Supplementary Task 1: Monte Carlo integration.

# This program calculates the mean value for the integral given using MC integration and for different
# values of N samples. It also makes plots of error to test theoretical statements for the relationship
# of error and N samples. It produces 6 plots and one dataframe summarising all the data.
# The style of the code is designed so as to respect the single responsibility principle and
# high performance vectorised implementation. It is also quite flexible in changing the
# integral's dimensions and other variables of the problem. One thing that can be made more
# general is the limit for each dimension in the integral so that they don't all have to be
# the same, namely from 0 to pi/8. The analytical solution to the problem is approximately 537.1873411.

import numpy as np
from numpy import pi
from numpy import random as rdm
import pandas as pd
import matplotlib.pyplot as plt
import time

start_time = time.time()
VOLUME_DIMENSION = 8
INTEGRATION_LOWER_LIMIT = 0
INTEGRATION_UPPER_LIMIT = pi/8 #0.39269908169872414
LIMIT_RANGE = INTEGRATION_UPPER_LIMIT - INTEGRATION_LOWER_LIMIT
VOLUME = LIMIT_RANGE**VOLUME_DIMENSION
N_SAMPLES = 1000
N_CALLS = 25
TRUE_ANS = 537.1873411

def get_random_numbers(lower = INTEGRATION_LOWER_LIMIT, upper = INTEGRATION_UPPER_LIMIT):
    """
    :param lower: (float) The lower integration limit
    :param upper: (float) The upper integration limit
    :return: (ndarray) An array containing VOLUME_DIMENSION random numbers to be passed into the integrand
    """
    return rdm.uniform(lower, upper, size = VOLUME_DIMENSION)

def integrand(x_input):
    """
    :param x_input: (ndarray) An array of VOLUME_DIMENSION random numbers
    :return: (float) The integrand value at a specific x_input point in the integral's volume space
    """
    return (10**6)*np.sin(np.sum(x_input))

def get_expected_value(n_samples):
    """
    :param n_samples: (int) Number of samples to evaluate the integrand in the integral's volume space
    :return: (float) Returns the expected value of the integrand <f>
    """
    sum_evaluations = 0.0
    for i in range(n_samples):
        x_input = get_random_numbers()
        sum_evaluations += integrand(x_input)
    return sum_evaluations/n_samples

def get_expected_square_value(n_samples):
    """
    :param n_samples: (int) Number of samples to evaluate the integrand in the integral's volume space
    :return: (float) Returns the square expected value of the integrand <f^2>
    """
    sum_evaluations = 0.0
    for i in range(n_samples):
        x_input = get_random_numbers()
        sum_evaluations += integrand(x_input)**2
    return sum_evaluations/n_samples

def integrate(n_samples = N_SAMPLES):
    """
    :param n_samples: (int) Number of samples to evaluate the integrand in the integral's volume space
    :return: (float), (float) Returns a 2-tuple of the estimated value for the integral and its error
    """
    expected_value = get_expected_value(n_samples)
    expected_square_value = get_expected_square_value(n_samples)

    integral_estimate = VOLUME*expected_value
    error_estimate = VOLUME*np.sqrt(abs((expected_square_value - expected_value**2))/n_samples)

    return integral_estimate, error_estimate

def mean_integral_estimate(n_calls = N_CALLS, n_samples = N_SAMPLES):
    """
    :param n_calls: (int) The number of times to call the monte carlo integration function
    :param n_samples: (int) Number of samples to evaluate the integrand in the integral's volume space
    :return: (float) Mean of the outcomes of the integration after n_calls with n_samples flexibilty and rms
    """
    sum_of_integrations = 0.0
    sum_of_squared_errors = 0.0
    mean_values = []

    for i in range(n_calls):
        integration_estimate = integrate(n_samples)[0]
        mean_values.append(integration_estimate)
        integration_error = integrate(n_samples)[1]
        sum_of_integrations += integration_estimate
        sum_of_squared_errors += integration_error**2
    rms = np.sqrt(sum_of_squared_errors/n_calls)
    mean_of_n_calls = sum_of_integrations/n_calls
    standard_deviation = np.std(mean_values)
    return mean_of_n_calls, rms, standard_deviation

# Create an array of n_samples values to try when calling the integrate function
n_samples_list = [100, 200, 300, 400, 600, 1000, 1500, 3000, 5000, 10000, 20000, 45000, 75000, 95000, 100000, 1000000]
# Compute samples^(-1/2) for plotting
n_samples_modified_array = (np.array(n_samples_list))**(-0.5)
# A list to store results of the mean value of the integral
integral_mean_list = []
# A list to store results of the root mean square error of the integral
integral_rms_list = []
# A list to store the standard deviation for each N trial
integral_std_list = []

#Generate data for each N value
for samples in n_samples_list:

    integral_mean, integral_rms, integral_std = mean_integral_estimate(n_samples = samples)
    integral_mean_list.append(integral_mean)
    integral_rms_list.append(integral_rms)
    integral_std_list.append(integral_std)

# Create a dataframe and store the data collected
integral_mean_df = pd.DataFrame(integral_mean_list, columns = ['Mean Integral Value'])
integral_rms_df = pd.DataFrame(integral_rms_list, columns = ['Integral RMS'])
integral_std_df = pd.DataFrame(integral_std_list, columns = ['Integral Std'])
number_of_samples_df = pd.DataFrame(n_samples_list, columns = ['N samples'])
n_samples_modified_array_df = pd.DataFrame(n_samples_modified_array, columns = ['N^(-1/2) samples'])
df = pd.concat([integral_mean_df, integral_rms_df, number_of_samples_df, n_samples_modified_array_df, integral_std_df], axis = 1)
df['Analytical Error'] = abs(df['Mean Integral Value'] - TRUE_ANS)

# Plot log(RMS) vs log(N^(-1/2)) to show their linear relationship
plt.scatter(np.log(df['N^(-1/2) samples']), np.log(df['Integral RMS']), marker = '+', color = 'k')
plt.title('Log(RMS) vs Log(N^(-1/2)) samples)')
plt.xlabel('Log(N^(-1/2)) samples')
plt.ylabel('Log(RMS)')
plt.savefig('log(rms)_vs_log(N^(-0.5)).png')
plt.grid(True)
plt.show()

# Plot log(std) vs log(N^(-1/2)) to show their linear relationship
plt.scatter(np.log(df['N^(-1/2) samples']), np.log(df['Integral Std']), marker = '+', color = 'k')
plt.title('Log(Std) vs Log(N^(-1/2)) samples)')
plt.xlabel('Log(N^(-1/2)) samples')
plt.ylabel('Log(Std)')
plt.savefig('log(Std)_vs_log(N^(-0.5)).png')
plt.grid(True)
plt.show()

# Plot Mean integral value vs log(N samples) with RMS as error bars
plt.xlabel('log(N samples)')
plt.ylabel('Integral value')
plt.title('Mean integral value vs log(N samples)\n(Analytical integral value as horizontal line)')
plt.axhline(y = TRUE_ANS, color = 'b', linestyle = '--', label = 'True integral value')
plt.errorbar(np.log(df['N samples']), df['Mean Integral Value'], yerr = df['Integral RMS'], color = 'k', ecolor = 'r')
plt.legend(['True integral value', 'Mean integral value with RMS'])
plt.savefig('mean_integral_value_vs_N.png')
plt.grid(True)
plt.show()

# Plot error vs N samples for RMS and the true error between the mean integral value and the analytical value
rms_error = df['Integral RMS']
analytical_error = df['Analytical Error']
N = df['N samples']
std_error = df['Integral Std']
plt.plot(N[7:], rms_error[7:], color = 'k', marker = '+')
plt.plot(N[7:], analytical_error[7:], color = 'r', marker = '+')
plt.title('RMS & True Error vs N samples')
plt.xlabel('N samples')
plt.ylabel('Error')
plt.legend(['Theoretical (RMS) Error', 'True (Analytical) Error'])
plt.savefig('rms_error_vs_N_samples.png')
plt.grid(True)
plt.show()

# Plot error vs N samples for Std and the true error between the mean integral value and the analytical value
plt.plot(N[7:], rms_error[7:], color = 'k', marker = '+')
plt.plot(N[7:], analytical_error[7:], color = 'r', marker = '+')
plt.title('Standard deviation & True Error vs N samples')
plt.xlabel('N samples')
plt.ylabel('Error')
plt.legend(['Standard deviation', 'True (Analytical) Error'])
plt.savefig('std_error_vs_N_samples.png')
plt.grid(True)
plt.show()

# Plot rms and std error vs N in a bar plot for comparison
df_test = pd.concat([N, rms_error, std_error], axis = 1)
df_test.columns = ['N', 'rms_error', 'std_error']
ax = df_test.plot.bar(x = 'N')
plt.title('Standard deviation & RMS vs N samples')
plt.xlabel('N samples')
plt.ylabel('Error')
plt.legend(['Standard deviation', 'RMS'])
plt.savefig('std_error_and_rms_vs_N_samples_bar_plot.png')
plt.grid(True)
plt.show()

# Calculate program execution duration
end_time = time.time()
execution_minutes = (end_time - start_time)/60
print(f"This program has taken {execution_minutes:.1f} minutes")
pd.set_option('display.max_columns', 500)
print(df)