import numpy as np
import matplotlib.pyplot as plt

# read csv file into np array
def read_csv(filename):
    return np.genfromtxt(filename, delimiter=',')

# plot all the data points and the corresponding line
def make_plot(data, title, coef):
    x_vals = np.zeros(len(data))
    y_vals = np.zeros(len(data))

    x = np.linspace(-2, 2, 100)
    y = 0
    for i in range(len(coef)):
        y += coef[i] * (x**i)

    for i in range(len(data)):
        x_vals[i] = data[i][0]
        y_vals[i] = data[i][1]

    plt.plot(x_vals, y_vals, 'bo')
    plt.plot(x, y, 'r', label='Prediction')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    plt.show()

# get the value from the summation of predicted - actual
def get_error(coef, row):
    prediction = 0
    x = row[0]
    actual = row[1]
    # loop through every coefficient to calculate prediction
    for i in range(len(coef)):
        prediction += coef[i] * (x ** i)
    error = prediction - actual
    return error

# continuously updates weight values using gradient descent, returns final weights and MSE
def update_coef(data, learn_rate, epochs, coef):
    # create array of zeros for MSE tracking
    error_arr = np.zeros(epochs)
    # get length of dataset
    m = len(data)
    # loop through all epochs
    for epoch in range(epochs):
        # reset the error sum for the epoch
        sum_error = 0
        # loop through each row in the dataset
        for row in data:
            # Get summation value here
            err = get_error(coef, row)
            # loop through and update all weights using update formula
            for i in range(len(coef)):
                coef[i] = coef[i] - learn_rate * (1/m) * err * (row[0]**i)
            # add to the sum_error the squared error (MSE)
            sum_error += err**2
        # add the sum_error to the MSE array
        error_arr[epoch] = sum_error
    # calculate mean of MSE array and print value
    mse = round(np.mean(error_arr), 3)
    # return the final weight and mse values
    return coef, mse

# driver for program
def main():
    np.set_printoptions(precision=3)

    # read in all csv files to np arrays
    syn1 = read_csv('./data/synthetic-1.csv')
    syn2 = read_csv('./data/synthetic-2.csv')
    syn3 = read_csv('./data/synthetic-3.csv')

    # create all coef arrays for each order
    coef1 = [0.0, 0.0]
    coef2 = [0.0, 0.0, 0.0]
    coef4 = [0.0, 0.0, 0.0, 0.0, 0.0]
    coef7 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # create dictionary mapping coef array to string that defines it
    coef_arr = [[coef1, '1st order'], [coef2, '2nd order'], [coef4, '4th order'], [coef7, '7th order']]

    # loops through every order for every dataset
    for i in range(len(coef_arr)):
        # set epochs and alpha values
        epochs = 10000
        alpha = .001

        # update and plot for syn1
        coefs, mse = update_coef(syn1, alpha, epochs, coef_arr[i][0])
        print('Weight results for syn1 and ' + coef_arr[i][1] + ':')
        weights = np.array(coefs)
        print(weights)
        print('The MSE of this model is ' + str(mse))
        print()
        make_plot(syn1, "Synthetic 1 - " + coef_arr[i][1], weights)

        # update and plot for syn 2
        coefs, mse = update_coef(syn2, alpha, epochs, coef_arr[i][0])
        print('Weight results for syn2 and ' + coef_arr[i][1] + ':')
        weights = np.array(coefs)
        print(weights)
        print('The MSE of this model is ' + str(mse))
        print()
        make_plot(syn2, "Synthetic 2 - " + coef_arr[i][1], weights)

        # update and plot for syn 3
        coefs, mse = update_coef(syn3, alpha, epochs, coef_arr[i][0])
        print('Weight results for syn3  and ' + coef_arr[i][1] + ':')
        weights = np.array(coefs)
        print(weights)
        print('The MSE of this model is ' + str(mse))
        print()
        make_plot(syn3, "Synthetic 3 - " + coef_arr[i][1], weights)

if __name__== "__main__": main()
