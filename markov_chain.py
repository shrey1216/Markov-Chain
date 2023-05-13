import numpy as np
import matplotlib.pyplot as plt


# A function which takes in n for the vector and matrix size for p and P respectively, and N
# which dictates the number of steps when computing the transition P.T * p.
# The function plots the norm of p - p_stationary, p being the probability distribution
# at each step and p_stationary being the eigenvector corresponding to the max eigenvalue
# of P.T with its values scaled to have a sum of 1.
def markov_chain(n, N):
    # 1 Constructing a random n vector and scaling its elements so their sum is 1
    # This is probability vector p
    p = np.random.random(n)

    # scale elements so their sum is 1
    p = p / np.sum(p)

    # 2 Forming a random nxn matrix and scaling its elements so the sum of each row is 1
    # This is transition matrix P
    P = np.random.random((n, n))

    # scale elements so the sum of each row is 1
    for x in range(n):
        P[x] = P[x] / np.sum(P[x])

    # 3 Computing the transition for N steps using P.T * p
    # for x in range(N):
    #    p = np.matmul(P.T, p)

    # 4 Computing the eigenvector of P.T corresponding to the largest eigenvalue.
    # Rescale the eigenvector so it adds up to 1. The answer should equal p after N steps.
    evals, evects = np.linalg.eig(P.T)

    # removing the imaginary parts from each element
    evals = evals.real
    evects = evects.real

    # finding the vector which corresponds to the max eigenvalue and assigning it to p_stationary
    p_stationary = evects[:, np.argmax(evals)]

    # scaling p_stationary so its sum is 1
    p_stationary = p_stationary / np.sum(p_stationary)

    # 5 Move 3 down and change the loop to find the norm of p - p stationary
    # plot the norms against i
    p_norm = np.array([])

    for i in range(N):
        # p should get closer to p_stationary each time
        p = np.matmul(P.T, p)

        # p_norm should decrease each time
        p_norm = np.append(p_norm, np.linalg.norm(p - p_stationary))

    # plotting the values of p_norm against i
    plt.plot(p_norm)
    plt.ylabel('Norm of p - p_stationary')
    plt.show()


# 6 Testing the code
# calling the function when n is 5 and N is 50
markov_chain(5, 50)

# other tests
markov_chain(2, 20)
markov_chain(3, 15)
markov_chain(4, 10)
markov_chain(10, 100)
markov_chain(7, 6)

# The norm difference of p-p_stationary very obviously decreases as we can see from the graphs.
