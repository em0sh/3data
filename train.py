###########################
# Model Training
##########################

import numpy as np

# TODO:
    # Stochastic Gradient Descent
        # Cost Function: C(w,b) = (1/2n) SUM(||y(x)-a||^2, x)
        # Activation: w1*x1 + w2*x2 + w3*x3 ... = y1
    # Visualize training (Chart, bars, etc)
    # Back propogation (Training)

class Train:
    def norm(m, val):
        # Normalize according to specified 'm'ode:
            # s -> sigmoid
            # r -> rectified linear 
        # Sigmoid
        if m == 's':
            return 1 / (1 + np.exp(-val))

        # RELU
        elif m == 'r':
            val = abs(val)
            if val > 2.5:
                return 2.5
            else:
                return val
