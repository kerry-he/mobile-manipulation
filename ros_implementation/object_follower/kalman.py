from black import out
from filterpy.kalman import KalmanFilter
import numpy as np, glob
import matplotlib.pyplot as plt


for file in glob.glob("./Data/*.csv"):


    f = KalmanFilter (dim_x=2, dim_z=1)

    initial_state_set = False

    # transition matrix
    f.F = np.array([[1.,1/60],
                    [0.,1.]])

    # measurement function
    f.H = np.array([[1.,0.]])

    # covairance function
    f.P *= 0.00001
    # low measurement noise
    f.R = 0.00001

    velocity = []
    velocity_estimate = []

    pos = []
    pos_estimate = []

    from filterpy.common import Q_discrete_white_noise
    f.Q = Q_discrete_white_noise(dim=2, dt=0.1, var=0.13)
    prev_angle = None
    output = []
    print(f"Starting file: {file}")
    for index,i in enumerate(open(file)):
        if index == 0:
            output.append(f"{i.strip()},kalman_pos, kalman_vel")
            continue

        z = float(i.split(",")[-2])
        if not initial_state_set:
            initial_state_set = True
            # initial state
            f.x = np.array([z, 0.])
            prev_angle = z
        else:
            f.predict()

            f.update([z])
            # print(f.x, z)

            velocity.append(f.x[1])
            velocity_estimate.append((z - prev_angle) / (1/60))

            pos.append(f.x[0])
            pos_estimate.append(z)

            prev_angle = z

            # input()
            output.append(f"{i.strip()},{f.x[0]},{f.x[1]}")

    x = open(file.replace("Data", "Data2"), "w")
    x.write("\n".join(output))
    x.close()

    plt.subplot(1, 2, 1)
    plt.plot(velocity, label="kalman filter")
    plt.plot(velocity_estimate, label="measured")
    plt.legend()


    plt.subplot(1, 2, 2)
    plt.plot(pos, label="kalman filter")
    plt.plot(pos_estimate, label="measured")


    plt.legend()
    plt.savefig(file + ".png")
    plt.clf()