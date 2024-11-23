# -*- coding: utf-8 -*-
#*************************************************************************************
#* FILE: self_balancing_robot_lqr_controller.py
#* AUTHOR: Khanh Phan ## https://www.linkedin.com/in/ptmkhanh29/
#* DATE: Dec 16, 2023
#* DESCRIPTION: - Animation Self Balancing Robotusing LQR Controller
#*              
#* REVISION HISTORY:
#* - V1.0.0:  20-Sept-2023  : Initial Version
#* - V1.0.1:  16-Dec-2023   : Improve LQR Controller
#*************************************************************************************
#*                                  Import Section
#*************************************************************************************
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from scipy.integrate import odeint
import numpy as np
from numpy import sin, cos, pi
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.transforms import Affine2D
#*************************************************************************************
#*                                 Class Definitions
#*************************************************************************************

#*************************************************************************************
#* Class Name    : LQRController
#* Description   : Calc LQR Controller 
#* Argument      : parameter
#* Return        : None
#*************************************************************************************
class LQRController:
    def __init__(self, Mw, Mb, Ib, Iw, l, b, r):
        # Init paramters of modeling robot
        self.Mw = Mw
        self.Mb = Mb
        self.Ib = Ib
        self.Iw = Iw
        self.l = l
        self.b = l
        self.g = 9.8 
        self.r = r
        self.k = self.Mb**2 * self.l**2 / (self.Mb * self.l**2 + self.Ib)
        self.a = self.Mb + 2 * self.Mw + 2 * self.Iw / self.r**2

        #! Create A (State Matrix) [linear_velocity, linear_velocity_dot, titl_angle, titl_angle_dot]
        #! Cretae B (Control Matrix) [force, moment,...]
        #! Find Q and R
        self.A = np.array([[0, 1, 0, 0],
                           [0, 0, -self.k * self.g / (self.a - self.k), 0],
                           [0, 0, 0, 1],
                           [0, 0, self.k * self.g * self.a / (self.Mb * self.l * (self.a - self.k)), 0]])

        self.B = np.array([[0],
                           [(1/self.r + self.k/(self.Mb*self.l)) / (self.a - self.k)],
                           [0],
                           [-self.k/(self.Mb*self.l)**2 - (self.k/(self.Mb*self.l)) * (1/self.r + self.k/(self.Mb*self.l))/(self.a - self.k)]])

        self.Q = np.array([[2, 0, 0, 0],
                           [0, 2, 0, 0],
                           [0, 0, 6, 0],
                           [0, 0, 0, 6]])

        self.R = np.array([[1000]])

        # Calc K gain LQR
        self.K, _, _ = self.lqr()
        print(f"K matrix gain = {self.K}")

    def lqr(self):
        """     
        Solves the continuous time Linear Quadratic Regulator (LQR) problem.
            dx/dt = A x + B u
        The LQR controller design is to minimize the cost function:
            cost = integral (x.T*Q*x + u.T*R*u) dt
        where x is the state vector and u is the control input.

        The controller solves the following continuous-time Algebraic Riccati Equation (ARE):
            A.T*X + X*A - X*B*R^-1*B.T*X + Q = 0
        Steps:
            1. Solve the continuous-time Algebraic Riccati Equation (ARE) to find matrix X.
            2. Compute the LQR gain, K, using the formula K = R^-1 * B.T * X.
            3. Calculate the eigenvalues (eigVals) of the closed-loop system (A - B*K) to analyze the system's stability.
        """
        X = np.matrix(scipy.linalg.solve_continuous_are(self.A, self.B, self.Q, self.R))
      
        K = np.matrix(scipy.linalg.inv(self.R)*(self.B.T*X))
        
        eigVals, eigVecs = scipy.linalg.eig(self.A - self.B * K)
        
        return np.array(K), np.array(X), np.array(eigVals)

    def msd(self, y, t):
        """
        This function represents the system dynamics.

        The dynamics are described by the differential equation:
        dx/dt = Ax + Bu (u = -Kx)
        ==> dx/dt = (A - B*K)*x

        Where:
        - x is the current state of the system.
        - A is the state matrix, representing the system dynamics.
        - B is the control matrix, linking the control input to the state.
        - K is the LQR controller gain, calculated to optimize the control performance.
        - dx/dt represents the rate of change of the system's state.

        Parameters:
        - y: The current state of the system (equivalent to x in the equation).
        - t: Time variable (not directly used in the calculation but can be used for time-dependent dynamics).

        The function computes the change in the state (dx/dt) based on the current state and the effect of the LQR controller.
        """
        x_dd = np.dot((self.A - np.dot(self.B, self.K)), y)
        return x_dd

#*************************************************************************************
#* Class Name    : RobotAnimation
#* Description   : Animation Robot implement LQR
#* Argument      : parameter
#* Return        : None
#*************************************************************************************
class RobotAnimation:
    def __init__(self, solution, r, l, b, title="Robot Animation"):
        self.solution = solution
        self.r = r
        self.l = l
        self.b = b
        self.title = title
        self.fig, self.ax = plt.subplots()
        self.setup_plot()

    def setup_plot(self):
        self.ax.set_xlim(-0.1, 0.5)
        self.ax.set_ylim(-0.1, 0.2)
        self.ax.set_aspect('equal')
        self.ax.grid(True)
        self.ax.set_title(self.title)

        self.body_robot = plt.Rectangle((0, 0), self.b, self.l+0.03, fill=True, color='red')
        self.wheel = plt.Circle((0.0, self.r), self.r, color='black', fill=True, lw=2)
        self.wheel_com = plt.Circle((0.0, self.r), self.r/2, color='blue', fill=True, lw=2)
        self.wheel_spot = plt.Circle((0.0, 0.7*self.r), 0.002, color='yellow')
        self.mass = plt.Circle((0.0, 0.0), 0.005, color='green')
        self.angle_text = self.ax.text(0.05, 0.05, '', transform=self.ax.transAxes)

    def init(self):
        return []

    def animate(self, i):
        theta = self.solution[i, 2]             # Tilt angle of the robot
        phi = self.solution[i, 0] / self.r      # Wheel rotation angle

        wheel_x = phi * self.r
        wheel_spot_x = wheel_x + 0.7 * self.r * cos(phi - pi / 2)
        wheel_spot_y = self.r - 0.7 * self.r * sin(phi - pi / 2)
        mass_x = wheel_x + self.l * cos(theta - pi / 2)
        mass_y = self.r - self.l * sin(theta - pi / 2)

        self.wheel.set_center((wheel_x, self.r))
        self.wheel_com.set_center((wheel_x, self.r))
        self.wheel_spot.set_center((wheel_spot_x, wheel_spot_y))
        self.mass.set_center((mass_x, mass_y))
        self.body_robot.set_xy((wheel_x - self.b/2, self.r))
        transform = Affine2D().rotate_around(wheel_x, self.r, theta) + self.ax.transData
        self.body_robot.set_transform(transform)
        
        angle_degrees = np.rad2deg(theta)
        self.angle_text.set_text('Tilt angle: {:.2f}°'.format(angle_degrees))

        patches = [self.ax.add_patch(self.body_robot), self.ax.add_patch(self.wheel),
                   self.ax.add_patch(self.wheel_com), self.ax.add_patch(self.wheel_spot),
                   self.ax.add_patch(self.mass), self.angle_text]
        return patches

    def show(self):
        ani = animation.FuncAnimation(self.fig, self.animate, np.arange(1, len(self.solution)),
                                      interval=25, blit=True, init_func=self.init)
        #plt.show()
        fs = 50
        ani.save('lqr_controller_robot.gif', writer='imagemagick', fps=fs)

if __name__ == "__main__":
    'Main Function'
    Mw = 0.033
    Mb = 1.083
    Ib = 0.00555
    Iw = 133.85 * 10**-7
    l = 0.0615
    b = 0.03
    g = 9.8
    r = 0.03
    lqr_controller = LQRController(Mw=Mw, Mb=Mb, Ib=Ib, Iw=Iw, l=l, b=b, r=r)
    angle = np.deg2rad(10)
    initial_state = [0.1, 0.0, angle, 0.0]
    fs = 50
    dt = 1/fs
    t_sim = 10
    t_vec = np.arange(0,t_sim,dt)
    solution = odeint(lqr_controller.msd,initial_state,t_vec) #return parameter [x, x_dot, titl, titl_dot]
    animation_robot = RobotAnimation(solution=solution,r=r, l=l, b=b, title="Self Balancing Robot Animation 2-DOF")
    ani = animation_robot.show()