# -*- coding: utf-8 -*-
#*************************************************************************************
#* FILE: simulation_3dof_self_balancing_robot.py
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
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from scipy.integrate import odeint
import numpy as np
from numpy import sin, cos, pi
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.transforms import Affine2D
from sympy import symbols, cos, sin, solve, diff, Matrix
from matplotlib.gridspec import GridSpec
#*************************************************************************************
#*                                  Global symbol Section
#*************************************************************************************
m, M, R, n, Jm, L, n, a, beta, fw, g, W, J_psi, Jw, vl, vr, J_phi, x1, x2, \
    x3, x4, x5, x6, x7, x8, x9 = sp.symbols(
    'm M R n Jm L n a beta fw g W J_psi Jw vl vr J_phi x1 x2 x3 x4 x5 x6 x7 x8 x9'
)
#*************************************************************************************
#*                                 Class Definitions
#*************************************************************************************

#*************************************************************************************
#* Class Name    : LQRController
#* Description   : Calc LQR Controller 
#* Argument      : parameter
#* Return        : None
#*************************************************************************************

class ParameterRobot():
    def __init__(self, m_value, M_value, R_value, W_value, D_value, H_value, L_value):
        self.m_value = m_value          # Weight wheel
        self.M_value = M_value
        self.R_value = R_value
        self.W_value = W_value
        self.D_value = D_value
        self.H_value = H_value
        self.L_value = L_value
        self.fw_value = 0.18
        self.fm_value = 0.002
        self.Jm_value = 10**-2
        self.Jw_value = (self.m_value * self.R_value**2) / 2
        self.J_psi_value = (self.M_value * self.L_value**2) / 3
        self.J_phi_value = (self.M_value * (self.W_value**2 + self.D_value**2)) / 12
        self.Rm_value = 50
        self.Kb_value = 0.468
        self.Kt_value = 0.317
        self.n_value = 40
        self.g_value = 9.81
        self.alpha_value = (self.n_value*self.Kt_value)/self.Rm_value; 
        self.beta_value= (self.n_value*self.Kt_value*self.Kb_value)/(self.Rm_value+self.fm_value)
        self.a_value = self.alpha_value
        self.T = 0.01

class StateSpaceModel():
    def __init__(self):
        self.A_Nonlinear = sp.zeros(6, 6)
        self.B_Nonlinear = sp.zeros(6,2)
        self.A = sp.zeros(6, 6)  
        self.B = sp.zeros(6, 2)
        self.K = sp.zeros(6, 2)
        self.second_derivative_robot()
        self.linearized_model()
        self.calc_K_gain()
          
    def second_derivative_robot(self):
        """ Define parameter of Robot
                θ = symbols('theta')  --> ave_wheel_velocity
                ψ = symbols('psi')    --> titl_angle (pitch or roll)
                φ = symbols('phi')    --> rotation angle robot (yaw)
                
                x1 = theta              x4 = psi                x4 = phi
                x2 = theta_dot          x5 = psi_dot            x4 = phi_dot
                x2 = theta_ddot         x6 = psi_ddot           x4 = phi_ddot
                
            Returns:
                Return matrices A, B in nonlinear form
        """
        theta_ddot = (
            (J_psi*a*vl + J_psi*a*vr - 2*J_psi*beta*x2 + 2*J_psi*beta*x5 - 2*J_psi*fw*x2 + L**2*M*a*vl + L**2*M*a*vr - 2*L**2*M*beta*x2 + 
            2*L**2*M*beta*x5 - 2*L**2*M*fw*x2 - 4*Jm*fw*n**2*x2 + L**3*M**2*R*x5**2*sp.sin(x4) - 2*L*M*R*beta*x2*sp.cos(x4) + 
            2*L*M*R*beta*x5*sp.cos(x4) - L**3*M**2*R*x8**2*sp.cos(x4)**2*sp.sin(x4) - L**2*M**2*R*g*sp.cos(x4)*sp.sin(x4) + 
            J_psi*L*M*R*x5**2*sp.sin(x4) + 2*Jm*L*M*g*n**2*sp.sin(x4) + L*M*R*a*vl*sp.cos(x4) + L*M*R*a*vr*sp.cos(x4) + 
            2*Jm*L**2*M*n**2*x8**2*sp.cos(x4)*sp.sin(x4) + 2*Jm*L*M*R*n**2*x5**2*sp.sin(x4)) /
            (2*J_psi*Jw + L**2*M**2*R**2 + 2*Jw*L**2*M + J_psi*M*R**2 + 2*J_psi*Jm*n**2 + 4*Jm*Jw*n**2 + 2*J_psi*R**2*m + 2*Jm*L**2*M*n**2 + 
            2*Jm*M*R**2*n**2 + 2*L**2*M*R**2*m + 4*Jm*R**2*m*n**2 - L**2*M**2*R**2*sp.cos(x4)**2 + 4*Jm*L*M*R*n**2*sp.cos(x4))
        )
        psi_ddot = (
            -(2*Jw*a*vl + 2*Jw*a*vr - 4*Jw*beta*x2 + 4*Jw*beta*x5 + M*R**2*a*vl + M*R**2*a*vr - 2*M*R**2*beta*x2 + 2*M*R**2*beta*x5 + 
            2*R**2*a*m*vl + 2*R**2*a*m*vr + 4*Jm*fw*n**2*x2 - 4*R**2*beta*m*x2 + 4*R**2*beta*m*x5 - L*M**2*R**2*g*sp.sin(x4) - 
            2*Jw*L*M*g*sp.sin(x4) - 2*L*M*R*beta*x2*sp.cos(x4) + 2*L*M*R*beta*x5*sp.cos(x4) - 2*L*M*R*fw*x2*sp.cos(x4) +
            L**2*M**2*R**2*x5**2*sp.cos(x4)*sp.sin(x4) - L**2*M**2*R**2*x8**2*sp.cos(x4)*sp.sin(x4) - 2*Jw*L**2*M*x8**2*sp.cos(x4)*sp.sin(x4) - 
            2*Jm*L*M*g*n**2*sp.sin(x4) - 2*L*M*R**2*g*m*sp.sin(x4) + L*M*R*a*vl*sp.cos(x4) + L*M*R*a*vr*sp.cos(x4) - 
            2*Jm*L**2*M*n**2*x8**2*sp.cos(x4)*sp.sin(x4) - 2*L**2*M*R**2*m*x8**2*sp.cos(x4)*sp.sin(x4) - 2*Jm*L*M*R*n**2*x5**2*sp.sin(x4)) /
            (2*J_psi*Jw + L**2*M**2*R**2 + 2*Jw*L**2*M + J_psi*M*R**2 + 2*J_psi*Jm*n**2 + 4*Jm*Jw*n**2 + 2*J_psi*R**2*m + 2*Jm*L**2*M*n**2 + 
            2*Jm*M*R**2*n**2 + 2*L**2*M*R**2*m + 4*Jm*R**2*m*n**2 - L**2*M**2*R**2*sp.cos(x4)**2 + 4*Jm*L*M*R*n**2*sp.cos(x4))
        )

        phi_ddot = (
            -(W**2*beta*x8 + W**2*fw*x8 + R*W*a*vl - R*W*a*vr + 4*L**2*M*R**2*x5*x8*sp.cos(x4)*sp.sin(x4)) /
            (2*J_phi*R**2 + Jw*W**2 + Jm*W**2*n**2 + R**2*W**2*m + 2*L**2*M*R**2*sp.sin(x4)**2)
        )
        f1 = x2
        f2 = theta_ddot
        f3 = x5
        f4 = psi_ddot
        f5 = x8
        f6 = phi_ddot

        self.A_Nonlinear = Matrix([
                    [diff(f1, x) for x in [x1, x2, x4, x5, x7, x8]],
                    [diff(f2, x) for x in [x1, x2, x4, x5, x7, x8]],
                    [diff(f3, x) for x in [x1, x2, x4, x5, x7, x8]],
                    [diff(f4, x) for x in [x1, x2, x4, x5, x7, x8]],
                    [diff(f5, x) for x in [x1, x2, x4, x5, x7, x8]],
                    [diff(f6, x) for x in [x1, x2, x4, x5, x7, x8]]
                ])
        
        
        self.B_Nonlinear = Matrix([
                    [diff(f1, x) for x in [vl, vr]],
                    [diff(f2, x) for x in [vl, vr]],
                    [diff(f3, x) for x in [vl, vr]],
                    [diff(f4, x) for x in [vl, vr]],
                    [diff(f5, x) for x in [vl, vr]],
                    [diff(f6, x) for x in [vl, vr]]
                ])

    def linearized_model(self):
        robot = ParameterRobot(m_value=0.15, M_value=2.0, R_value=0.065, W_value=0.22, D_value=0.12, H_value=0.2, L_value=0.08)
        #robot = ParameterRobot(m_value=1.0, M_value=5.0, R_value=0.0725, W_value=0.24, D_value=0.2, H_value=0.5, L_value=0.18)
        state_linear = {x1: 0.0, x2: 0.0, x4: 0.0, x5: 0.5, x7: 0.0, x8: 0.0, 
                        vl: 0.0, vr: 0.0,
                        m: robot.m_value,  M: robot.M_value, R: robot.R_value, W: robot.W_value, L: robot.L_value,
                        Jw: robot.Jw_value, Jm: robot.Jm_value, J_psi: robot.J_psi_value, J_phi: robot.J_phi_value,
                        n: robot.n_value, a: robot.a_value, beta: robot.beta_value, fw: robot.fw_value, g: robot.g_value}

        self.A = self.A_Nonlinear.subs(state_linear)
        self.A = np.array(self.A).astype(np.float64)
        self.B = self.B_Nonlinear.subs(state_linear)
        self.B = np.array(self.B).astype(np.float64)

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
    
    def calc_K_gain(self):
        self.Q = np.array([ [1, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 1]])
        
        self.R = np.array([ [1, 0],
                            [0, 1]])
        self.K, _, _ = self.lqr()
    
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
        """error = np.array(y) - np.array(y_desired)
        x_dd = np.dot((self.A - np.dot(self.B, self.K)), error)
        return x_dd"""
    
    def msd_desired(self, y, t, y_desired):
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
        #x_dd = np.dot((self.A - np.dot(self.B, self.K)), y)
        #return x_dd
        error = np.array(y) - np.array(y_desired)
        x_dd = np.dot((self.A - np.dot(self.B, self.K)), error)
        return x_dd
        
class RobotAnimation:
    def __init__(self, solution, t_vec, r, l, d, h, title="Robot Animation"):
        self.solution = solution
        self.t_vec = t_vec
        self.r = r
        self.l = l
        self.d = d
        self.h = h
        self.title = title
        self.fig = plt.figure(figsize=(14, 8))
        gs = GridSpec(4, 4, height_ratios=[1.2, 1, 1, 1], width_ratios=[3, 1, 1, 1])

        x_min = 0
        x_max = max(self.t_vec)
        self.ax_robot = self.fig.add_subplot(gs[0, :])  
        self.ax_path = self.fig.add_subplot(gs[1:4, :2])
        
        self.ax_theta = self.fig.add_subplot(gs[1, 2:4])
        self.ax_theta.set_xlim(x_min, x_max)
        self.ax_theta.set_ylim(np.min(self.solution[:, 0]) - 0.2, np.max(self.solution[:, 0]) + 0.5)
        self.ax_theta.set_ylabel("Angle (rad)", labelpad=5)  
        self.ax_theta.set_xlabel("Time (s)", labelpad=3)
        
        self.ax_psi = self.fig.add_subplot(gs[2, 2:4])
        self.ax_psi.set_xlim(x_min, x_max)
        self.ax_psi.set_ylim(np.min(self.solution[:, 2]) - 0.2, np.max(self.solution[:, 2]) + 0.5)
        self.ax_psi.set_ylabel("Angle (rad)", labelpad=5)  
        self.ax_psi.set_xlabel("Time (s)", labelpad=3)
        
        self.ax_phi = self.fig.add_subplot(gs[3, 2:4])
        self.ax_phi.set_xlim(x_min, x_max)
        self.ax_phi.set_ylim(np.min(self.solution[:, 4]) - 0.2, np.max(self.solution[:, 4]) + 0.5)
        self.ax_phi.set_ylabel("Angle (rad)", labelpad=5)  
        self.ax_phi.set_xlabel("Time (s)", labelpad=3)
        
        self.lines = []
        self.lines.append(self.ax_theta.plot([], [], 'r-')[0]) 
        self.lines.append(self.ax_psi.plot([], [], 'g-')[0])
        self.lines.append(self.ax_phi.plot([], [], 'b-')[0])
        
        theta_unicode = '\u03B8' 
        psi_unicode = '\u03C8'   
        phi_unicode = '\u03C6'   
        line1, = self.ax_theta.plot(0, 0, color='red', label=f'{theta_unicode} (Wheel rotation angle)')
        line2, = self.ax_psi.plot(0, 0, color='green', label=f'{psi_unicode} (Robot tilt angle)')
        line3, = self.ax_phi.plot(0, 0, color='blue', label=f'{phi_unicode} (Robot rotation angle)')

        self.ax_theta.legend()
        self.ax_psi.legend()
        self.ax_phi.legend()
        
    def setup_plot(self, ax, title):
        ax.set_title(title)
        line, = ax.plot([], [], lw=2)
        return line
    
    def animate_theta(self, i):
        """Update plot theta."""
        self.lines[0].set_data(self.t_vec[:i], self.solution[:i, 0])
        return [self.lines[0]]

    def animate_psi(self, i):
        """Update plot psi."""
        self.lines[1].set_data(self.t_vec[:i], self.solution[:i, 2])
        return [self.lines[1]]

    def animate_phi(self, i):
        """Update plot phi."""
        self.lines[2].set_data(self.t_vec[:i], self.solution[:i, 4])
        return [self.lines[2]]
    
    def setup_plot_robot(self):
        self.ax_robot.set_xlim(-0.5, 5.0)
        self.ax_robot.set_ylim(-0.1, 0.4)
        self.ax_robot.set_aspect('equal')
        self.ax_robot.grid(True)
        self.ax_robot.set_title(self.title)

        self.body_robot = plt.Rectangle((0, 0), self.d, self.h, fill=True, color='red')
        self.wheel = plt.Circle((0.0, self.r), self.r, color='black', fill=True, lw=2)
        self.wheel_com = plt.Circle((0.0, self.r), self.r/2, color='blue', fill=True, lw=2)
        self.wheel_spot = plt.Circle((0.0, 0.7*self.r), 0.002, color='yellow')
        self.mass = plt.Circle((0.0, 0.0), 0.005, color='green')
        self.angle_text = self.ax_robot.text(0.05, 0.05, '', transform=self.ax_robot.transAxes)
    
    def setup_plot_trajectory(self):
        """Add another plot animation trajectory of robot"""
        self.ax_path.set_xlim(-0.5, 1.0)
        self.ax_path.set_ylim(-0.1, 0.5)
        self.ax_path.set_aspect('equal')
        self.ax_path.grid(True)
        self.ax_path.set_title("Tracking Trajectory")
        self.path_line, = self.ax_path.plot([], [], 'b-', label='Path')
        self.yaw_text = self.ax_path.text(0.05, 0.05, '', transform=self.ax_path.transAxes)
        self.ax_path.legend()

    def init(self):
        return []
    
    def init_path(self):
        """Init trajectory of robot"""
        self.path_line.set_data([], [])
        return self.path_line,

    def calculate_position(self, theta, phi, wheel_radius):
        """
        Calculate x,y postion of robot arcording theta and yaw angle

        Parameters:
        - theta: rotation angle of wheel (rad).
        - phi: rotation angle of robot (rad).
        - wheel_radius: radius of ưheel (m) 

        Returns:
        - x, y: Position x, y of robot.
        """
        distance = theta * wheel_radius

        x = distance * np.cos(phi)
        y = distance * np.sin(phi)

        return x, y

    def animate_path(self, i):
        """Update trajectory robot"""
        x, y = self.calculate_position(self.solution[i, 0], self.solution[i, 4], self.r)
        old_x, old_y = self.path_line.get_data()
        new_x, new_y = np.append(old_x, x), np.append(old_y, y)
        self.path_line.set_data(new_x, new_y)
        yaw_degrees = np.rad2deg(self.solution[i, 4]) # Convert yaw rad to degree
        self.yaw_text.set_text('Rotate angle: {:.2f}°'.format(yaw_degrees))
        return [self.path_line, self.yaw_text]
    
    def animate_robot(self, i):
        current_time = self.t_vec[i]
        tilt_angle = self.solution[i, 2]        # Tilt angle of the robot -> pitch
        theta = self.solution[i, 0] / self.r    # Wheel rotation angle
        yaw = self.solution[i, 4]               # Rotation angle of the robot -> yaw

        wheel_x = theta * self.r
        wheel_spot_x = wheel_x + 0.7 * self.r * cos(theta - pi / 2)
        wheel_spot_y = self.r - 0.7 * self.r * sin(theta - pi / 2)
        mass_x = wheel_x + self.l * cos(tilt_angle - pi / 2)
        mass_y = self.r - self.l * sin(tilt_angle - pi / 2)
        
        self.wheel.set_center((wheel_x, self.r))
        self.wheel_com.set_center((wheel_x, self.r))
        self.wheel_spot.set_center((wheel_spot_x, wheel_spot_y))
        self.mass.set_center((mass_x, mass_y))
        self.body_robot.set_xy((wheel_x - self.d/2, self.r))
        transform = Affine2D().rotate_around(wheel_x, self.r, tilt_angle) + self.ax_robot.transData
        self.body_robot.set_transform(transform)
        
        angle_degrees = np.rad2deg(tilt_angle)
        self.angle_text.set_text('Tilt angle: {:.2f}°'.format(angle_degrees))

        patches = [self.ax_robot.add_patch(self.body_robot), self.ax_robot.add_patch(self.wheel),
                   self.ax_robot.add_patch(self.wheel_com), self.ax_robot.add_patch(self.wheel_spot),
                   self.ax_robot.add_patch(self.mass), self.angle_text]
        return patches

    def show(self):
        self.setup_plot_robot()
        self.setup_plot_trajectory()
            
        ani_theta = animation.FuncAnimation(self.fig, self.animate_theta, frames=len(t_vec), interval=10, blit=True)
        ani_psi = animation.FuncAnimation(self.fig, self.animate_psi, frames=len(t_vec), interval=10, blit=True)
        ani_phi = animation.FuncAnimation(self.fig, self.animate_phi, frames=len(t_vec), interval=10, blit=True)
        
        ani_robot = animation.FuncAnimation(self.fig, self.animate_robot, np.arange(1, len(self.solution)),
                                      interval=10, blit=True, init_func=self.init)
        ani_path = animation.FuncAnimation(self.fig, self.animate_path, np.arange(1, len(self.solution)),
                                 interval=10, blit=True, init_func=self.init_path)
        self.fig.tight_layout()
        
        # Save gif file.
        # ani_robot.save('3dof_self_balancing_robot.gif', writer='imagemagick', fps=60)
        # plt.close(self.fig)
        plt.show()
    
if __name__ == "__main__":
    lqr_controller = StateSpaceModel()
    angle = np.deg2rad(0.0002)
    # X = [ave_wheel, ave_wheel_dot, pitch, pitch_dot, yaw, yaw_dot]
    #y_desired = [0.0001, -0.00012, 0.0002, -0.0002, 0.0002, 0.0002] # Case robot in position balancing 
    y_desired = [3.14, 1.0, 0.0002, -0.0002, 1.57, 0.0002]
    y_0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    fs = 100
    dt = 1/fs
    t_sim = 100
    t_vec = np.arange(0,t_sim,dt)
    msd_with_desired = lambda y, t: lqr_controller.msd_desired(y, t, y_desired)
    solution = odeint(msd_with_desired, y_0, t_vec)
    #solution = odeint(lqr_controller.msd, y_desired, t_vec)
    param = ParameterRobot(m_value=0.15, M_value=2.0, R_value=0.065, W_value=0.22, D_value=0.12, H_value=0.2, L_value=0.08)
    animation_robot = RobotAnimation(solution=solution,t_vec=t_vec,r=param.R_value, l=param.L_value, d=param.D_value, h=param.H_value, title="Self Balancing Robot Animation 3-DOF")
    animate = animation_robot.show()