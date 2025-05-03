import control as ct
import control.optimal as opt
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
print("GOGOGO!!!")

"""
OCP utils
"""
class cartpole():
    def __init__(self, params=[9.8, 1.0, 0.1, 0.5]):
        self.params = params
        self.force_mag = 100.
        self.theta_threshold_radians = np.pi
        self.x_threshold = 2.4
        self.max_velocity = 10
        self.dt = 0.05
        self.lower = -self.force_mag
        self.upper = self.force_mag
        self.goal_state = [0., 0., 1., 0., 0.]
        self.goal_weights = [0.1, 0.1, 1., 1., 0.1]
        self.ctrl_penalty = 0.001

    def get_update_func(self):
        def cartpole_update(t, states, inputs, params):
            gravity, masscart, masspole, length = params
            total_mass = masspole + masscart
            polemass_length = masspole * length

            u = inputs[0]
            u = np.clip(u, self.lower, self.upper)

            x, dx, cos_th, sin_th, dth = states
            th = np.arctan2(sin_th, cos_th)

            cart_in = (u + polemass_length * dth**2 * sin_th) / total_mass
            th_acc = (gravity * sin_th - cos_th * cart_in) / \
                (length * (4./3. - masspole * cos_th**2 / total_mass))
            xacc = cart_in - polemass_length * th_acc * cos_th / total_mass

            x = x + self.dt * dx
            dx = dx + self.dt * xacc
            th = th + self.dt * dth
            dth = dth + self.dt * th_acc

            return np.array([x, dx, np.cos(th), np.sin(th), dth])
        return cartpole_update
    
    def get_frame(self, state, ax=None):
        x, dx, cos_th, sin_th, dth = state
        gravity, masscart, masspole, length = self.params

        th = np.arctan2(sin_th, cos_th)
        th_x = sin_th*length
        th_y = cos_th*length

        if ax is None:
            fig, ax = plt.subplots(figsize=(6,6))
        else:
            fig = ax.get_figure()
        ax.plot((x,x+th_x), (0, th_y), color='k')
        ax.set_xlim((-length*2, length*2))
        ax.set_ylim((-length*2, length*2))
        return fig, ax

    def get_true_obj(self):
        q = np.concatenate((self.goal_weights, [self.ctrl_penalty]))
        p = -np.sqrt(self.goal_weights) * self.goal_state
        return q, p
    
    def get_system(self):
        cartpole_update = self.get_update_func()
        cartpole_sys = ct.nlsys(updfcn=cartpole_update, outfcn=cartpole_update, inputs=1, outputs=5, states=5, params=self.params, name='cartpole_sys', dt=1)
        return cartpole_sys

def uniform(shape, low, high):
    r = high - low
    return np.random.rand(*shape) * r + low

def cartpole_initx(n_batch):
    th = uniform((n_batch, 1), -2*np.pi, 2*np.pi)
    thdot = uniform((n_batch, 1), -.5, .5)
    x = uniform((n_batch, 1), -0.5, 0.5)
    xdot = uniform((n_batch, 1), -0.5, 0.5)
    xinit = np.concatenate((x, xdot, np.cos(th), np.sin(th), thdot), axis=1)
    return xinit

def solve_ocp(cartpole_sys, timepts, x0, Q, R, Qf, lower, upper):
    """
        Q: n_array, n_state x n_state
        R: n_array, n_ctrl x n_ctrl
        Qf: n_array, n_state x n_state
        lower: n_array, n_ctrl x 1
        upper: n_array, n_ctrl x 1
    """
    constraints = [opt.input_range_constraint(cartpole_sys, lower, upper)]
    running_cost = opt.quadratic_cost(cartpole_sys, Q, R)
    terminal_cost = opt.quadratic_cost(cartpole_sys, Qf, None)
    result = opt.solve_ocp(cartpole_sys, timepts, x0, running_cost, constraints, terminal_cost)
    return result

"""
MPC utils
"""
def cost_cartpole(x, u, Q, R, Qf, is_terminal):
    if is_terminal:
        return torch.matmul(torch.matmul(torch.tensor(x).unsqueeze(0), Qf), torch.tensor(x).unsqueeze(1)).squeeze(0).squeeze(0)
    else:
        return torch.matmul(torch.matmul(torch.tensor(x).unsqueeze(0), Q), torch.tensor(x).unsqueeze(1)).squeeze(0).squeeze(0) + \
               torch.matmul(torch.matmul(torch.tensor(u).unsqueeze(0), R), torch.tensor(u).unsqueeze(1)).squeeze(0).squeeze(0)

def VN_cartpole(x, u, Q, R, Qf):
    VN = 0
    T = x.shape[1]
    for t in range(T):
        if t == T - 1:
            VN += cost_cartpole(x[:, t], u[:, t], Q, R, Qf, True)
        else:
            VN += cost_cartpole(x[:, t], u[:, t], Q, R, Qf, False)
    return VN

def RDP_criteria_cartpole(VN_list, x_list, u_list, alpha, Q, R, Qf, MPC_T, func, test=False, log_path=None):
    loss = 0
    for i in range(MPC_T - 1):
        RDP = (VN_list[i+1] + alpha * cost_cartpole(x_list[i], u_list[i], Q, R, Qf, False)) - VN_list[i] # Wish RDP <= 0

        if test:
            if log_path is not None:
                with open(log_path, 'a') as f:
                    f.write(f'RDP{i}: {RDP}\n')
        loss += func(RDP)
    return loss

def mpc_cartpole(cartpole_sys, Q, R, Qf, MPC_T, T, x_init, u_lower, u_upper):
    x = x_init

    timepts = np.arange(0, T, 1)
    lower = u_lower
    upper = u_upper

    x_list = []
    u_list = []
    VN_list = []
    for i in range(MPC_T):
        result = solve_ocp(cartpole_sys, timepts, x, Q.data, R.data, Qf.data, lower, upper)
        print(f"MPC Timestamp{i} success? : ", result.success)
        u = result.inputs[:, 0] # u: 1 x n_ctrl list
        x_list.append(x)
        u_list.append(u)
        x = result.states[:, 1] # x: 1 x n_state list
        VN_ = VN_cartpole(result.states, result.inputs, Q, R, Qf)
        VN_list.append(VN_)
    return x_list, u_list, VN_list

if __name__ == '__main__':
    # System params
    DTYPES = torch.float64
    torch.set_default_dtype(DTYPES)
    cartpole_ori = cartpole()
    cartpole_sys = cartpole_ori.get_system()
    Q = nn.Parameter(torch.randn(5, 5))
    R = torch.Tensor([[1.]])
    F = nn.Parameter(torch.randn(5, 5))
    MPC_T = 100
    T = 30
    u_lower = cartpole_ori.lower
    u_upper = cartpole_ori.upper

    # Exp params
    epochs = 500
    batch_size = 2
    optimizer = torch.optim.Adam([Q, F], lr=0.01)
    test_name = 'Initial'
    log_path = './' + f'/{test_name}.txt'

    Q_list = []
    F_list = []
    # Train
    for epoch in tqdm(range(epochs)):
        Q_list.append(Q.data)
        F_list.append(F.data)
        with open(log_path, 'a') as f:
            f.write(f'epoch: {epoch}, Q: {Q.T @ Q}\n, F: {F.T @ F}\n')

        loss = 0.0
        for batch in range(batch_size):
            with open(log_path, 'a') as f:
                f.write(f'batch: {batch}\n')
            x_init = cartpole_initx(1)[0]
            x_list, u_list, VN_list = mpc_cartpole(cartpole_sys, Q.T @ Q, R, F.T @ F, MPC_T, T, x_init, u_lower, u_upper)
            loss += RDP_criteria_cartpole(VN_list, x_list, u_list, 1, Q.T @ Q, R, F.T @ F, MPC_T, lambda x: torch.relu(x), test=True, log_path=log_path)
        loss /= batch_size
        print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()