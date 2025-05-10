import control as ct
import control.optimal as opt
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial # Useful for passing fixed arguments
from tqdm import tqdm
print("GOGOGO!!!")

ref = np.array([0., 0., 1., 0., 0.])
# my_inf = np.inf
# state_lower = [-my_inf, -my_inf, np.cos(15/180 * np.pi), np.sin(-15/180 * np.pi), -my_inf]
# state_upper = [my_inf, my_inf, np.cos(0), np.sin(15/180 * np.pi), my_inf]

"""
OCP utils
"""
def cartpole_update(t, states, inputs, params):
    """
    We will pass (x - ref) as states. 
    But transition should be done on the original state.
    """
    states = states + ref
    gravity, masscart, masspole, length = params
    total_mass = masspole + masscart
    polemass_length = masspole * length

    u = inputs[0]
    lower = -100
    upper = 100
    u = np.clip(u, lower, upper)

    x, dx, cos_th, sin_th, dth = states
    th = np.arctan2(sin_th, cos_th)

    cart_in = (u + polemass_length * dth**2 * sin_th) / total_mass
    th_acc = (gravity * sin_th - cos_th * cart_in) / \
        (length * (4./3. - masspole * cos_th**2 / total_mass))
    xacc = cart_in - polemass_length * th_acc * cos_th / total_mass

    dt = 0.05
    x = x + dt * dx
    dx = dx + dt * xacc
    th = th + dt * dth
    dth = dth + dt * th_acc

    next_states = np.array([x, dx, np.cos(th), np.sin(th), dth])

    return next_states - ref

def get_cartpole_sys():
    cartpole_sys = ct.nlsys(updfcn=cartpole_update, outfcn=cartpole_update, inputs=1, outputs=5, states=5, params=[9.8, 1.0, 0.1, 0.5], name='cartpole_sys', dt=1)
    return cartpole_sys

def uniform(shape, low, high):
    r = high - low
    return np.random.rand(*shape) * r + low

def cartpole_initx(n_batch, angle=180):
    ratio = angle / 180
    th = uniform((n_batch, 1), -ratio*np.pi, ratio*np.pi)
    thdot = uniform((n_batch, 1), -.5, .5)
    x = uniform((n_batch, 1), -0.5, 0.5)
    xdot = uniform((n_batch, 1), -0.5, 0.5)
    xinit = np.concatenate((x, xdot, np.cos(th), np.sin(th), thdot), axis=1)
    return xinit - ref

def solve_ocp(x0, cartpole_sys, timepts, Q, q, R, lower, upper):
    """
        Q: n_array, n_state x n_state
        R: n_array, n_ctrl x n_ctrl
        Qf: n_array, n_state x n_state
        lower: n_array, n_ctrl x 1
        upper: n_array, n_ctrl x 1
    """
    constraints = [opt.input_range_constraint(cartpole_sys, lower, upper)]
    running_cost = partial(mpc_torch_running_cost, Q=Q, q=q, R=R)
    # result = opt.solve_ocp(cartpole_sys, timepts, x0, cost=running_cost, trajectory_constraints=constraints, terminal_cost=terminal_cost)
    result = opt.solve_ocp(cartpole_sys, timepts, x0, cost=running_cost)
    return result

def mpc_torch_running_cost(x, u, Q, q, R):
    """
    All numpy arrays.
    x shape: (n_state,)
    u shape: (n_ctrl,)
    Q shape: (n_state, n_state)
    q shape: (n_state,)
    R shape: (n_ctrl, n_ctrl)
    use numpy to compute
    """
    x = np.array(x)
    u = np.array(u)
    Q = np.array(Q)
    q = np.array(q)
    R = np.array(R)
    # x^T Q x + q^T x
    state_cost = np.matmul(np.matmul(x.T, Q), x) + np.matmul(q.T, x)
    # u^T R u
    ctrl_cost = np.matmul(np.matmul(u.T, R), u)
    return state_cost + ctrl_cost

def mpc_torch_terminal_cost(x, u, Q, q):
    return np.matmul(np.matmul(x.T, Q), x) + np.matmul(q.T, x)

"""
MPC utils
"""
def cost_cartpole(x, u, Q, q, R, is_terminal):
    """
    x shape: (n_batch, n_state, 1)
    u shape: (n_batch, n_ctrl, 1)
    q shape: (n_state,)

    return shape: (n_batch, )
    """
    if is_terminal:
        cost = torch.matmul(torch.matmul(x.transpose(1, 2), Q), x) + torch.matmul(x.transpose(1, 2), q.unsqueeze(0).unsqueeze(-1))
    else:
        cost = torch.matmul(torch.matmul(x.transpose(1, 2), Q), x) + torch.matmul(x.transpose(1, 2), q.unsqueeze(0).unsqueeze(-1)) + \
               torch.matmul(torch.matmul(u.transpose(1, 2), R), u)
    return cost.squeeze(-1).squeeze(-1).squeeze(-1)

def VN_cartpole_multi(results_states, results_inputs, Q, q, R):
    """
    return shape: (n_batch, )
    """
    n_batch, MPC_T, n_state, T = results_states.shape

    VN_list = []
    for n in range(MPC_T):
        VN = 0
        for t in range(T):
            if t == T - 1:
                VN += cost_cartpole(results_states[:, n, :, t].unsqueeze(2), results_inputs[:, n, :, t].unsqueeze(2), Q, q, R, True)
            else:
                VN += cost_cartpole(results_states[:, n, :, t].unsqueeze(2), results_inputs[:, n, :, t].unsqueeze(2), Q, q, R, False)
        VN_list.append(VN)

    return VN_list

def RDP_criteria_cartpole(VN_list, x_list, u_list, alpha, Q, q, R, MPC_T, func, test=False, log_path=None):
    lossRDP = 0
    lossLyap = 0
    for i in range(MPC_T - 1):
        RDP = (VN_list[i+1] + alpha * cost_cartpole(x_list[i].unsqueeze(2), u_list[i].unsqueeze(2), Q, q, R, False)) - VN_list[i] # Wish RDP <= 0
        Lyap = cost_cartpole(x_list[i+1].unsqueeze(2), None, Q, q, R, True) + \
               cost_cartpole(x_list[i].unsqueeze(2), u_list[i].unsqueeze(2), Q, q, R, False) - \
               cost_cartpole(x_list[i].unsqueeze(2), None, Q, q, R, True) 

        if test:
            if log_path is not None:
                with open(log_path, 'a') as f:
                    f.write(f'RDP{i}: {RDP}\n')
        lossRDP += func(RDP)
        lossLyap += func(Lyap)
    return lossRDP, lossLyap

def mpc_cartpole_single(x_init, cartpole_sys, Q, q, R, MPC_T, T, u_lower, u_upper):
    x = x_init

    timepts = np.arange(0, T, 1)
    lower = u_lower
    upper = u_upper

    x_list = []
    u_list = []
    for i in range(MPC_T):
        result = solve_ocp(x, cartpole_sys, timepts, Q.data, q.data, R.data, lower, upper)
        print(f"MPC Timestamp{i} success? : ", result.success)
        u = result.inputs[:, 0] # u: list size n_ctrl
        x_list.append(result.states)
        u_list.append(result.inputs)
        x = result.states[:, 1] # x: list size n_state
    return x_list, u_list

def solve_multi_mpc(initial_states, cartpole_sys, Q, q, R, MPC_T, T, u_lower, u_upper, max_workers=4):
    """
    initial_states: [state_1, state_2, ...], executor will automatically dispatch, even when n_batch > max_workers
    """
    solve_mpc_with_params = partial(mpc_cartpole_single, 
                                    cartpole_sys=cartpole_sys,
                                    Q=Q, q=q, R=R, MPC_T=MPC_T, T=T, u_lower=u_lower, u_upper=u_upper)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results_iter = executor.map(solve_mpc_with_params, initial_states)
        results_all = list(results_iter)
    return results_all

"""
Training utils
"""
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def inverse_square_weight_penalty(Q, q, lambda_param=0.01):
    penalty = 0
    penalty += torch.sum(1.0 / (Q.pow(2) + 1e-8))
    penalty += torch.sum(1.0 / (q.pow(2) + 1e-8))
    
    return lambda_param * penalty

if __name__ == '__main__':
    set_seed(42)
    # Experiment params
    epochs = 100
    batch_size = 32
    lr = 0.01
    max_workers = 7
    test_name = 'test_mpc_torch'
    # log_path_root = 'D:/Docs/code_lib/graduation_test/control_lib/log_path'
    log_path_root = './'
    log_path = log_path_root + f'/{test_name}.txt'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # System params
    cartpole_sys = get_cartpole_sys()


    target_weight = [0.1, 0.1, 1., 1., 0.1]
    Q_data = torch.tensor(target_weight).to(device)
    q_data = torch.tensor([0., 0., 1., 0., 0.]).to(device)
    rand_q_bias = torch.randn(5).to(device) * 0.1

    Q_ori = nn.Parameter(Q_data)
    q = nn.Parameter(q_data)
    R = torch.Tensor([[0.001]]).to(device)

    MPC_T = 30
    T = 30
    u_lower = -100
    u_upper = 100

    loss_list = []
    # Train
    load_params = False
    optimizer = torch.optim.Adam([Q_ori, q], lr=lr)
    if load_params:
        optimizer.load_state_dict(torch.load('D:/Docs/code_lib/graduation_test/control_lib/log_path/model_path/Parallel_lr0.001_ref5_Opt_18.pth'))
        print("Load optimizer params success!")
        # change lr 
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    for epoch in tqdm(range(epochs)):
        # Save model params
        torch.save(Q_ori.data, log_path_root + f'/{test_name}_Q_{epoch}.pt')
        torch.save(q.data, log_path_root + f'/{test_name}_q_{epoch}.pt')
        # Save optimizer state
        torch.save(optimizer.state_dict(), log_path_root + f'/{test_name}_Opt_{epoch}.pth')

        Q = torch.diag(Q_ori)
        with open(log_path, 'a') as f:
            f.write(f'epoch: {epoch}, Q: {Q}\n, q: {q}\n')

        loss = 0.0
        # Forward: Sampling use multiprocess MPC
        Q0, q0, R0 = Q.detach().cpu().numpy(), q.detach().cpu().numpy(), R.detach().cpu().numpy()
        initial_states = cartpole_initx(batch_size)
        results = solve_multi_mpc(initial_states, cartpole_sys, Q0, q0, R0, MPC_T, T, u_lower, u_upper, max_workers=max_workers)

        """
        results shape: (n_batch, 2, MPC_T, n_state/n_ctrl, T), list[list[list[array]]]
        Has inhomogeneous part, cant convert to numpy/tensor directly.

        Convert to numpy first, because convert list of numpy to tensor is slow.
        """
        results_states = torch.tensor(np.array([result[0] for result in results]), dtype=torch.float32) # (n_batch, MPC_T, n_state, T)
        results_inputs = torch.tensor(np.array([result[1] for result in results]), dtype=torch.float32) # (n_batch, MPC_T, n_ctrl, T)

        results_states = results_states.to(device)
        results_inputs = results_inputs.to(device)

        # Caculate loss
        """
        x_list: torch tensor, shape(MPC_T, n_batch, n_state), using the first state of each OCP
        u_list: torch tensor, shape(MPC_T, n_batch, n_ctrl), using the first input of each OCP
        VN_list: torch tensor, shape(MPC_T, n_batch)
        """
        x_list = results_states[:, :, :, 0].permute(1, 0, 2) # (MPC_T, n_batch, n_state)
        u_list = results_inputs[:, :, :, 0].permute(1, 0, 2) # (MPC_T, n_batch, n_ctrl)
        VN_list = VN_cartpole_multi(results_states, results_inputs, Q, q, R)
        # VN_list = VN_cartpole_multi(results_states, results_inputs, Q, R, F)
        lossRDP, lossLyap = RDP_criteria_cartpole(VN_list, x_list, u_list, 1, Q, q, R, 
                                                  MPC_T, lambda x: torch.relu(x), test=True, log_path=log_path)
        lossRDP = lossRDP.mean()
        lossLyap = lossLyap.mean()
        # loss += RDP_criteria_cartpole(VN_list, x_list, u_list, 1, Q, R, F, MPC_T, lambda x: torch.relu(x), test=True, log_path=log_path).mean()
        # bound_penalty = bounded_weight_penalty(Q, F, weight_min, weight_max, lambda_weight)
        inverse_weight_penalty = inverse_square_weight_penalty(Q, q)
        loss += lossRDP + lossLyap + inverse_weight_penalty
        with open(log_path, 'a') as f:
            # f.write(f'bound_penalty: {bound_penalty}\n')
            f.write(f'lossRDP: {lossRDP}\n')
            f.write(f'lossLyap: {lossLyap}\n')
            f.write(f'inverse_weight_penalty: {inverse_weight_penalty}\n')

        # Backward: Training use RDP
        optimizer.zero_grad()
        loss.backward()
        loss_list.append(loss.item())
        optimizer.step()

        with open(log_path, 'a') as f:
            f.write(f'loss: {loss}\n')
    
    print(Q)
    print(q)