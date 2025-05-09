from pysmt.shortcuts import Symbol, GT, Plus, Times, Real  
from pysmt.typing import REAL  
import numpy as np  
  
def verify_lyap(n_state, n_ctrl, Q, R, F, dyna):
    # 2. Create symbolic variables for vectors x and u  
    x = [Symbol(f"x_{i}", REAL) for i in range(n_state)]  
    u = [Symbol(f"u_{i}", REAL) for i in range(n_ctrl)]  

    Fx = Real(0)

    lx = Real(0)

    Fx1 = Real(0)

if __name__ == '__main__':
    pass