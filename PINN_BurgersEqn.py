import torch
import torch.nn as nn
from torch.autograd import grad
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# --- 1. Generate Training Data ---

# Number of points for each condition
num_bc_points = [25, 25]
num_ic_points = 50
num_internal_points = 10000

# Boundary Condition (BC) points: u(-1,t) = 0 and u(1,t) = 0
x0_bc1 = -1 * np.ones(num_bc_points[0])
x0_bc2 = 1 * np.ones(num_bc_points[1])
t0_bc1 = np.linspace(0, 1, num_bc_points[0])
t0_bc2 = np.linspace(0, 1, num_bc_points[1])
u0_bc1 = np.zeros(num_bc_points[0])
u0_bc2 = np.zeros(num_bc_points[1])

# Initial Condition (IC) points: u(x,0) = -sin(pi*x)
x0_ic = np.linspace(-1, 1, num_ic_points)
t0_ic = np.zeros(num_ic_points)
u0_ic = -np.sin(np.pi * x0_ic)

# Group all initial and boundary condition data
X0 = np.concatenate((x0_ic, x0_bc1, x0_bc2)).reshape(-1, 1)
T0 = np.concatenate((t0_ic, t0_bc1, t0_bc2)).reshape(-1, 1)
U0 = np.concatenate((u0_ic, u0_bc1, u0_bc2)).reshape(-1, 1)

# Internal Collocation Points: uniformly sampled from (0,1) x (-1,1)
points_internal = np.random.rand(num_internal_points, 2)
data_x = (2 * points_internal[:, 0] - 1).reshape(-1, 1)
data_t = points_internal[:, 1].reshape(-1, 1)

# Convert all data to PyTorch tensors and set them to double precision
X = torch.tensor(data_x, dtype=torch.float64, requires_grad=True)
T = torch.tensor(data_t, dtype=torch.float64, requires_grad=True)
X0_tensor = torch.tensor(X0, dtype=torch.float64, requires_grad=True)
T0_tensor = torch.tensor(T0, dtype=torch.float64, requires_grad=True)
U0_tensor = torch.tensor(U0, dtype=torch.float64)

# --- 2. Define Neural Network Architecture ---

class PINN(nn.Module):
    def __init__(self, input_size=2, fc_output_size=20, num_blocks=8):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, fc_output_size),
            nn.Tanh(),
            *[nn.Sequential(
                nn.Linear(fc_output_size, fc_output_size),
                nn.Tanh()
            ) for _ in range(num_blocks - 1)],
            nn.Linear(fc_output_size, 1)
        )
        self.double() # Convert to double precision, as in MATLAB

    def forward(self, x, t):
        # Concatenate x and t to create the input to the network
        input_tensor = torch.cat([x, t], dim=1)
        return self.net(input_tensor)

# Instantiate the model
model = PINN()

# --- 3. Define Model Loss Function ---
# The loss function combines the MSE from the PDE residual (f_loss) and
# the MSE from the initial and boundary conditions (u_loss).
def model_loss(model, X, T, X0_tensor, T0_tensor, U0_tensor):
    # Predict u for the internal collocation points
    u = model(X, T)

    # Compute derivatives with autograd. This is the core of the PINN.
    # Create_graph=True for the second-order derivative.
    u_t = grad(u, T, torch.ones_like(u), create_graph=True)[0]
    u_x = grad(u, X, torch.ones_like(u), create_graph=True)[0]
    u_xx = grad(u_x, X, torch.ones_like(u_x), create_graph=True)[0]
    
    # Burger's equation residual (f): u_t + u*u_x - (0.01/pi)*u_xx = 0
    nu = 0.01 / np.pi
    f = u_t + u * u_x - nu * u_xx
    
    # MSE for the PDE residual
    mse_f = torch.mean(f**2)
    
    # Predict u for initial and boundary condition points
    u0_pred = model(X0_tensor, T0_tensor)
    
    # MSE for the initial and boundary conditions
    mse_u = nn.functional.mse_loss(u0_pred, U0_tensor)
    
    # Total loss
    loss = mse_f + mse_u
    return loss

# --- 4. Training with Adam and L-BFGS Optimizers ---
# Stage 1: Warm-up with Adam
print("Starting Adam warm-up...")
adam_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for i in range(2000): # 2000 iterations for warm-up
    adam_optimizer.zero_grad()
    loss = model_loss(model, X, T, X0_tensor, T0_tensor, U0_tensor)
    loss.backward()
    adam_optimizer.step()
    if (i + 1) % 500 == 0:
        print(f'Adam Warm-up Iteration: {i+1}, Loss: {loss.item():.6f}')
print("Adam warm-up finished.")

# Stage 2: Fine-tuning with L-BFGS
print("Starting L-BFGS fine-tuning...")
lbfgs_optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=1500)

def closure():
    lbfgs_optimizer.zero_grad()
    loss = model_loss(model, X, T, X0_tensor, T0_tensor, U0_tensor)
    loss.backward()
    return loss

for i in range(1500):
    loss = lbfgs_optimizer.step(closure)
    if (i + 1) % 100 == 0:
        print(f'L-BFGS Iteration: {i+1}, Loss: {loss.item():.6f}')
print("L-BFGS fine-tuning finished.")

# --- 5. Evaluate Model Accuracy ---
# Compare the PINN's predictions with the true analytical solution.

# Analytical solution function
def solve_burgers(x, t, nu):
    def f_func(y):
        return np.exp(-np.cos(np.pi * y) / (2 * np.pi * nu))
    def g_func(y):
        return np.exp(-y**2 / (4 * nu * t))
    
    u_sol = np.zeros_like(x)
    for i in range(len(x)):
        xi = x[i]
        if np.abs(xi) != 1:
            integrand_num = lambda eta: np.sin(np.pi * (xi - eta)) * f_func(xi - eta) * g_func(eta)
            integrand_den = lambda eta: f_func(xi - eta) * g_func(eta)
            
            integral_num, _ = quad(integrand_num, -np.inf, np.inf)
            integral_den, _ = quad(integrand_den, -np.inf, np.inf)
            
            u_sol[i] = -integral_num / integral_den
            
    return u_sol

# Test the model at specific time points
t_test = [0.25, 0.5, 0.75, 1.0]
num_observations_test = len(t_test)
sz_x_test = 1001
x_test = np.linspace(-1, 1, sz_x_test)
x_test_tensor = torch.tensor(x_test, dtype=torch.float64).reshape(-1, 1)

# Store predictions and true solutions
u_pred_list = []
u_test_list = []

print("Evaluating model...")
# Switch to evaluation mode
model.eval()
with torch.no_grad():
    for t in t_test:
        t_test_tensor = torch.full((sz_x_test, 1), t, dtype=torch.float64)
        xt_test_tensor = torch.cat([x_test_tensor, t_test_tensor], dim=1)
        
        u_pred = model(x_test_tensor, t_test_tensor)
        u_true = solve_burgers(x_test, t, 0.01 / np.pi)
        
        u_pred_list.append(u_pred.squeeze().numpy())
        u_test_list.append(u_true)

# Calculate relative error
u_pred_array = np.array(u_pred_list)
u_test_array = np.array(u_test_list)
err = np.linalg.norm(u_pred_array - u_test_array) / np.linalg.norm(u_test_array)
print(f"Relative L2 Error: {err:.6f}")

# --- 6. Visualize Results ---
plt.figure(figsize=(12, 8))
plt.style.use('seaborn-v0_8-whitegrid')
for i, t in enumerate(t_test):
    plt.subplot(2, 2, i + 1)
    plt.plot(x_test, u_pred_list[i], 'b-', linewidth=2, label='Prediction')
    plt.plot(x_test, u_test_list[i], 'r--', linewidth=2, label='Target')
    plt.ylim([-1.1, 1.1])
    plt.xlabel("x")
    plt.ylabel(f"u(x,{t})")
    plt.title(f"t = {t}")
    plt.legend()
plt.tight_layout()
plt.show()
