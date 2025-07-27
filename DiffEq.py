# ---------- DiffEq.py ----------

import torch
import torch.nn as nn
import numpy as np
import plotly.graph_objs as go


def make_expr_func(expr_str):
    allowed_names = {
        'sin': torch.sin, 'cos': torch.cos, 'exp': torch.exp, 'log': torch.log,
        'tan': torch.tan, 'abs': torch.abs, 'sqrt': torch.sqrt,
        'pi': torch.pi, 'e': np.e,
        'x': None, 't': None, 'u': None,
        'du_dx': None, 'du_dt': None, 'd2u_dx2': None, 'd2u_dt2': None
    }
    code = compile(expr_str, "<string>", "eval")

    def expr_func(**kwargs):
        env = dict(allowed_names)  # use allowed names
        env.update(kwargs)         # fill in actual torch tensors
        return eval(code, {"__builtins__": {}}, env)

    return expr_func


class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1)
        )

    def forward(self, x):
        return self.net(x)


def solve_ode(lhs_expr, rhs_expr, x0_val, y0_val, epochs, x_min, x_max):
    device = torch.device("cpu")
    torch.manual_seed(42)

    if x_min > x_max:
        x_min, x_max = x_max, x_min

    x = torch.arange(x_min, x_max, 0.05, requires_grad=True, device=device).unsqueeze(1)
    x0 = torch.tensor([[x0_val]], dtype=torch.float32, device=device, requires_grad=True)
    y0 = torch.tensor([[y0_val]], dtype=torch.float32, device=device, requires_grad=True)

    model = NeuralNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    lhs_func = make_expr_func(lhs_expr)
    rhs_func = make_expr_func(rhs_expr)
    loss_list = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(x)
        dy_dx = torch.autograd.grad(y_pred, x, grad_outputs=torch.ones_like(y_pred), create_graph=True)[0]

        lhs_val = lhs_func(x=x, y=y_pred, dy_dx=dy_dx)
        rhs_val = rhs_func(x=x, y=y_pred, dy_dx=dy_dx)

        loss_eq = loss_fn(lhs_val, rhs_val)
        y0_pred = model(x0)
        loss_ic = loss_fn(y0_pred, y0)
        loss = loss_eq + loss_ic
        loss_list.append(loss.item())
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        y_sol = model(x).cpu().numpy().flatten()
        x_plot = x.cpu().numpy().flatten()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_plot, y=y_sol, mode='lines', name='NN Solution', line=dict(color='royalblue')))

    if lhs_expr.strip().replace(' ', '') == "dy_dx" and rhs_expr.strip().replace(' ', '') == "x**4" and x0_val == 0 and y0_val == 0:
        y_true = x_plot ** 5 / 5
        fig.add_trace(go.Scatter(x=x_plot, y=y_true, mode='lines', name='Analytical: x^5/5', line=dict(color='green', dash='dash')))

    fig.update_layout(
        title=f"Solution domain: [{x_min}, {x_max}]",
        xaxis_title="x", yaxis_title="y(x)",
        hovermode='x unified', template='plotly_white',
        legend=dict(x=0.01, y=0.99)
    )

    return fig, loss_list, (x_plot, y_sol)


class NeuralNetPDE(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1)
        )

    def forward(self, xt):
        return self.net(xt)


def solve_pde(lhs_expr, rhs_expr, ic_expr, x_bounds, t_bounds, epochs=1000, aspect_ratio=None):
    device = torch.device("cpu")
    torch.manual_seed(42)

    x_vals = torch.linspace(x_bounds[0], x_bounds[1], 50, device=device)
    t_vals = torch.linspace(t_bounds[0], t_bounds[1], 50, device=device)
    X, T = torch.meshgrid(x_vals, t_vals, indexing='ij')
    xt = torch.stack([X.reshape(-1), T.reshape(-1)], dim=1).requires_grad_()

    model = NeuralNetPDE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    lhs_func = make_expr_func(lhs_expr)
    rhs_func = make_expr_func(rhs_expr)
    ic_func = make_expr_func(ic_expr)

    loss_list = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        u_pred = model(xt)

        grads = torch.autograd.grad(u_pred, xt, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
        du_dx = grads[:, 0:1]
        du_dt = grads[:, 1:2]
        d2u_dx2 = torch.autograd.grad(du_dx, xt, grad_outputs=torch.ones_like(du_dx), create_graph=True)[0][:, 0:1]

        env = {
            'x': xt[:, 0:1],
            't': xt[:, 1:2],
            'u': u_pred,
            'du_dx': du_dx,
            'du_dt': du_dt,
            'd2u_dx2': d2u_dx2
        }

        lhs_val = lhs_func(**env)
        rhs_val = rhs_func(**env)
        loss_residual = loss_fn(lhs_val, rhs_val)

        x_ic = torch.linspace(x_bounds[0], x_bounds[1], 50, device=device).unsqueeze(1)
        t_ic = torch.zeros_like(x_ic)
        xt_ic = torch.cat([x_ic, t_ic], dim=1)
        u_ic = model(xt_ic)
        u_ic_target = ic_func(x=x_ic)
        loss_ic = loss_fn(u_ic, u_ic_target)

        t_bc = torch.linspace(t_bounds[0], t_bounds[1], 50, device=device).unsqueeze(1)
        x0 = torch.full_like(t_bc, x_bounds[0])
        x1 = torch.full_like(t_bc, x_bounds[1])
        xt_bc0 = torch.cat([x0, t_bc], dim=1)
        xt_bc1 = torch.cat([x1, t_bc], dim=1)
       # Neumann BCs: ∂u/∂x = 0 at boundaries
        xt_bc0.requires_grad_(True)
        xt_bc1.requires_grad_(True)

        u_bc0 = model(xt_bc0)
        u_bc1 = model(xt_bc1)

        grad_bc0 = torch.autograd.grad(u_bc0, xt_bc0, grad_outputs=torch.ones_like(u_bc0), create_graph=True)[0]
        grad_bc1 = torch.autograd.grad(u_bc1, xt_bc1, grad_outputs=torch.ones_like(u_bc1), create_graph=True)[0]

        du_dx_bc0 = grad_bc0[:, 0:1]
        du_dx_bc1 = grad_bc1[:, 0:1]

        loss_bc = loss_fn(du_dx_bc0, torch.zeros_like(du_dx_bc0)) + loss_fn(du_dx_bc1, torch.zeros_like(du_dx_bc1))

        loss = loss_residual + loss_ic + loss_bc
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())

    with torch.no_grad():
        u_sol = model(xt).cpu().numpy().reshape(50, 50)
        x_np = x_vals.cpu().numpy()
        t_np = t_vals.cpu().numpy()

    fig = go.Figure(data=[
        go.Surface(z=u_sol.T, x=x_np, y=t_np)
    ])

    fig.update_layout(
        title="Solution Surface: u(x,t)",
        scene=dict(
            xaxis_title='x',
            yaxis_title='t',
            zaxis_title='u(x,t)',
            aspectmode='manual',
            aspectratio=aspect_ratio if aspect_ratio else dict(x=1, y=1, z=0.7)
        ),
        template='plotly_white'
    )

    return fig, loss_list, u_sol, x_np, t_np