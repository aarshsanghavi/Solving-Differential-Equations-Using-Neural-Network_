# model.py
import torch
import torch.nn as nn
import numpy as np
import plotly.graph_objs as go  

def make_expr_func(expr_str):
    allowed_names = {
        'torch': torch,
        'sin': torch.sin, 'cos': torch.cos, 'exp': torch.exp, 'log': torch.log,
        'tan': torch.tan, 'abs': torch.abs, 'sqrt': torch.sqrt,
        'pi': torch.pi, 'e': np.e
    }
    code = compile(expr_str, "<string>", "eval")
    def expr_func(x, y, dy_dx):
        env = dict(allowed_names)
        env.update({'x': x, 'y': y, 'dy_dx': dy_dx})
        return eval(code, {"__builtins__": None}, env)
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
        lhs_val = lhs_func(x, y_pred, dy_dx)
        rhs_val = rhs_func(x, y_pred, dy_dx)

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
    fig.add_trace(go.Scatter(
        x=x_plot,
        y=y_sol,
        mode='lines',
        name='NN Solution',
        line=dict(color='royalblue')
    ))

    # Optional analytical solution (if trivial test case)
    if lhs_expr.strip().replace(' ', '') == "dy_dx" and rhs_expr.strip().replace(' ', '') == "x**4" and x0_val == 0 and y0_val == 0:
        y_true = x_plot ** 5 / 5
        fig.add_trace(go.Scatter(
            x=x_plot,
            y=y_true,
            mode='lines',
            name='Analytical: x^5/5',
            line=dict(color='green', dash='dash')
        ))

    fig.update_layout(
        title=f"Solution domain: [{x_min}, {x_max}]",
        xaxis_title="x",
        yaxis_title="y(x)",
        hovermode='x unified',
        template='plotly_white',
        legend=dict(x=0.01, y=0.99)
    )

    return fig, loss_list, (x_plot, y_sol)

