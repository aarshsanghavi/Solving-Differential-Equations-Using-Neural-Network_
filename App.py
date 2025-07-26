import streamlit as st
from DiffEq import solve_ode, solve_pde
import plotly.graph_objs as go
import pandas as pd

def format_pde_latex(lhs: str, rhs: str) -> str:
    replacements = {
        "du_dt": r"\frac{\partial u}{\partial t}",
        "d2u_dx2": r"\frac{\partial^2 u}{\partial x^2}",
        "d2u_dt2": r"\frac{\partial^2 u}{\partial t^2}",
        "du_dx": r"\frac{\partial u}{\partial x}",
        "*": r"\cdot"
    }
    for key, val in replacements.items():
        lhs = lhs.replace(key, val)
        rhs = rhs.replace(key, val)
    return f"{lhs.strip()} = {rhs.strip()}"


# ---------------------- Layout + Meta ----------------------
st.set_page_config(page_title="Neural DE Solver (ODE + PDE)", layout="wide")
st.markdown("""
    <style>
        html, body, [class*="css"] {
            font-family: 'Segoe UI', sans-serif;
        }
        h1 {
            color: #003366;
        }
        .stDownloadButton {
            margin-top: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------- Page Header ----------------------
st.markdown("<h1 style='text-align: center;'>Neural DE Solver (PINNs)</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 16px;'>Solve ODEs and PDEs using Physics-Informed Neural Networks.</p>", unsafe_allow_html=True)
st.markdown("---")

# ---------------------- Mode Toggle ----------------------
mode = st.radio("Select Solver Mode", ["ODE Solver", "PDE Solver"])

# ---------------------- ODE Solver ----------------------
if mode == "ODE Solver":
    st.markdown("### 🧮 ODE Setup")
    left, right = st.columns([1, 2], gap="large")

    with left:
        st.subheader("Define the ODE")
        lhs_expr = st.text_input("LHS Expression", value="dy_dx")
        rhs_expr = st.text_input("RHS Expression", value="x**4")

        st.subheader("Initial Conditions")
        x0 = st.number_input("Initial x₀", value=0.0)
        y0 = st.number_input("Initial y₀", value=0.0)

        st.subheader("Domain & Training")
        x_min = st.number_input("x_min", value=-2.0)
        x_max = st.number_input("x_max", value=2.0)
        epochs = st.slider("Training Epochs", 100, 10000, value=1000, step=100)

        run_ode = st.button("Run ODE Solver")

    with right:
        if run_ode:
            try:
                fig, loss_list, (x_vals, y_vals) = solve_ode(lhs_expr, rhs_expr, x0, y0, epochs, x_min, x_max)

                def format_latex(expr: str) -> str:
                    return expr.replace("dy_dx", r"\frac{dy}{dx}").replace("**", "^").replace("*", "")

                st.markdown("#### Parsed ODE")
                st.latex(f"{format_latex(lhs_expr)} = {format_latex(rhs_expr)}")

                st.markdown("#### Solution Plot")
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("#### Training Loss")
                loss_fig = go.Figure()
                loss_fig.add_trace(go.Scatter(x=list(range(len(loss_list))), y=loss_list, mode='lines', name='Loss'))
                loss_fig.update_layout(xaxis_title='Epoch', yaxis_title='Loss', template='plotly_white')
                st.plotly_chart(loss_fig, use_container_width=True)

                st.markdown("#### Export Solution")
                df_sol = pd.DataFrame({"x": x_vals, "y": y_vals})
                csv = df_sol.to_csv(index=False).encode("utf-8")
                st.download_button("Download CSV", data=csv, file_name="ode_solution.csv")

            except Exception as e:
                st.error(f"⚠️ An error occurred: {e}")

# ---------------------- PDE Solver ----------------------
elif mode == "PDE Solver":
    st.markdown("### 🌊 PDE Setup")
    left, right = st.columns([1, 2], gap="large")

    with left:
        st.subheader("Define the PDE")
        lhs_expr = st.text_input("LHS Expression", value="du_dt")
        rhs_expr = st.text_input("RHS Expression", value="0.01 * d2u_dx2")

        st.subheader("Initial Condition")
        ic_expr = st.text_input("Initial Condition (u(x,0))", value="torch.sin(torch.pi * x)")

        st.subheader("Domain & Training")
        x_min = st.number_input("x_min", value=0.0)
        x_max = st.number_input("x_max", value=1.0)
        t_min = st.number_input("t_min", value=0.0)
        t_max = st.number_input("t_max", value=1.0)
        epochs = st.slider("Training Epochs", 100, 10000, value=1000, step=100)

        run_pde = st.button("Run PDE Solver")

    with right:
        if run_pde:
            try:
                fig, loss_list = solve_pde(lhs_expr, rhs_expr, ic_expr, x_bounds=(x_min, x_max), t_bounds=(t_min, t_max), epochs=epochs)

                st.markdown("#### Parsed PDE")
                st.latex(format_pde_latex(lhs_expr, rhs_expr))

                st.markdown("#### 3D Solution Surface")
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("#### Training Loss")
                loss_fig = go.Figure()
                loss_fig.add_trace(go.Scatter(x=list(range(len(loss_list))), y=loss_list, mode='lines', name='Loss'))
                loss_fig.update_layout(xaxis_title='Epoch', yaxis_title='Loss', template='plotly_white')
                st.plotly_chart(loss_fig, use_container_width=True)

            except Exception as e:
                st.error(f"⚠️ An error occurred: {e}")
