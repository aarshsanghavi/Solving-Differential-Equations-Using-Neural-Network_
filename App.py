import streamlit as st
from DiffEq import solve_ode, solve_pde
import plotly.graph_objs as go
import pandas as pd
import numpy as np

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

# ---------------------- Session State Init ----------------------
if 'ode_history' not in st.session_state:
    st.session_state.ode_history = []

if 'pde_history' not in st.session_state:
    st.session_state.pde_history = []

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

                # Save to history
                st.session_state.ode_history.insert(0, (lhs_expr, rhs_expr))
                st.session_state.ode_history = st.session_state.ode_history[:5]

            except Exception as e:
                st.error(f"⚠️ An error occurred: {e}")

    if st.session_state.ode_history:
        st.markdown("### 📜 ODE History (Last 5)")
        for i, (lhs, rhs) in enumerate(st.session_state.ode_history, 1):
            st.markdown(f"**{i}.** \\( {format_latex(lhs)} = {format_latex(rhs)} \\)", unsafe_allow_html=True)

# ---------------------- PDE Solver ----------------------
elif mode == "PDE Solver":
    st.markdown("### 🌊 PDE Setup")
    left, right = st.columns([1, 2], gap="large")

    with left:
        st.subheader("Define the PDE")
        lhs_expr = st.text_input("LHS Expression", value="du_dt")
        rhs_expr = st.text_input("RHS Expression", value="0.01 * d2u_dx2")

        st.subheader("Initial Condition")
        ic_expr = st.text_input("Initial Condition (u(x,0))", value="sin(pi * x)")

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
                fig, loss_list, u_arr, x_vals, t_vals = solve_pde(
                    lhs_expr, rhs_expr, ic_expr,
                    x_bounds=(x_min, x_max),
                    t_bounds=(t_min, t_max),
                    epochs=epochs,
                    aspect_ratio=None
                )

                # Save to session
                st.session_state.pde_solution = {
                    "u_arr": u_arr,
                    "x_vals": x_vals,
                    "t_vals": t_vals,
                    "lhs_expr": lhs_expr,
                    "rhs_expr": rhs_expr,
                    "ic_expr": ic_expr
                }
                st.session_state.pde_loss_list = loss_list

            except Exception as e:
                st.error(f"⚠️ An error occurred: {e}")

        if "pde_solution" in st.session_state:
            st.subheader("Visualization Controls")
            x_scale = st.slider("X-axis Scale", 0.1, 5.0, 1.0, step=0.1)
            t_scale = st.slider("T-axis Scale", 0.1, 5.0, 1.0, step=0.1)
            z_scale = st.slider("U(x,t) Scale", 0.1, 5.0, 0.7, step=0.1)

            aspect = dict(x=x_scale, y=t_scale, z=z_scale)

            fig = go.Figure(data=[
                go.Surface(
                    z=st.session_state.pde_solution["u_arr"].T,
                    x=st.session_state.pde_solution["x_vals"],
                    y=st.session_state.pde_solution["t_vals"]
                )
            ])
            fig.update_layout(
                title="Solution Surface: u(x,t)",
                scene=dict(
                    xaxis_title='x',
                    yaxis_title='t',
                    zaxis_title='u(x,t)',
                    aspectmode='manual',
                    aspectratio=aspect
                ),
                template='plotly_white'
            )

            st.markdown("#### Parsed PDE")
            st.latex(format_pde_latex(
                st.session_state.pde_solution["lhs_expr"],
                st.session_state.pde_solution["rhs_expr"]
            ))

            st.markdown("#### 3D Solution Surface")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("#### 🔁 Animated 2D Heat Distribution (u(x,t))")

            x_vals = st.session_state.pde_solution["x_vals"]
            t_vals = st.session_state.pde_solution["t_vals"]
            u_arr = st.session_state.pde_solution["u_arr"]

            # --- Slider for manual control
            selected_frame = st.slider(
                "Select frame index",  # Show index, not value
                min_value=0,
                max_value=len(t_vals) - 1,
                value=0,
            )

            # --- Display actual t value
            st.markdown(f"**Selected t = {t_vals[selected_frame]:.2f}**")

            # --- Manual Frame Plot (Static)
            fig_static = go.Figure()
            fig_static.add_trace(go.Scatter(
                x=x_vals,
                y=u_arr[:, selected_frame],
                mode='lines',
                name=f't={t_vals[selected_frame]:.2f}'
            ))
            fig_static.update_layout(
                title=f"u(x) at t = {t_vals[selected_frame]:.2f}",
                xaxis_title='x',
                yaxis_title='u(x,t)',
                template='plotly_white'
            )
            st.plotly_chart(fig_static, use_container_width=True)

            # Spacer to separate plot and slider
            st.markdown("<br>", unsafe_allow_html=True)

            # --- Animated Plot (Play once forward and backward)
            frames = [
                go.Frame(
                    data=[go.Scatter(x=x_vals, y=u_arr[:, i], mode='lines')],
                    name=str(i)
                )
                for i in range(len(t_vals))
            ]

            # Add reverse frames for ping-pong
            frames += [
                go.Frame(
                    data=[go.Scatter(x=x_vals, y=u_arr[:, i], mode='lines')],
                    name=str(len(t_vals) + i)
                )
                for i in reversed(range(len(t_vals)))
            ]

            fig_anim = go.Figure(
                data=[go.Scatter(x=x_vals, y=u_arr[:, 0], mode='lines')],
                layout=go.Layout(
                    title="🔁 Animation of Heat Distribution",
                    xaxis=dict(title="x", range=[x_vals[0], x_vals[-1]]),
                    yaxis=dict(title="u(x,t)", range=[float(np.min(u_arr)), float(np.max(u_arr))]),
                    updatemenus=[dict(
                        type="buttons",
                        showactive=False,
                        buttons=[
                            dict(
                                label="Play Once ↔",
                                method="animate",
                                args=[None, {
                                    "frame": {"duration": 100, "redraw": True},
                                    "fromcurrent": True,
                                    "mode": "immediate"
                                }]
                            )
                        ]
                    )]
                ),
                frames=frames
            )

            st.plotly_chart(fig_anim, use_container_width=True)

            st.markdown("#### Training Loss")
            loss_fig = go.Figure()
            loss_fig.add_trace(go.Scatter(
                x=list(range(len(st.session_state.pde_loss_list))),
                y=st.session_state.pde_loss_list,
                mode='lines', name='Loss'
            ))
            loss_fig.update_layout(xaxis_title='Epoch', yaxis_title='Loss', template='plotly_white')
            st.plotly_chart(loss_fig, use_container_width=True)
