import streamlit as st
from DiffEq import solve_ode
import plotly.graph_objs as go
import pandas as pd

# ---------------------- Layout + Meta ----------------------
st.set_page_config(page_title="Neural ODE Solver", layout="wide")
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
st.markdown("<h1 style='text-align: center;'>Neural ODE Solver (PINNs)</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 16px;'>Solve first-order ODEs of the form <code>LHS(x, y, dy_dx) = RHS(x, y, dy_dx)</code> using a neural network.</p>", unsafe_allow_html=True)
st.markdown("---")

# ---------------------- Layout Columns ----------------------
left, right = st.columns([1, 2], gap="large")

# ---------------------- INPUT SECTION (LEFT) ----------------------
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

    run = st.button("Run Solver")

# ---------------------- OUTPUT SECTION (RIGHT) ----------------------
with right:
    if run:
        try:
            fig, loss_list, (x_vals, y_vals) = solve_ode(lhs_expr, rhs_expr, x0, y0, epochs, x_min, x_max)

            # --- Format LaTeX expression ---
            def format_latex(expr: str) -> str:
                return expr.replace("dy_dx", r"\frac{dy}{dx}").replace("**", "^").replace("*", "")

            latex_clean = f"{format_latex(lhs_expr)} = {format_latex(rhs_expr)}"
            st.markdown("#### Parsed Differential Equation")
            st.latex(latex_clean)

            # --- Solution Plot ---
            st.markdown("#### Solution Plot")
            st.plotly_chart(fig, use_container_width=True)

            # --- Loss Plot ---
            st.markdown("#### Training Loss")
            loss_fig = go.Figure()
            loss_fig.add_trace(go.Scatter(
                x=list(range(len(loss_list))),
                y=loss_list,
                mode='lines',
                name='Training Loss',
                line=dict(color='darkblue')
            ))
            loss_fig.update_layout(
                xaxis_title='Epoch',
                yaxis_title='Loss',
                template='plotly_white',
                margin=dict(l=40, r=10, t=30, b=30)
            )
            st.plotly_chart(loss_fig, use_container_width=True)

            # --- CSV Download ---
            st.markdown("#### Export Solution")
            df_sol = pd.DataFrame({"x": x_vals, "y": y_vals})
            csv = df_sol.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", data=csv, file_name="solution.csv")

        except Exception as e:
            st.error(f"⚠️ An error occurred: {e}")
