import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Clear matplotlib to prevent stale plots
plt.close('all')

# -----------------------------
# Core model
# -----------------------------
def run(v=0.0, c=1.0, N=10000, I0=10, d=160, b=0.3, g=0.05):
    """
    SIR-style simulation with:
    v = vaccination fraction
    c = contact/closure factor (1.0 = no closure, lower = stronger closure)
    """
    # Input validation
    if N <= 0 or I0 <= 0 or d <= 0:
        raise ValueError("N, I0, and d must be positive")
    if I0 > N:
        I0 = N
    if v < 0 or v > 1:
        v = max(0, min(1, v))
    if c <= 0 or c > 1:
        c = max(0.01, min(1, c))
    if b < 0 or b > 1:
        b = max(0, min(1, b))
    if g < 0 or g > 1:
        g = max(0, min(1, g))
    
    S = N - I0 - v * N
    I = float(I0)
    R = v * N

    S_hist = [S]
    I_hist = [I]
    R_hist = [R]

    for _ in range(d - 1):
        inf = (b * c) * S * I / N
        rec = g * I

        S = max(S - inf, 0)
        I = max(I + inf - rec, 0)
        R = R + rec

        S_hist.append(S)
        I_hist.append(I)
        R_hist.append(R)

    return np.array(S_hist), np.array(I_hist), np.array(R_hist)


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Epidemic Simulation", layout="wide")
st.title("Epidemic Simulation Dashboard")

st.write(
    "Use the controls to explore how vaccination and closures affect the spread. "
    "Suggested presets are included for quick testing."
)

# Presets
preset = st.sidebar.selectbox(
    "Suggested preset",
    [
        "Custom",
        "No intervention",
        "Moderate intervention",
        "Strong intervention",
        "High vaccination + closures",
    ],
)

# Default values per preset
preset_values = {
    "No intervention": dict(N=10000, I0=10, d=160, b=0.30, g=0.05, v=0.0, c=1.0),
    "Moderate intervention": dict(N=10000, I0=10, d=160, b=0.30, g=0.05, v=0.4, c=0.6),
    "Strong intervention": dict(N=10000, I0=10, d=160, b=0.30, g=0.05, v=0.6, c=0.4),
    "High vaccination + closures": dict(N=10000, I0=10, d=160, b=0.30, g=0.05, v=0.8, c=0.2),
}

defaults = preset_values.get(
    preset,
    dict(N=10000, I0=10, d=160, b=0.30, g=0.05, v=0.0, c=1.0),
)

st.sidebar.header("Model Inputs")

N = st.sidebar.number_input("Population (N)", min_value=100, value=int(defaults["N"]), step=100)
I0 = st.sidebar.number_input("Initial infected (I0)", min_value=1, value=int(defaults["I0"]), step=1)
d = st.sidebar.number_input("Days (d)", min_value=10, max_value=365, value=int(defaults["d"]), step=1)

b = st.sidebar.slider("Transmission rate (b)", min_value=0.01, max_value=1.00, value=float(defaults["b"]), step=0.01)
g = st.sidebar.slider("Recovery rate (g)", min_value=0.01, max_value=1.00, value=float(defaults["g"]), step=0.01)

v = st.sidebar.slider("Vaccination fraction (v)", min_value=0.0, max_value=0.95, value=float(defaults["v"]), step=0.01)
c = st.sidebar.slider("Closure/contact factor (c)", min_value=0.05, max_value=1.0, value=float(defaults["c"]), step=0.01)

st.sidebar.caption("Hints: higher v lowers susceptible population; lower c reduces transmission.")

# -----------------------------
# Run selected scenario
# -----------------------------
t = np.arange(d)
fmt = FuncFormatter(lambda x, _: f"{int(x):,}")

try:
    S, I, R = run(v=v, c=c, N=N, I0=I0, d=d, b=b, g=g)
except Exception as e:
    st.error(f"Error running simulation: {e}")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Selected scenario")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, S, label="S", linewidth=2)
    ax.plot(t, I, label="I", linewidth=2)
    ax.plot(t, R, label="R", linewidth=2)

    peak_day = int(np.argmax(I))
    peak_val = int(I.max())
    ax.annotate(
        f"Peak: {peak_val:,}\nDay {peak_day}",
        (peak_day, peak_val),
        (min(peak_day + 10, d - 1), peak_val * 0.8),
        arrowprops=dict(arrowstyle="->"),
    )

    ax.set_title("SIR Curves")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.yaxis.set_major_formatter(fmt)
    st.pyplot(fig)
    plt.close(fig)

with col2:
    st.subheader("Summary")
    st.metric("Peak infected", f"{int(I.max()):,}")
    st.metric("Day of peak", f"{peak_day}")
    st.metric("Total infected by end", f"{int(N - S[-1]):,}")
    st.metric("Final recovered", f"{int(R[-1]):,}")

# -----------------------------
# Comparison scenarios
# -----------------------------
st.subheader("Scenario comparison")

scenarios = {
    "No intervention": run(v=0.0, c=1.0, N=N, I0=I0, d=d, b=b, g=g),
    "With interventions": run(v=v, c=c, N=N, I0=I0, d=d, b=b, g=g),
}

fig1, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)

for ax, (title, (Sx, Ix, Rx)) in zip(axes, scenarios.items()):
    ax.plot(t, Sx, label="S", linewidth=2)
    ax.plot(t, Ix, label="I", linewidth=2)
    ax.plot(t, Rx, label="R", linewidth=2)

    pk = int(np.argmax(Ix))
    pv = int(Ix.max())
    ax.annotate(
        f"Peak: {pv:,}\nDay {pk}",
        (pk, pv),
        (min(pk + 8, d - 1), pv * 0.8),
        arrowprops=dict(arrowstyle="->"),
    )

    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.yaxis.set_major_formatter(fmt)

st.pyplot(fig1)
plt.close(fig1)

# -----------------------------
# Sensitivity analysis
# -----------------------------
st.subheader("Sensitivity analysis")

tab1, tab2 = st.tabs(["Vaccination sensitivity", "Closure sensitivity"])

with tab1:
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    v_list = [0, 0.2, 0.4, 0.6, 0.8]

    for vv in v_list:
        _, Ii, _ = run(v=vv, c=c, N=N, I0=I0, d=d, b=b, g=g)
        ax2.plot(t, Ii, label=f"Vax {int(vv * 100)}%")

    ax2.set_title("Effect of Vaccination")
    ax2.grid(alpha=0.3)
    ax2.legend()
    ax2.yaxis.set_major_formatter(fmt)
    st.pyplot(fig2)
    plt.close(fig2)

with tab2:
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    c_list = [1.0, 0.8, 0.6, 0.4, 0.2]

    for cc in c_list:
        _, Ii, _ = run(v=v, c=cc, N=N, I0=I0, d=d, b=b, g=g)
        ax3.plot(t, Ii, label=f"Closure {int((1 - cc) * 100)}%")

    ax3.set_title("Effect of Closures")
    ax3.grid(alpha=0.3)
    ax3.legend()
    ax3.yaxis.set_major_formatter(fmt)
    st.pyplot(fig3)
    plt.close(fig3)

# -----------------------------
# Table of values
# -----------------------------
with st.expander("Show first 20 days of the selected scenario"):
    df = pd.DataFrame(
        {
            "Day": t,
            "Susceptible": S.astype(int),
            "Infected": I.astype(int),
            "Recovered": R.astype(int),
        }
    )
    st.dataframe(df.head(20), use_container_width=True)
