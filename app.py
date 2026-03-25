import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
import io

st.set_page_config(page_title="Optimización de Producción – LP Solver", page_icon="🏭", layout="wide")

st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #1a3a5c 0%, #2d6a9f 100%);
    padding: 1.4rem 2rem; border-radius: 10px; margin-bottom: 1.5rem; color: white;
}
.main-header h1 { margin: 0; font-size: 1.7rem; }
.main-header p  { margin: 0.3rem 0 0; opacity: 0.85; font-size: 0.9rem; }
.kpi-box {
    background: #eef4fb; border-left: 4px solid #2d6a9f;
    padding: 0.9rem 1.1rem; border-radius: 6px; margin-bottom: 0.7rem;
}
.kpi-box .label { font-size: 0.8rem; color: #555; text-transform: uppercase; letter-spacing: 0.05em; }
.kpi-box .val   { font-size: 1.9rem; font-weight: 700; color: #1a3a5c; }
.sec { color: #1a3a5c; border-bottom: 2px solid #2d6a9f; padding-bottom: 0.3rem;
       margin-top: 1.2rem; margin-bottom: 0.8rem; }
[data-testid="stSidebar"] { background-color: #f5f8fc; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🏭 Optimización de Producción — Programación Lineal</h1>
    <p>Problema de planificación · AMPL MO-Book Cap. 1 · Curso II-1122 UCR Alajuela</p>
</div>
""", unsafe_allow_html=True)

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Parámetros del Modelo")

    with st.expander("💰 Precios y Costos", expanded=True):
        price_U = st.number_input("Precio venta U  ($/unid.)", value=270, min_value=0, step=10)
        price_V = st.number_input("Precio venta V  ($/unid.)", value=210, min_value=0, step=10)
        cost_M  = st.number_input("Costo insumo M  ($/unid.)", value=10,  min_value=0, step=5)
        cost_A  = st.number_input("Costo mano obra A ($/hr)", value=50,  min_value=0, step=5)
        cost_B  = st.number_input("Costo mano obra B ($/hr)", value=40,  min_value=0, step=5)

    with st.expander("📦 Capacidades", expanded=True):
        cap_U = st.number_input("Prod. máx. U (unid.)", value=40,  min_value=1, step=5)
        cap_A = st.number_input("Cap. máx. A  (hr)",   value=80,  min_value=1, step=10)
        cap_B = st.number_input("Cap. máx. B  (hr)",   value=100, min_value=1, step=10)

    with st.expander("⚗️ Coeficientes Tecnológicos", expanded=False):
        st.markdown("**Insumo M requerido:**")
        m_U = st.number_input("M por U", value=10, min_value=0, step=1)
        m_V = st.number_input("M por V", value=9,  min_value=0, step=1)
        st.markdown("**Mano de obra A:**")
        a_U = st.number_input("A por U (hr)", value=2, min_value=0, step=1)
        a_V = st.number_input("A por V (hr)", value=1, min_value=0, step=1)
        st.markdown("**Mano de obra B:**")
        b_U = st.number_input("B por U (hr)", value=1, min_value=0, step=1)
        b_V = st.number_input("B por V (hr)", value=1, min_value=0, step=1)

    st.markdown("---")
    solve_btn = st.button("🚀 Resolver Optimización", type="primary", use_container_width=True)

# ─── SOLVER ───────────────────────────────────────────────────────────────────
# Variables: [yU, yV, xM, xA, xB]
def solve_model(pu, pv, cm, ca, cb, cu, cap_a, cap_b, mu, mv, au, av, bu, bv):
    c     = [-pu, -pv, cm, ca, cb]
    A_ub  = [[mu, mv, -1, 0, 0], [au, av, 0, -1, 0], [bu, bv, 0, 0, -1]]
    b_ub  = [0, 0, 0]
    bounds = [(0, cu), (0, None), (0, None), (0, cap_a), (0, cap_b)]
    return linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

def get_shadows(pu, pv, cm, ca, cb, cu, cap_a, cap_b, mu, mv, au, av, bu, bv):
    base  = solve_model(pu, pv, cm, ca, cb, cu, cap_a, cap_b, mu, mv, au, av, bu, bv)
    if base.status != 0:
        return {}
    eps   = 1.0
    out   = {}
    for name, new_cu, new_a, new_b in [
        ("cap_U", cu + eps, cap_a, cap_b),
        ("cap_A", cu, cap_a + eps, cap_b),
        ("cap_B", cu, cap_a, cap_b + eps),
    ]:
        r = solve_model(pu, pv, cm, ca, cb, new_cu, new_a, new_b, mu, mv, au, av, bu, bv)
        out[name] = round((-r.fun - (-base.fun)) / eps, 4) if r.status == 0 else 0
    return out

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["📐 Modelo Matemático", "✅ Solución Óptima", "📊 Análisis de Sensibilidad", "📥 Exportar"])

# TAB 1 — MODEL
with tab1:
    st.markdown('<h3 class="sec">Formulación del Problema</h3>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Variables de Decisión**")
        st.markdown(f"""
| Variable | Descripción | Límites |
|----------|-------------|---------|
| $y_U$ | Unidades producidas de U | $0 \\leq y_U \\leq {cap_U}$ |
| $y_V$ | Unidades producidas de V | $y_V \\geq 0$ |
| $x_M$ | Insumo M comprado | $x_M \\geq 0$ |
| $x_A$ | Mano de obra A | $0 \\leq x_A \\leq {cap_A}$ |
| $x_B$ | Mano de obra B | $0 \\leq x_B \\leq {cap_B}$ |
""")
    with col2:
        st.markdown("**Función Objetivo (Maximizar)**")
        st.latex(rf"\max \quad {price_U}y_U + {price_V}y_V - {cost_M}x_M - {cost_A}x_A - {cost_B}x_B")
        st.markdown("**Restricciones**")
        st.latex(rf"""
\begin{{aligned}}
{m_U}y_U + {m_V}y_V &\leq x_M \\
{a_U}y_U + {a_V}y_V &\leq x_A \\
{b_U}y_U + {b_V}y_V &\leq x_B
\end{{aligned}}
""")

    st.markdown('<h3 class="sec">Código AMPL Equivalente</h3>', unsafe_allow_html=True)
    st.code(f"""var xM >= 0;
var xA >= 0, <= {cap_A};
var xB >= 0, <= {cap_B};
var yU >= 0, <= {cap_U};
var yV >= 0;

maximize Profit:
    {price_U}*yU + {price_V}*yV - {cost_M}*xM - {cost_A}*xA - {cost_B}*xB;

subj to raw_materials: {m_U}*yU + {m_V}*yV <= xM;
subj to labor_A:       {a_U}*yU + {a_V}*yV <= xA;
subj to labor_B:       {b_U}*yU + {b_V}*yV <= xB;""", language="text")

# ─── SOLVE ────────────────────────────────────────────────────────────────────
if solve_btn:
    result  = solve_model(price_U, price_V, cost_M, cost_A, cost_B,
                          cap_U, cap_A, cap_B, m_U, m_V, a_U, a_V, b_U, b_V)
    shadows = get_shadows(price_U, price_V, cost_M, cost_A, cost_B,
                          cap_U, cap_A, cap_B, m_U, m_V, a_U, a_V, b_U, b_V)
    st.session_state.update({
        "result": result, "shadows": shadows,
        "params": dict(price_U=price_U, price_V=price_V, cost_M=cost_M,
                       cost_A=cost_A, cost_B=cost_B, cap_U=cap_U, cap_A=cap_A, cap_B=cap_B,
                       m_U=m_U, m_V=m_V, a_U=a_U, a_V=a_V, b_U=b_U, b_V=b_V)
    })

result  = st.session_state.get("result")
shadows = st.session_state.get("shadows", {})
params  = st.session_state.get("params", {})

# TAB 2 — SOLUTION
with tab2:
    if result is None:
        st.info("👈 Ajusta parámetros y presiona **Resolver Optimización**.")
    elif result.status != 0:
        st.error(f"❌ Sin solución óptima. {result.message}")
    else:
        yU, yV, xM, xA, xB = result.x
        profit  = -result.fun
        revenue = price_U * yU + price_V * yV
        cost    = cost_M * xM + cost_A * xA + cost_B * xB

        st.success("✅ Solución óptima encontrada")

        st.markdown('<h3 class="sec">Resultados Financieros</h3>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        for col, lbl, val in [(c1, "Ganancia Óptima", profit),
                               (c2, "Ingreso Total", revenue),
                               (c3, "Costo Total", cost)]:
            with col:
                st.markdown(f'<div class="kpi-box"><div class="label">{lbl}</div>'
                            f'<div class="val">${val:,.0f}</div></div>', unsafe_allow_html=True)

        st.markdown('<h3 class="sec">Variables Óptimas</h3>', unsafe_allow_html=True)
        ca, cb = st.columns(2)
        with ca:
            st.markdown("**Producción**")
            st.dataframe(pd.DataFrame({
                "Producto": ["U", "V"],
                "Unidades": [round(yU, 2), round(yV, 2)],
                "Ingreso ($)": [round(price_U * yU, 2), round(price_V * yV, 2)],
            }), use_container_width=True, hide_index=True)
        with cb:
            st.markdown("**Insumos**")
            st.dataframe(pd.DataFrame({
                "Insumo": ["M", "A", "B"],
                "Cantidad": [round(xM, 2), round(xA, 2), round(xB, 2)],
                "Costo ($)": [round(cost_M*xM, 2), round(cost_A*xA, 2), round(cost_B*xB, 2)],
            }), use_container_width=True, hide_index=True)

        st.markdown('<h3 class="sec">Gráficas</h3>', unsafe_allow_html=True)
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        fig.patch.set_facecolor("#f5f8fc")
        BLUE = "#2d6a9f"; DARK = "#1a3a5c"; AMBER = "#f0a500"; RED = "#e05252"

        ax = axes[0]
        ax.set_facecolor("#f5f8fc")
        bars = ax.barh(["U", "V"], [yU, yV], color=[BLUE, AMBER], height=0.5)
        ax.set_title("Producción (unidades)", fontweight="bold", color=DARK)
        ax.invert_yaxis()
        for b in bars:
            ax.text(b.get_width() + 0.3, b.get_y() + b.get_height()/2,
                    f"{b.get_width():.0f}", va="center")

        ax = axes[1]
        ax.set_facecolor("#f5f8fc")
        bars = ax.barh(["M", "A", "B"], [xM, xA, xB], color=[DARK, BLUE, "#4a9fd4"], height=0.5)
        ax.set_title("Insumos comprados", fontweight="bold", color=DARK)
        ax.invert_yaxis()
        for b in bars:
            ax.text(b.get_width() + 1, b.get_y() + b.get_height()/2,
                    f"{b.get_width():.0f}", va="center")

        ax = axes[2]
        ax.pie([revenue, cost], labels=["Ingreso", "Costo"], colors=[BLUE, RED],
               autopct="%1.1f%%", startangle=90,
               textprops={"color": "white", "fontweight": "bold"})
        ax.set_title(f"Ingreso vs Costo\nGanancia: ${profit:,.0f}", fontweight="bold", color=DARK)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# TAB 3 — SENSITIVITY
with tab3:
    if result is None:
        st.info("👈 Resuelve el modelo primero.")
    elif result.status != 0:
        st.error("Sin solución.")
    else:
        profit_base = -result.fun
        st.markdown('<h3 class="sec">Precios Sombra (Shadow Prices)</h3>', unsafe_allow_html=True)
        st.markdown("""
> **¿Qué es un precio sombra?**  
> Incremento en la ganancia óptima si se agrega **1 unidad** de capacidad.  
> Precio sombra = 0 → la restricción **no está activa** (hay holgura disponible).
""")

        nm = {"cap_U": "Cap. producción U", "cap_A": "Cap. mano obra A", "cap_B": "Cap. mano obra B"}
        cv = {"cap_U": cap_U, "cap_A": cap_A, "cap_B": cap_B}
        df_sh = pd.DataFrame([{
            "Restricción": nm[k], "Capacidad actual": cv[k],
            "Precio sombra ($/unidad)": v,
            "Estado": "🟢 Activa" if v > 0 else "⚪ Con holgura"
        } for k, v in shadows.items()])
        st.dataframe(df_sh, use_container_width=True, hide_index=True)

        fig2, ax2 = plt.subplots(figsize=(7, 3))
        fig2.patch.set_facecolor("#f5f8fc")
        ax2.set_facecolor("#f5f8fc")
        lbls = [nm[k] for k in shadows]; vals = list(shadows.values())
        colors = ["#2d6a9f" if v > 0 else "#aac4df" for v in vals]
        bars2 = ax2.bar(lbls, vals, color=colors)
        for b, v in zip(bars2, vals):
            ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 0.3,
                     f"${v:.2f}", ha="center", fontweight="bold", color="#1a3a5c")
        ax2.set_title("Precio Sombra por Restricción", fontweight="bold", color="#1a3a5c")
        ax2.set_ylabel("$/unidad adicional")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

        st.markdown('<h3 class="sec">Análisis What-If — Capacidad A</h3>', unsafe_allow_html=True)
        rng = st.slider("Rango de capacidad A a explorar (hr)", 10, 250, (40, 160), step=10)
        cap_A_rng = list(range(rng[0], rng[1] + 1, 5))
        profits_rng = []
        for ca_val in cap_A_rng:
            r = solve_model(price_U, price_V, cost_M, cost_A, cost_B,
                            cap_U, ca_val, cap_B, m_U, m_V, a_U, a_V, b_U, b_V)
            profits_rng.append(-r.fun if r.status == 0 else np.nan)

        fig3, ax3 = plt.subplots(figsize=(8, 3.5))
        fig3.patch.set_facecolor("#f5f8fc")
        ax3.set_facecolor("#f5f8fc")
        ax3.plot(cap_A_rng, profits_rng, color="#2d6a9f", linewidth=2.5, marker="o", markersize=3)
        ax3.axvline(cap_A, color="#e05252", linestyle="--", linewidth=1.5, label=f"Cap. actual = {cap_A}")
        ax3.axhline(profit_base, color="#f0a500", linestyle="--", linewidth=1.5,
                    label=f"Ganancia actual = ${profit_base:,.0f}")
        ax3.set_xlabel("Capacidad A (hr)")
        ax3.set_ylabel("Ganancia Óptima ($)")
        ax3.set_title("Ganancia óptima vs. Capacidad A", fontweight="bold", color="#1a3a5c")
        ax3.legend()
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

# TAB 4 — EXPORT
with tab4:
    if result is None:
        st.info("👈 Resuelve el modelo primero.")
    elif result.status != 0:
        st.error("Sin solución para exportar.")
    else:
        yU, yV, xM, xA, xB = result.x
        profit  = -result.fun
        revenue = price_U * yU + price_V * yV
        cost    = cost_M * xM + cost_A * xA + cost_B * xB

        st.markdown('<h3 class="sec">Exportar a Excel</h3>', unsafe_allow_html=True)

        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            pd.DataFrame({
                "Indicador": ["Ganancia Óptima ($)", "Ingreso Total ($)", "Costo Total ($)"],
                "Valor": [round(profit, 2), round(revenue, 2), round(cost, 2)],
            }).to_excel(writer, sheet_name="Resumen", index=False)

            pd.DataFrame({
                "Variable": ["yU", "yV", "xM", "xA", "xB"],
                "Descripción": ["Prod. U", "Prod. V", "Insumo M", "M. obra A", "M. obra B"],
                "Valor Óptimo": [round(v, 4) for v in [yU, yV, xM, xA, xB]],
            }).to_excel(writer, sheet_name="Variables", index=False)

            if params:
                pd.DataFrame(list(params.items()), columns=["Parámetro", "Valor"]
                             ).to_excel(writer, sheet_name="Parámetros", index=False)

            if shadows:
                nm = {"cap_U": "Cap. U", "cap_A": "Cap. A", "cap_B": "Cap. B"}
                pd.DataFrame({
                    "Restricción": [nm[k] for k in shadows],
                    "Precio Sombra ($/unidad)": list(shadows.values()),
                }).to_excel(writer, sheet_name="Precios Sombra", index=False)

        buf.seek(0)
        st.download_button(
            "📥 Descargar Excel",
            data=buf,
            file_name="solucion_lp_produccion.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        st.markdown("**Vista previa — Resumen:**")
        st.dataframe(pd.DataFrame({
            "Indicador": ["Ganancia Óptima ($)", "Ingreso Total ($)", "Costo Total ($)"],
            "Valor": [round(profit, 2), round(revenue, 2), round(cost, 2)],
        }), use_container_width=True, hide_index=True)
