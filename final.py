# source venv/bin/activate
# streamlit run final.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from scipy.optimize import minimize
from scipy.stats import norm

# ─────────────────────────────────────────────
# Model Parameters (from notebook non-linear regression)
# ─────────────────────────────────────────────
a1, a2, a3, a4, b1 = 883.58722, 0.35408, 0.755684, 0.826155, 0.963532


# ─────────────────────────────────────────────
# Pre-Trade TCA Class
# ─────────────────────────────────────────────
class PreTradeTCA:
    @staticmethod
    def MarketImpact(Size, POV, Volatility):
        Istar = a1 * (Size ** a2) * (Volatility ** a3)
        return Istar * (b1 * POV ** a4 + (1 - b1))

    @staticmethod
    def TimingRisk(Size, POV, Volatility):
        return Volatility * np.sqrt((1/3) * (1/250) * Size * ((1 - POV) / POV)) * 10000

    @staticmethod
    def PriceAppreciation(Side, AlphaBp, Size, POV):
        return Side * 0.5 * AlphaBp * Size * ((1 - POV) / POV)

    @staticmethod
    def PovToTime(Size, POV):
        return Size * ((1 - POV) / POV)

    @staticmethod
    def TimeToPOV(Size, TradeTime):
        return Size / (Size + TradeTime)

    @staticmethod
    def ExpectedCost(MI, PA):
        return MI + PA

    @staticmethod
    def ExpectedPrice(P0, Side, MI, PA):
        return P0 * (1 + (Side * (MI + PA) / 10000))


# ─────────────────────────────────────────────
# Trade Optimizer Class
# ─────────────────────────────────────────────
class TradeOptimizer(PreTradeTCA):
    def traders_dilemma(self, size, vol, risk_aversion):
        obj = lambda pov: self.MarketImpact(size, pov, vol) + risk_aversion * self.TimingRisk(size, pov, vol)
        res = minimize(obj, x0=0.15, bounds=[(0.01, 0.99)])
        return res.x[0]

    def minimize_cost(self, side, size, vol, alpha_bp):
        obj = lambda pov: self.MarketImpact(size, pov, vol) + self.PriceAppreciation(side, alpha_bp, size, pov)
        res = minimize(obj, x0=0.15, bounds=[(0.01, 0.99)])
        return res.x[0]

    def price_improvement(self, size, vol, bid_bps):
        obj = lambda pov: (self.MarketImpact(size, pov, vol) - bid_bps) / self.TimingRisk(size, pov, vol)
        res = minimize(obj, x0=0.15, bounds=[(0.01, 0.99)])
        return res.x[0]

    def mi_constraint(self, target_mi, vol, time_days=1.0):
        obj = lambda s: abs(self.MarketImpact(s, s / (s + 1), vol) - target_mi)
        res = minimize(obj, x0=0.05, bounds=[(0.001, 0.5)])
        return res.x[0]


# ─────────────────────────────────────────────
# Post-Trade TCA Class
# ─────────────────────────────────────────────
class PostTradeTCA:
    @staticmethod
    def implementation_shortfall(side, S, X, Pd, P0, Pn, Pavg, commission_per_share):
        R = S - X
        fixed_cost = X * commission_per_share
        paper_return = side * S * (Pn - Pd)
        portfolio_return = (side * X * (Pn - Pavg)) - fixed_cost
        is_total = paper_return - portfolio_return
        delay_cost = side * S * (P0 - Pd)
        execution_cost = side * X * (Pavg - P0)
        opportunity_cost = side * R * (Pn - P0)
        return {
            "Paper Return": paper_return,
            "Portfolio Return": portfolio_return,
            "IS Total": is_total,
            "Delay Cost": delay_cost,
            "Execution Cost": execution_cost,
            "Opportunity Cost": opportunity_cost,
            "Fixed Cost": fixed_cost,
        }

    @staticmethod
    def arrival_cost_bp(side, Pavg, P0):
        return side * ((Pavg - P0) / P0) * 10000

    @staticmethod
    def vwap_slippage_bp(side, Pavg, VWAP):
        return side * ((Pavg - VWAP) / VWAP) * 10000

    @staticmethod
    def benchmark_cost_bp(side, Pavg, Pb):
        return side * ((Pavg - Pb) / Pb) * 10000

    @staticmethod
    def value_add(expected_cost, actual_arrival_cost, timing_risk):
        value_add = expected_cost - actual_arrival_cost
        z_score = value_add / timing_risk
        return {"Value Add": value_add, "Z-Score": z_score}

    @staticmethod
    def calculate_rpm(side, p_avg, trade_data):
        total_volume = sum(v for p, v in trade_data)
        if side == 1:
            vol_above = sum(v for p, v in trade_data if p > p_avg)
            vol_at = sum(v for p, v in trade_data if p == p_avg)
            rpm = (vol_above + 0.5 * vol_at) / total_volume
        else:
            vol_below = sum(v for p, v in trade_data if p < p_avg)
            vol_at = sum(v for p, v in trade_data if p == p_avg)
            rpm = (vol_below + 0.5 * vol_at) / total_volume
        return rpm * 100


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
tca_pre = PreTradeTCA()
opt = TradeOptimizer()
tca_post = PostTradeTCA()

SIDE_OPTIONS = {"Buy (1)": 1, "Sell (-1)": -1}

def metric_card(label, value, unit=""):
    st.metric(label=label, value=f"{value:,.4f} {unit}".strip())

def result_table(data: dict, fmt=",.4f"):
    rows = "".join(
        f"<tr><td style='padding:6px 16px 6px 0;color:#888'>{k}</td>"
        f"<td style='padding:6px 0;font-weight:600;font-family:monospace'>{v:{fmt}}</td></tr>"
        for k, v in data.items()
    )
    st.markdown(
        f"<table style='border-collapse:collapse;width:100%'>{rows}</table>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Algo TCA Calculator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📊 Algo TCA Calculator")
st.caption("Pre-Trade & Post-Trade Transaction Cost Analysis")

tab_pre, tab_post = st.tabs(["🔵 Pre-Trade (HW 7 & 8)", "🟢 Post-Trade (HW 6)"])

# ─────────────────────────────────────────────
# Sidebar: Model Parameters
# ─────────────────────────────────────────────
st.sidebar.header("⚙️ Model Parameters")
st.sidebar.caption("Non-linear regression coefficients")

a1 = st.sidebar.number_input("a1 (Scale)", value=883.58722, format="%.5f")
a2 = st.sidebar.number_input("a2 (Size Exponent)", value=0.35408, format="%.5f")
a3 = st.sidebar.number_input("a3 (Vol Exponent)", value=0.755684, format="%.5f")
a4 = st.sidebar.number_input("a4 (POV Exponent)", value=0.826155, format="%.5f")
b1 = st.sidebar.number_input("b1 (Weight)", value=0.963532, format="%.5f")

# ══════════════════════════════════════════════
# PRE-TRADE TAB
# ══════════════════════════════════════════════
with tab_pre:
    # pre_section = st.sidebar.radio(
    #     "Pre-Trade Section",
    #     [
    #         "MI / TR / PA",
    #         "Expected Cost & Price",
    #         "Trader's Dilemma",
    #         "Cost Distribution (PDF)",
    #         "Cumulative Distribution (CDF)",
    #         "Price Improvement",
    #         "Efficient Frontier",
    #         "MI Constraint",
    #     ],
    #     key="pre_section",
    # ) if st.session_state.get("active_tab", "pre") == "pre" else None

    # ── Store active tab ──
    col_tabs = st.columns(2)

    # ── MI / TR / PA ──────────────────────────────────────────────────────
    st.subheader("Market Impact · Timing Risk · Price Appreciation")
    with st.expander("MI / TR / PA Calculator", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            side_mi = SIDE_OPTIONS[st.selectbox("Side", list(SIDE_OPTIONS), key="mi_side")]
            size_mi = st.number_input("Size (fraction of ADV)", value=0.10, min_value=0.001, max_value=1.0, step=0.01, format="%.4f", key="mi_size")
        with c2:
            vol_mi = st.number_input("Volatility (annual, decimal)", value=0.35, min_value=0.01, max_value=2.0, step=0.01, format="%.4f", key="mi_vol")
            alpha_mi = st.number_input("Alpha (bps)", value=30.0, step=1.0, key="mi_alpha")
        with c3:
            input_type = st.radio("Input via", ["POV Rate", "Trade Time (days)"], key="mi_input_type")
            if input_type == "POV Rate":
                pov_mi = st.number_input("POV Rate (0–1)", value=0.20, min_value=0.01, max_value=0.99, step=0.01, format="%.4f", key="mi_pov")
            else:
                tt_mi = st.number_input("Trade Time (days)", value=0.5, min_value=0.001, step=0.1, format="%.4f", key="mi_tt")
                pov_mi = tca_pre.TimeToPOV(size_mi, tt_mi)

        if st.button("Calculate", key="btn_mi"):
            mi = tca_pre.MarketImpact(size_mi, pov_mi, vol_mi)
            tr = tca_pre.TimingRisk(size_mi, pov_mi, vol_mi)
            pa = tca_pre.PriceAppreciation(side_mi, alpha_mi, size_mi, pov_mi)
            tt = tca_pre.PovToTime(size_mi, pov_mi)
            result_table({
                "Market Impact (bps)": mi,
                "Timing Risk (bps)": tr,
                "Price Appreciation (bps)": pa,
                "POV Rate": pov_mi * 100,
                "Trade Time (days)": tt,
            })

    # ── Expected Cost & Price ─────────────────────────────────────────────
    st.subheader("Expected Cost & Expected Price")
    with st.expander("Expected Cost & Price Calculator"):
        c1, c2, c3 = st.columns(3)
        with c1:
            side_ep = SIDE_OPTIONS[st.selectbox("Side", list(SIDE_OPTIONS), key="ep_side")]
            size_ep = st.number_input("Size (fraction of ADV)", value=0.05, min_value=0.001, max_value=1.0, step=0.01, format="%.4f", key="ep_size")
        with c2:
            vol_ep = st.number_input("Volatility (annual, decimal)", value=0.25, min_value=0.01, max_value=2.0, step=0.01, format="%.4f", key="ep_vol")
            alpha_ep = st.number_input("Alpha (bps)", value=10.0, step=1.0, key="ep_alpha")
        with c3:
            price_ep = st.number_input("Arrival Price (P0)", value=75.0, step=0.01, format="%.4f", key="ep_price")
            tt_ep = st.number_input("Trade Time (days)", value=1.0, min_value=0.001, step=0.1, format="%.4f", key="ep_tt")

        if st.button("Calculate", key="btn_ep"):
            pov_ep = tca_pre.TimeToPOV(size_ep, tt_ep)
            mi = tca_pre.MarketImpact(size_ep, pov_ep, vol_ep)
            tr = tca_pre.TimingRisk(size_ep, pov_ep, vol_ep)
            pa = tca_pre.PriceAppreciation(side_ep, alpha_ep, size_ep, pov_ep)
            ec = tca_pre.ExpectedCost(mi, pa)
            ep = tca_pre.ExpectedPrice(price_ep, side_ep, mi, pa)
            result_table({
                "Market Impact (bps)": mi,
                "Timing Risk (bps)": tr,
                "Price Appreciation (bps)": pa,
                "POV Rate (%)": pov_ep * 100,
                "Expected Cost (bps)": ec,
                "Expected Price ($)": ep,
            })

    # ── Trader's Dilemma ──────────────────────────────────────────────────
    st.subheader("Trader's Dilemma")
    with st.expander("Trader's Dilemma Optimizer & Chart"):
        c1, c2, c3 = st.columns(3)
        with c1:
            size_td = st.number_input("Size (fraction of ADV)", value=0.10, min_value=0.001, max_value=1.0, step=0.01, format="%.4f", key="td_size")
        with c2:
            vol_td = st.number_input("Volatility (annual, decimal)", value=0.30, min_value=0.01, max_value=2.0, step=0.01, format="%.4f", key="td_vol")
        with c3:
            lam_td = st.number_input("Risk Aversion (λ)", value=1.0, min_value=0.01, step=0.1, format="%.2f", key="td_lam")

        if st.button("Solve & Plot", key="btn_td"):
            opt_pov = opt.traders_dilemma(size_td, vol_td, lam_td)
            opt_time = tca_pre.PovToTime(size_td, opt_pov)
            opt_mi = tca_pre.MarketImpact(size_td, opt_pov, vol_td)
            opt_tr = tca_pre.TimingRisk(size_td, opt_pov, vol_td)
            opt_cost = opt_mi + lam_td * opt_tr

            result_table({
                "Optimal POV (%)": opt_pov * 100,
                "Optimal Trade Time (days)": opt_time,
                "Market Impact (bps)": opt_mi,
                "Timing Risk (bps)": opt_tr,
                "Min Total Cost (bps)": opt_cost,
            })

            pov_range = np.linspace(0.01, 0.8, 120)
            times = [tca_pre.PovToTime(size_td, p) for p in pov_range]
            mi_vals = [tca_pre.MarketImpact(size_td, p, vol_td) for p in pov_range]
            tr_vals = [lam_td * tca_pre.TimingRisk(size_td, p, vol_td) for p in pov_range]
            total = [m + t for m, t in zip(mi_vals, tr_vals)]

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(times, mi_vals, color="steelblue", label="Market Impact (MI)", alpha=0.8)
            ax.plot(times, tr_vals, color="orange", label=f"λ × Timing Risk", alpha=0.8)
            ax.plot(times, total, color="green", label="Total Cost", linewidth=2.5)
            ax.scatter(opt_time, opt_cost, color="red", s=100, zorder=5, label=f"Optimal: {opt_cost:.2f} bps")
            ax.axhline(opt_cost, color="black", linestyle="--", alpha=0.5)
            ax.axvline(opt_time, color="black", linestyle="--", alpha=0.5)
            ax.annotate(f"{opt_cost:.2f} bps @ {opt_time:.3f} days",
                        xy=(opt_time, opt_cost), xytext=(opt_time + 0.25, opt_cost + 8),
                        arrowprops=dict(arrowstyle="->", color="black"))
            ax.set_xlim(0, 3); ax.set_ylim(0, max(total) * 1.1)
            ax.set_xlabel("Trade Time (Days)"); ax.set_ylabel("Cost (bps)")
            ax.set_title(f"Trader's Dilemma  |  Size={size_td}, Vol={vol_td}, λ={lam_td}")
            ax.legend(); ax.grid(True, linestyle="--", alpha=0.4)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    # ── Cost Distribution (PDF) ───────────────────────────────────────────
    st.subheader("Trade Cost Distribution (PDF)")
    with st.expander("PDF Chart"):
        c1, c2, c3 = st.columns(3)
        with c1:
            size_pdf = st.number_input("Size", value=0.10, min_value=0.001, max_value=1.0, step=0.01, format="%.4f", key="pdf_size")
        with c2:
            vol_pdf = st.number_input("Volatility", value=0.30, min_value=0.01, step=0.01, format="%.4f", key="pdf_vol")
        with c3:
            pov_pdf = st.number_input("POV Rate (0–1)", value=0.20, min_value=0.01, max_value=0.99, step=0.01, format="%.4f", key="pdf_pov")

        if st.button("Plot PDF", key="btn_pdf"):
            mi = tca_pre.MarketImpact(size_pdf, pov_pdf, vol_pdf)
            tr = tca_pre.TimingRisk(size_pdf, pov_pdf, vol_pdf)
            x = np.linspace(mi - 3 * tr, mi + 3 * tr, 200)
            pdf = norm.pdf(x, mi, tr)
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.plot(x, pdf, color="steelblue")
            ax.fill_between(x, pdf, color="steelblue", alpha=0.2)
            ax.axvline(mi, color="black", linestyle=":", linewidth=2, label=f"Mean (MI): {mi:.2f} bps")
            ax.set_xlabel("Estimated Cost (bps)"); ax.set_ylabel("Probability Density")
            ax.set_title("Trade Cost Distribution (PDF)")
            ax.legend(); ax.grid(True, linestyle="--", alpha=0.4)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    # ── CDF ───────────────────────────────────────────────────────────────
    st.subheader("Cumulative Trade Cost Distribution (CDF)")
    with st.expander("CDF Chart"):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            size_cdf = st.number_input("Size", value=0.10, min_value=0.001, max_value=1.0, step=0.01, format="%.4f", key="cdf_size")
        with c2:
            vol_cdf = st.number_input("Volatility", value=0.30, min_value=0.01, step=0.01, format="%.4f", key="cdf_vol")
        with c3:
            pov_cdf = st.number_input("POV Rate", value=0.20, min_value=0.01, max_value=0.99, step=0.01, format="%.4f", key="cdf_pov")
        with c4:
            conf_cdf = st.slider("Confidence Level", 0.05, 0.99, 0.50, 0.01, key="cdf_conf")

        if st.button("Plot CDF", key="btn_cdf"):
            mi = tca_pre.MarketImpact(size_cdf, pov_cdf, vol_cdf)
            tr = tca_pre.TimingRisk(size_cdf, pov_cdf, vol_cdf)
            x = np.linspace(mi - 4 * tr, mi + 4 * tr, 200)
            cdf_vals = norm.cdf(x, mi, tr)
            cost_at_conf = norm.ppf(conf_cdf, mi, tr)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(x, cdf_vals, color="steelblue", linewidth=2, label="CDF")
            ax.axhline(y=conf_cdf, xmin=0, xmax=(cost_at_conf - x[0]) / (x[-1] - x[0]),
                       color="black", linestyle="--", linewidth=1)
            ax.axvline(x=cost_at_conf, ymin=0, ymax=conf_cdf,
                       color="black", linestyle="--", linewidth=1)
            ax.scatter(cost_at_conf, conf_cdf, color="red", s=50, zorder=5)
            ax.text(x[0], conf_cdf + 0.02, f" {conf_cdf*100:.1f}%", fontweight="bold")
            ax.text(cost_at_conf, -0.05, f"{cost_at_conf:.2f} bps",
                    ha="center", fontweight="bold")
            ax.set_xlabel("Est. Cost (bps)"); ax.set_ylabel("CDF (Confidence Level)")
            ax.set_title("Cumulative Trade Cost Distribution")
            ax.set_xlim(x[0], x[-1]); ax.set_ylim(0, 1.05)
            ax.legend()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    # ── Price Improvement ─────────────────────────────────────────────────
    st.subheader("Price Improvement Strategy")
    with st.expander("Price Improvement Optimizer & Chart"):
        c1, c2, c3 = st.columns(3)
        with c1:
            size_pi = st.number_input("Size", value=0.10, min_value=0.001, max_value=1.0, step=0.01, format="%.4f", key="pi_size")
        with c2:
            vol_pi = st.number_input("Volatility", value=0.30, min_value=0.01, step=0.01, format="%.4f", key="pi_vol")
        with c3:
            bid_pi = st.number_input("Bid (bps)", value=40.0, step=1.0, key="pi_bid")

        if st.button("Solve & Plot", key="btn_pi"):
            opt_pov = opt.price_improvement(size_pi, vol_pi, bid_pi)
            opt_time = tca_pre.PovToTime(size_pi, opt_pov)
            opt_obj = (bid_pi - tca_pre.MarketImpact(size_pi, opt_pov, vol_pi)) / tca_pre.TimingRisk(size_pi, opt_pov, vol_pi)

            result_table({
                "Optimal POV (%)": opt_pov * 100,
                "Optimal Trade Time (days)": opt_time,
                "Max Objective (Bid−MI)/TR": opt_obj,
            })

            pov_range = np.linspace(0.01, 0.9, 100)
            times = [tca_pre.PovToTime(size_pi, p) for p in pov_range]
            obj_vals = [(bid_pi - tca_pre.MarketImpact(size_pi, p, vol_pi)) / tca_pre.TimingRisk(size_pi, p, vol_pi) for p in pov_range]

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(times, obj_vals, color="green", linewidth=2, label="Objective (Bid−MI)/TR")
            ax.scatter(opt_time, opt_obj, color="red", s=100, zorder=5, label=f"Max: {opt_obj:.4f}")
            ax.axhline(opt_obj, color="gray", linestyle="--", alpha=0.5)
            ax.axvline(opt_time, color="gray", linestyle="--", alpha=0.5)
            ax.annotate(f"Max: {opt_obj:.4f}\n@ {opt_time:.3f} days",
                        xy=(opt_time, opt_obj), xytext=(opt_time + 0.3, opt_obj),
                        arrowprops=dict(arrowstyle="->"))
            ax.set_xlabel("Trade Time (Days)"); ax.set_ylabel("Objective Value")
            ax.set_title(f"Price Improvement Strategy  (Bid={bid_pi} bps)")
            ax.legend(); ax.grid(True, linestyle="--", alpha=0.4)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    # ── Efficient Frontier ────────────────────────────────────────────────
    st.subheader("Efficient Trading Frontier")
    with st.expander("Efficient Frontier Chart"):
        c1, c2 = st.columns(2)
        with c1:
            size_ef = st.number_input("Size", value=0.10, min_value=0.001, max_value=1.0, step=0.01, format="%.4f", key="ef_size")
        with c2:
            vol_ef = st.number_input("Volatility", value=0.30, min_value=0.01, step=0.01, format="%.4f", key="ef_vol")

        if st.button("Plot Frontier", key="btn_ef"):
            lambdas = np.linspace(0.05, 3.0, 40)
            mi_pts, tr_pts = [], []
            for l in lambdas:
                p = opt.traders_dilemma(size_ef, vol_ef, l)
                mi_pts.append(tca_pre.MarketImpact(size_ef, p, vol_ef))
                tr_pts.append(tca_pre.TimingRisk(size_ef, p, vol_ef))
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(tr_pts, mi_pts, "o-", markersize=4, color="steelblue")
            ax.set_xlabel("Timing Risk (TR, bps)"); ax.set_ylabel("Market Impact (MI, bps)")
            ax.set_title("Efficient Trading Frontier")
            ax.grid(True, linestyle="--", alpha=0.4)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    # ── MI Constraint ─────────────────────────────────────────────────────
    st.subheader("Market Impact Constraint")
    with st.expander("MI Constraint Solver & Chart"):
        c1, c2, c3 = st.columns(3)
        with c1:
            target_mi = st.number_input("Target MI (bps)", value=50.0, step=1.0, key="mic_target")
        with c2:
            vol_mic = st.number_input("Volatility", value=0.25, min_value=0.01, step=0.01, format="%.4f", key="mic_vol")
        with c3:
            adv_mic = st.number_input("ADV (shares)", value=2_000_000, step=100_000, key="mic_adv")

        if st.button("Solve & Plot", key="btn_mic"):
            opt_size = opt.mi_constraint(target_mi, vol_mic, 1.0)
            shares = opt_size * adv_mic
            mi_check = tca_pre.MarketImpact(opt_size, tca_pre.TimeToPOV(opt_size, 1.0), vol_mic)

            result_table({
                "Max Size (fraction of ADV)": opt_size,
                "Max Shares": shares,
                "Verified MI (bps)": mi_check,
            })

            sizes = np.linspace(0.001, 0.5, 100)
            shares_range = sizes * adv_mic
            mi_vals = [tca_pre.MarketImpact(s, tca_pre.TimeToPOV(s, 1.0), vol_mic) for s in sizes]

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(shares_range, mi_vals, color="steelblue", linewidth=2, label="MI Cost")
            ax.axhline(target_mi, xmin=0, xmax=shares / max(shares_range),
                       color="black", linestyle="--", linewidth=1)
            ax.axvline(shares, ymin=0, ymax=target_mi / max(mi_vals),
                       color="black", linestyle="--", linewidth=1)
            ax.scatter(shares, target_mi, color="red", s=60, zorder=5)
            ax.annotate(f"Max Shares: {shares:,.0f}\nMI: {target_mi:.1f} bps",
                        xy=(shares, target_mi), xytext=(shares + 0.04 * adv_mic, target_mi - 8),
                        arrowprops=dict(arrowstyle="->"))
            ax.set_xlabel("Shares"); ax.set_ylabel("MI Cost (bps)")
            ax.set_title("MI Cost as a Function of Shares")
            ax.set_xlim(0, max(shares_range)); ax.set_ylim(0, max(mi_vals))
            ax.legend(loc="upper left")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)


# ══════════════════════════════════════════════
# POST-TRADE TAB
# ══════════════════════════════════════════════
with tab_post:

    # ── Implementation Shortfall ──────────────────────────────────────────
    st.subheader("Implementation Shortfall")
    with st.expander("IS Calculator", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            side_is = SIDE_OPTIONS[st.selectbox("Side", list(SIDE_OPTIONS), key="is_side")]
            S_is = st.number_input("Order Size S (shares)", value=200_000, step=1000, key="is_S")
            X_is = st.number_input("Executed Shares X", value=180_000, step=1000, key="is_X")
        with c2:
            Pd_is = st.number_input("Decision Price Pd ($)", value=60.00, step=0.01, format="%.4f", key="is_Pd")
            P0_is = st.number_input("Arrival Price P0 ($)", value=60.05, step=0.01, format="%.4f", key="is_P0")
            Pn_is = st.number_input("Close Price Pn ($)", value=60.65, step=0.01, format="%.4f", key="is_Pn")
        with c3:
            Pavg_is = st.number_input("Avg Exec Price Pavg ($)", value=60.45, step=0.01, format="%.4f", key="is_Pavg")
            comm_is = st.number_input("Commission / Share ($)", value=0.01, step=0.001, format="%.4f", key="is_comm")

        if st.button("Calculate IS", key="btn_is"):
            res = tca_post.implementation_shortfall(side_is, S_is, X_is, Pd_is, P0_is, Pn_is, Pavg_is, comm_is)
            result_table({k + " ($)": v for k, v in res.items()}, fmt=",.2f")

    # ── Arrival Cost ──────────────────────────────────────────────────────
    st.subheader("Arrival Cost")
    with st.expander("Arrival Cost Calculator"):
        c1, c2, c3 = st.columns(3)
        with c1:
            side_ac = SIDE_OPTIONS[st.selectbox("Side", list(SIDE_OPTIONS), key="ac_side")]
        with c2:
            pavg_ac = st.number_input("Avg Exec Price Pavg ($)", value=25.30, step=0.01, format="%.4f", key="ac_pavg")
        with c3:
            p0_ac = st.number_input("Arrival Price P0 ($)", value=25.15, step=0.01, format="%.4f", key="ac_p0")

        if st.button("Calculate", key="btn_ac"):
            cost = tca_post.arrival_cost_bp(side_ac, pavg_ac, p0_ac)
            st.metric("Arrival Cost", f"{cost:.4f} bps")

    # ── VWAP Slippage ─────────────────────────────────────────────────────
    st.subheader("VWAP Slippage")
    with st.expander("VWAP Slippage Calculator"):
        c1, c2, c3 = st.columns(3)
        with c1:
            side_vw = SIDE_OPTIONS[st.selectbox("Side", list(SIDE_OPTIONS), key="vw_side")]
        with c2:
            pavg_vw = st.number_input("Avg Exec Price ($)", value=55.37, step=0.01, format="%.4f", key="vw_pavg")
        with c3:
            vwap_vw = st.number_input("VWAP ($)", value=55.35, step=0.01, format="%.4f", key="vw_vwap")

        if st.button("Calculate", key="btn_vw"):
            slip = tca_post.vwap_slippage_bp(side_vw, pavg_vw, vwap_vw)
            st.metric("VWAP Slippage", f"{slip:.5f} bps")

    # ── Benchmark Cost ────────────────────────────────────────────────────
    st.subheader("Benchmark Cost")
    with st.expander("Benchmark Cost Calculator"):
        c1, c2, c3 = st.columns(3)
        with c1:
            side_bc = SIDE_OPTIONS[st.selectbox("Side", list(SIDE_OPTIONS), key="bc_side")]
        with c2:
            pavg_bc = st.number_input("Avg Exec Price ($)", value=150.65, step=0.01, format="%.4f", key="bc_pavg")
        with c3:
            pb_bc = st.number_input("Benchmark Price Pb ($)", value=150.95, step=0.01, format="%.4f", key="bc_pb")

        if st.button("Calculate", key="btn_bc"):
            bc = tca_post.benchmark_cost_bp(side_bc, pavg_bc, pb_bc)
            st.metric("Benchmark Cost", f"{bc:.5f} bps")

    # ── Value Add ─────────────────────────────────────────────────────────
    st.subheader("Broker Value-Add & Z-Score")
    with st.expander("Value-Add Calculator"):
        c1, c2 = st.columns(2)
        with c1:
            side_va = SIDE_OPTIONS[st.selectbox("Side", list(SIDE_OPTIONS), key="va_side")]
            pavg_va = st.number_input("Avg Exec Price ($)", value=135.35, step=0.01, format="%.4f", key="va_pavg")
            p0_va = st.number_input("Arrival Price P0 ($)", value=134.90, step=0.01, format="%.4f", key="va_p0")
        with c2:
            exp_cost_va = st.number_input("Expected Cost (MI + PA, bps)", value=30.0, step=1.0, key="va_expcost")
            tr_va = st.number_input("Timing Risk (bps)", value=40.0, step=1.0, key="va_tr")

        if st.button("Calculate", key="btn_va"):
            arr_cost = tca_post.arrival_cost_bp(side_va, pavg_va, p0_va)
            va = tca_post.value_add(exp_cost_va, arr_cost, tr_va)
            result_table({
                "Arrival Cost (bps)": arr_cost,
                "Value Add (bps)": va["Value Add"],
                "Z-Score": va["Z-Score"],
            })

    # ── RPM ───────────────────────────────────────────────────────────────
    st.subheader("Relative Performance Measure (RPM)")
    with st.expander("RPM Calculator"):
        st.caption("Enter market trade data as comma-separated price:volume pairs, one per line. Example: `25.50:5000`")
        col_l, col_r = st.columns(2)
        with col_l:
            rpm_raw = st.text_area(
                "Market Trade Data (price:volume, one per line)",
                value="\n".join([
                    "24.50:5000", "26.50:2500", "27.50:9000", "25.25:4000",
                    "25.50:5000", "26.00:3000", "25.00:5000", "27.10:5000",
                    "24.00:9000", "24.75:2500",
                ]),
                height=180,
                key="rpm_data",
            )
        with col_r:
            side_rpm = SIDE_OPTIONS[st.selectbox("Side", list(SIDE_OPTIONS), key="rpm_side")]
            pavg_rpm = st.number_input("Your Avg Exec Price ($)", value=25.50, step=0.01, format="%.4f", key="rpm_pavg")

        if st.button("Calculate RPM", key="btn_rpm"):
            try:
                trade_data = []
                for line in rpm_raw.strip().splitlines():
                    p, v = line.strip().split(":")
                    trade_data.append((float(p), float(v)))
                rpm = tca_post.calculate_rpm(side_rpm, pavg_rpm, trade_data)
                st.metric("RPM", f"{rpm:.5f}%")

                # Quick bar chart
                prices = [p for p, v in trade_data]
                vols = [v for p, v in trade_data]
                colors = []
                for p in prices:
                    if side_rpm == 1:
                        colors.append("green" if p > pavg_rpm else ("gray" if p == pavg_rpm else "salmon"))
                    else:
                        colors.append("green" if p < pavg_rpm else ("gray" if p == pavg_rpm else "salmon"))
                fig, ax = plt.subplots(figsize=(9, 4))
                ax.bar([str(p) for p in prices], vols, color=colors)
                ax.axhline(0, color="black", linewidth=0.5)
                ax.set_xlabel("Price ($)"); ax.set_ylabel("Volume")
                fav_label = "Favorable (below avg)" if side_rpm == -1 else "Favorable (above avg)"
                ax.set_title(f"Trade Volume by Price  |  {'Buy' if side_rpm==1 else 'Sell'} RPM = {rpm:.2f}%")
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor="green", label=fav_label),
                    Patch(facecolor="salmon", label="Unfavorable"),
                    Patch(facecolor="gray", label="At avg price"),
                ]
                ax.legend(handles=legend_elements)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            except Exception as e:
                st.error(f"Error parsing trade data: {e}")