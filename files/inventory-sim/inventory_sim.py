"""Microcontroller Inventory Simulator.

Monte Carlo sim of a reorder point (s, Q) policy where both daily demand and the
supplier's lead time are random. Run it directly to rebuild charts/ and results/.
Numbers are made up for illustration.
"""
import math
import os
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass(frozen=True)
class Item:
    name: str = "32-bit microcontroller, 2,500 per reel"
    unit: str = "MCUs"
    demand_mean: float = 1200.0   # MCUs/day
    demand_sd: float = 360.0
    lead_mean: float = 21.0       # days from PO to receipt, 3 weeks from the distributor
    lead_sd: float = 6.0
    unit_cost: float = 6.50       # $/MCU
    holding_rate: float = 0.25    # per year, a bit higher for electronics because parts go obsolete
    order_cost: float = 400.0     # $/PO, includes receiving and incoming inspection
    shortage_cost: float = 15.0   # $/MCU short, lost margin and expediting when a board build slips
    days_per_year: int = 365

    @property
    def annual_demand(self):
        return self.demand_mean * self.days_per_year

    @property
    def holding_cost(self):  # $/MCU/yr
        return self.unit_cost * self.holding_rate


# textbook formulas
def eoq(item):
    return math.sqrt(2 * item.annual_demand * item.order_cost / item.holding_cost)


def sigma_dlt(item):
    # std dev of demand during lead time when demand AND lead time vary
    return math.sqrt(item.lead_mean * item.demand_sd ** 2 + item.demand_mean ** 2 * item.lead_sd ** 2)


def safety_stock(item, csl):
    return norm.ppf(csl) * sigma_dlt(item)


def reorder_point(item, csl):
    return item.demand_mean * item.lead_mean + safety_stock(item, csl)


def analytic_annual_cost(item, q, ss, unit_cost=None):
    # standard EOQ cost pieces, no shortage cost here
    p = item.unit_cost if unit_cost is None else unit_cost
    h = p * item.holding_rate
    return {
        "purchase": item.annual_demand * p,
        "ordering": item.annual_demand / q * item.order_cost,
        "cycle_holding": q / 2 * h,
        "safety_holding": ss * h,
    }


def _gamma(rng, mean, sd, size):
    # gamma instead of normal: lead times can't go negative and late shipments
    # are more common than early ones, so the right skew matters
    # sd of 0 just means a fixed value
    if sd <= 0:
        return np.full(size, mean, dtype=float)
    shape = (mean / sd) ** 2
    scale = sd ** 2 / mean
    return rng.gamma(shape, scale, size)


def simulate(item, s, q, reps=400, days=365, warmup=90, seed=7, trace=False):
    """Run an (s, Q) policy reviewed daily for `reps` separate years. Shortages are lost
    sales (line stops), not backorders. Each array is one value per simulated year."""
    rng = np.random.default_rng(seed)
    total = warmup + days
    max_lead = int(item.lead_mean + 12 * item.lead_sd) + 2
    demand = np.rint(_gamma(rng, item.demand_mean, item.demand_sd, (reps, total))).astype(int)

    on_hand = np.full(reps, int(round(s + q / 2)))  # start somewhere mid-cycle, warmup handles the rest
    pipeline = np.zeros((reps, total + max_lead + 1), dtype=int)
    on_order = np.zeros(reps, dtype=int)
    stockout_in_cycle = np.zeros(reps, dtype=bool)
    last_arrival = np.zeros(reps, dtype=int)  # orders come in the order they were placed

    filled = np.zeros(reps); demanded = np.zeros(reps); short = np.zeros(reps)
    orders = np.zeros(reps); cycles = np.zeros(reps); bad_cycles = np.zeros(reps)
    oh_sum = np.zeros(reps); stockout_days = np.zeros(reps)
    trace_oh = []
    trace_ip = []
    rows = np.arange(reps)

    for t in range(total):
        # receive whatever is due today
        arriving = pipeline[:, t]
        on_hand += arriving
        on_order -= arriving
        d = demand[:, t]
        f = np.minimum(on_hand, d)
        on_hand -= f
        shortfall = d - f
        stockout_in_cycle |= shortfall > 0
        # reorder off inventory position (on hand + on order), not on hand
        ip = on_hand + on_order
        need = ip <= s
        n_orders = np.where(need, np.ceil((s - ip + 1) / q), 0).astype(int)
        if need.any():
            idx = rows[need]
            for k in range(int(n_orders.max())):  # almost always just 1 order
                sub = idx[n_orders[idx] > k]
                if sub.size == 0:
                    break
                lt = np.maximum(1, np.rint(_gamma(rng, item.lead_mean, item.lead_sd, sub.size))).astype(int)
                # a supplier ships POs in order, so a PO can't show up before the one placed ahead of it
                arr = np.maximum(t + lt, last_arrival[sub])
                last_arrival[sub] = arr
                np.add.at(pipeline, (sub, arr), int(q))
                on_order[sub] += int(q)
            if t >= warmup:
                cycles[idx] += 1
                bad_cycles[idx] += stockout_in_cycle[idx]
                orders += n_orders
            stockout_in_cycle[idx] = False
        if t >= warmup:
            filled += f; demanded += d; short += shortfall
            oh_sum += on_hand; stockout_days += shortfall > 0
            if trace:
                trace_oh.append(on_hand[0])
                trace_ip.append(on_hand[0] + on_order[0])

    avg_oh = oh_sum / days
    holding = avg_oh * item.holding_cost
    ordering = orders * item.order_cost
    shortage = short * item.shortage_cost
    total_cost = holding + ordering + shortage
    fill = filled / demanded
    csl = 1 - np.divide(bad_cycles, cycles, out=np.zeros(reps), where=cycles > 0)

    def summ(x):
        return float(np.mean(x)), float(np.percentile(x, 5)), float(np.percentile(x, 95))

    out = {
        "s": s, "Q": q,
        "fill_rate": summ(fill)[0], "cycle_service_level": summ(csl)[0],
        "avg_on_hand": summ(avg_oh)[0], "inventory_value": summ(avg_oh)[0] * item.unit_cost,
        "orders_per_year": summ(orders)[0], "stockout_days": summ(stockout_days)[0],
        "units_short": summ(short)[0],
        "holding_cost": summ(holding)[0], "ordering_cost": summ(ordering)[0],
        "shortage_cost": summ(shortage)[0], "total_cost": summ(total_cost)[0],
        "total_cost_p5": summ(total_cost)[1], "total_cost_p95": summ(total_cost)[2],
    }
    if trace:
        out["trace_on_hand"] = np.array(trace_oh)
        out["trace_position"] = np.array(trace_ip)
        out["trace_demand"] = demand[0, warmup:]
    return out


TARGETS = [0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995, 0.998, 0.999, 0.9995, 0.9999]


def exp_service_frontier(item, reps=400):
    q = eoq(item)
    rows = []
    for csl in TARGETS:
        s = reorder_point(item, csl)
        r = simulate(item, s, q, reps=reps, seed=11)
        rows.append({"target_csl": csl, "z": norm.ppf(csl), "safety_stock": safety_stock(item, csl),
                     "safety_stock_value": safety_stock(item, csl) * item.unit_cost, **r})
    return pd.DataFrame(rows)


def exp_lead_time_variability(item, csl=0.98, reps=400, sds=(0.0, 1.5, 3.0, 4.5, 6.0, 9.0, 12.0)):
    rows = []
    for sd in sds:
        it = replace(item, lead_sd=sd)
        q = eoq(it)
        s = reorder_point(it, csl)
        r = simulate(it, s, q, reps=reps, seed=23)
        ss = safety_stock(it, csl)
        rows.append({"lead_sd": sd, "sigma_dlt": sigma_dlt(it), "safety_stock": ss,
                     "safety_stock_value": ss * it.unit_cost, "safety_holding_cost": ss * it.holding_cost,
                     **{k: r[k] for k in ("fill_rate", "cycle_service_level", "total_cost", "shortage_cost")}})
    return pd.DataFrame(rows)


def exp_quantity_discount(item, csl, break_qty=25000, discount=0.03):
    # all-units discount: EOQ at list price vs ordering at the break
    ss = safety_stock(item, csl)
    q_list = eoq(item)
    disc_price = item.unit_cost * (1 - discount)
    q_disc_eoq = math.sqrt(2 * item.annual_demand * item.order_cost / (disc_price * item.holding_rate))
    q_disc = max(break_qty, q_disc_eoq)
    a = analytic_annual_cost(item, q_list, ss)
    b = analytic_annual_cost(item, q_disc, ss, unit_cost=disc_price)
    curve = []
    for q in np.linspace(2500, 50000, 120):
        p = disc_price if q >= break_qty else item.unit_cost
        c = analytic_annual_cost(item, q, ss, unit_cost=p)
        c_list = analytic_annual_cost(item, q, ss)
        c_disc = analytic_annual_cost(item, q, ss, unit_cost=disc_price)
        curve.append({"Q": q, "effective_total": sum(c.values()), "list_total": sum(c_list.values()),
                      "discount_total": sum(c_disc.values())})
    return {"q_list": q_list, "q_disc": q_disc, "disc_price": disc_price, "break_qty": break_qty,
            "discount": discount, "cost_list": a, "cost_disc": b,
            "total_list": sum(a.values()), "total_disc": sum(b.values()),
            "savings": sum(a.values()) - sum(b.values()), "curve": pd.DataFrame(curve)}


ACCENT, INK, MUTED, GRID, AXIS, RED = "#2440E8", "#17181A", "#6B6D70", "#E1E0D9", "#C3C2B7", "#9E2A20"
ORANGE = "#EB6834"


def _style(ax, title, subtitle=None):
    ax.set_facecolor("white")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=12.5, fontweight="bold", color=INK, pad=24 if subtitle else 10)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9.5, color=MUTED, va="bottom")


def make_charts(item, frontier, ltv, qd, trace, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, PercentFormatter
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 150, "text.parse_math": False})
    from matplotlib.ticker import MultipleLocator
    os.makedirs(outdir, exist_ok=True)
    kfmt = FuncFormatter(lambda v, _: f"${v/1000:,.0f}K")
    paths = []

    # 1. cost vs service level
    fig, ax = plt.subplots(figsize=(8, 4.6))
    x = pd.Series(np.arange(len(frontier)), dtype=float)
    xt = [f"{v*100:g}" for v in frontier["target_csl"]]
    lines = [("total_cost", "Total cost", INK, 2.4), ("holding_cost", "Holding", ACCENT, 2),
             ("shortage_cost", "Shortages", RED, 2), ("ordering_cost", "Ordering", MUTED, 2)]
    ends = []
    for col, lab, colr, lw in lines:
        ax.plot(x, frontier[col], color=colr, lw=lw, solid_capstyle="round")
        ends.append([frontier[col].iloc[-1], lab])
    # nudge end labels apart so they don't sit on top of each other
    gap = frontier["total_cost"].max() * 0.055
    ends.sort(key=lambda e: e[0])
    for i in range(1, len(ends)):
        if ends[i][0] - ends[i - 1][0] < gap:
            ends[i][0] = ends[i - 1][0] + gap
    for yv, lab in ends:
        ax.text(x.iloc[-1] + 0.25, yv, lab, color=INK, fontsize=9, va="center")
    bi = int(frontier["total_cost"].idxmin())
    best = frontier.loc[bi]
    ax.scatter([x.iloc[bi]], [best["total_cost"]], s=60, color=INK, zorder=5, edgecolor="white", linewidth=2)
    ax.annotate(f"Lowest at {best['target_csl']:.1%} (${best['total_cost']/1000:,.1f}K/yr)\nbut pretty flat from 99.8% on",
                (x.iloc[bi], best["total_cost"]), xytext=(-215, 46), textcoords="offset points",
                fontsize=9, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.yaxis.set_major_formatter(kfmt)
    ax.set_xticks(x, xt, fontsize=8)
    ax.set_xlabel("Target cycle service level used to set safety stock (%)", color=MUTED, fontsize=9)
    ax.set_xlim(-0.4, x.max() + 1.6)
    _style(ax, "Annual cost by service level target",
           f"400 simulated years per policy, order qty = EOQ ({eoq(item):,.0f} {item.unit})")
    fig.tight_layout()
    p = os.path.join(outdir, "1_cost_vs_service_level.png"); fig.savefig(p, facecolor="white"); plt.close(fig); paths.append(p)

    # 2. target vs achieved service
    fig, ax = plt.subplots(figsize=(8, 4.6))
    tgt = frontier["target_csl"] * 100
    ax.plot(x, tgt, color=AXIS, lw=1.2)
    ax.text(x.iloc[1], tgt.iloc[1] - 2.2, "Target", color=MUTED, fontsize=8.5)
    ax.plot(x, frontier["cycle_service_level"] * 100, color=ACCENT, lw=2, marker="o", ms=5, mec="white", mew=1.5)
    ax.plot(x, frontier["fill_rate"] * 100, color=ORANGE, lw=2, marker="o", ms=5, mec="white", mew=1.5)
    ax.text(x.iloc[-1] + 0.3, frontier["cycle_service_level"].iloc[-1] * 100 - 0.9, "Simulated cycle\nservice level", color=INK, fontsize=9, va="center")
    ax.text(x.iloc[-1] + 0.3, frontier["fill_rate"].iloc[-1] * 100 + 0.7, "Simulated fill rate", color=INK, fontsize=9, va="center")
    ax.set_xlim(-0.4, x.max() + 2.6)
    ax.set_xticks(x, xt, fontsize=8)
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.set_xlabel("Target cycle service level used to set safety stock (%)", color=MUTED, fontsize=9)
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    _style(ax, "Target vs. simulated service level",
           "Safety stock set with the normal approximation formula")
    fig.tight_layout()
    p = os.path.join(outdir, "2_target_vs_achieved_service.png"); fig.savefig(p, facecolor="white"); plt.close(fig); paths.append(p)

    # 3. lead-time variability
    fig, ax = plt.subplots(figsize=(8, 4.6))
    bars = ax.bar(ltv["lead_sd"].map(lambda v: f"{v:g}"), ltv["safety_stock_value"], color=ACCENT, width=0.55)
    cur = ltv.index[ltv["lead_sd"] == item.lead_sd]
    for i, b in enumerate(bars):
        if len(cur) and i == cur[0]:
            b.set_color(INK)
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + ltv["safety_stock_value"].max() * 0.015,
                f"${b.get_height()/1000:,.0f}K", ha="center", fontsize=9, color=INK)
    ax.yaxis.set_major_formatter(kfmt)
    ax.set_xlabel(f"Lead time std dev in days (average lead time is {item.lead_mean:g} days)", color=MUTED, fontsize=9)
    _style(ax, "Safety stock needed vs. lead time variability",
           f"Value of safety stock for a {frontier.loc[frontier['total_cost'].idxmin(), 'target_csl']:.1%} cycle service level. Dark bar is the current supplier")
    fig.tight_layout()
    p = os.path.join(outdir, "3_lead_time_variability.png"); fig.savefig(p, facecolor="white"); plt.close(fig); paths.append(p)

    # 4. quantity discount
    fig, ax = plt.subplots(figsize=(8, 4.6))
    c = qd["curve"]
    ax.plot(c["Q"], c["list_total"] / 1e6, color=MUTED, lw=1.5)
    ax.plot(c["Q"], c["discount_total"] / 1e6, color=AXIS, lw=1.5)
    ax.plot(c["Q"], c["effective_total"] / 1e6, color=ACCENT, lw=2.4)
    ax.axvline(qd["break_qty"], color=MUTED, lw=0.8, ls=(0, (4, 4)))
    ax.text(qd["break_qty"] + 600, c["list_total"].max() / 1e6, f"Price break at {qd['break_qty']:,} {item.unit}\n({qd['discount']:.0%} off all units)",
            fontsize=8.5, color=MUTED, va="top")
    ax.scatter([qd["q_list"]], [qd["total_list"] / 1e6], color=INK, s=45, zorder=5, edgecolor="white", linewidth=2)
    ax.scatter([qd["q_disc"]], [qd["total_disc"] / 1e6], color=ACCENT, s=60, zorder=5, edgecolor="white", linewidth=2)
    ax.annotate(f"EOQ at list price\n{qd['q_list']:,.0f} {item.unit}: ${qd['total_list']/1e6:,.3f}M", (qd["q_list"], qd["total_list"] / 1e6),
                xytext=(10, 18), textcoords="offset points", fontsize=8.5, color=INK)
    ax.annotate(f"{qd['q_disc']:,.0f} at the discount\n${qd['total_disc']/1e6:,.3f}M, saves ${qd['savings']/1000:,.0f}K/yr",
                (qd["q_disc"], qd["total_disc"] / 1e6), xytext=(26, 8), textcoords="offset points", fontsize=8.5, color=INK)
    ax.text(c["Q"].iloc[-1], c["list_total"].iloc[-1] / 1e6, " List price", fontsize=8.5, color=MUTED, va="center")
    ax.text(c["Q"].iloc[-1], c["discount_total"].iloc[-1] / 1e6, " Discount price", fontsize=8.5, color=MUTED, va="center")
    ax.set_xlim(1500, 58000)
    ax.set_ylim(bottom=c["discount_total"].min() / 1e6 - 0.012)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.2f}M"))
    ax.set_xlabel(f"Order quantity ({item.unit} per PO)", color=MUTED, fontsize=9)
    _style(ax, "Total annual cost by order size",
           "Purchase + ordering + holding. Blue line is what you'd actually pay at each order size")
    fig.tight_layout()
    p = os.path.join(outdir, "4_quantity_discount.png"); fig.savefig(p, facecolor="white"); plt.close(fig); paths.append(p)

    # 5. inventory trace
    fig, ax = plt.subplots(figsize=(8, 3.8))
    oh = trace["trace_on_hand"]
    days = np.arange(len(oh))
    ip = trace["trace_position"]
    ax.plot(days, ip, color=AXIS, lw=1.1)
    ax.text(len(ip) + 2, ip.max(), "On hand +\non order", fontsize=8.5, color=MUTED, va="center")
    ax.fill_between(days, oh, color=ACCENT, alpha=0.10, lw=0)
    ax.plot(days, oh, color=ACCENT, lw=1.6)
    ax.text(len(oh) + 2, oh[-1], "On hand", fontsize=8.5, color=ACCENT, va="center")
    ax.axhline(trace["s"], color=MUTED, lw=0.8, ls=(0, (4, 4)))
    ax.text(len(oh) + 2, trace["s"], f"Reorder point\n{trace['s']:,.0f}", fontsize=8.5, color=MUTED, va="center")
    ax.axhline(trace["ss"], color=RED, lw=0.8, ls=(0, (4, 4)))
    ax.text(len(oh) + 2, trace["ss"], f"Safety stock\n{trace['ss']:,.0f}", fontsize=8.5, color=RED, va="center")
    ax.set_xlim(0, len(oh) + 48)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Day of year", color=MUTED, fontsize=9)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    _style(ax, "One simulated year of inventory", f"{item.unit} each day. A PO goes out when on hand + on order drops to the reorder point")
    fig.tight_layout()
    p = os.path.join(outdir, "5_inventory_trace.png"); fig.savefig(p, facecolor="white"); plt.close(fig); paths.append(p)
    return paths


def main(outdir="."):
    item = Item()
    q = eoq(item)
    print(f"Item: {item.name}")
    print(f"Annual demand {item.annual_demand:,.0f} {item.unit} | EOQ {q:,.0f} | sigma(DLT) {sigma_dlt(item):,.1f}")

    frontier = exp_service_frontier(item)
    best = frontier.loc[frontier["total_cost"].idxmin()]
    ltv = exp_lead_time_variability(item, csl=float(best["target_csl"]))
    qd = exp_quantity_discount(item, csl=float(best["target_csl"]))
    s_best = reorder_point(item, float(best["target_csl"]))
    tr = simulate(item, s_best, q, reps=1, seed=5, trace=True)
    tr["s"] = s_best
    tr["ss"] = safety_stock(item, float(best["target_csl"]))

    os.makedirs(os.path.join(outdir, "results"), exist_ok=True)
    frontier.to_csv(os.path.join(outdir, "results", "service_level_frontier.csv"), index=False)
    ltv.to_csv(os.path.join(outdir, "results", "lead_time_variability.csv"), index=False)
    qd["curve"].to_csv(os.path.join(outdir, "results", "quantity_discount_curve.csv"), index=False)
    paths = make_charts(item, frontier, ltv, qd, tr, os.path.join(outdir, "charts"))

    pd.set_option("display.width", 160)
    print("\nService level sweep (400 sim years each):")
    print(frontier[["target_csl", "safety_stock", "cycle_service_level", "fill_rate", "holding_cost",
                    "ordering_cost", "shortage_cost", "total_cost"]].round(3).to_string(index=False))
    print(f"\nLowest total cost: target {best['target_csl']:.1%}, s = {s_best:,.0f}, Q = {q:,.0f}, "
          f"${best['total_cost']:,.0f}/yr")
    print(f"\nLead-time variability ({best['target_csl']:.1%} target):")
    print(ltv.round(3).to_string(index=False))
    print(f"\nQuantity discount: list EOQ {qd['q_list']:,.0f} -> ${qd['total_list']:,.0f}; "
          f"order {qd['q_disc']:,.0f} at ${qd['disc_price']:.2f} -> ${qd['total_disc']:,.0f}; "
          f"savings ${qd['savings']:,.0f}/yr")
    print("\nCharts:", *paths, sep="\n  ")
    return {"item": item, "frontier": frontier, "ltv": ltv, "qd": qd, "trace": tr, "best": best}


if __name__ == "__main__":
    main(os.path.dirname(os.path.abspath(__file__)))
