"""The four benchmark metrics, computed identically for every attack.

Definitions follow the benchmark specification exactly:

  V_clean = videos the victim model classifies CORRECTLY  (the attack set)
  V_succ  = subset of V_clean the attack manages to make misclassified
            while respecting the perturbation budget

  FR  = |V_succ| / |V_clean| * 100                       (higher is better)
  QS  = mean over v in V_succ  of q(v)                   (lower is better)
  QT  = mean over v in V_clean of q(v)                   (lower is better)
  MAP = mean over v in V_succ  of ||d_v||_1 / |x_v|      (lower is better)

q(v) for QS is the number of queries spent up to and including the query that
first produced a valid adversarial example.  q(v) for QT is the total number of
queries spent on v (a failed video therefore contributes the full budget).
"""
from statistics import mean

QUERY_BUDGETS = [1000, 10000, 20000, 30000, 40000, 50000, 60000]


def compute(records, budget=None, tier=None):
    """records: list of per-video dicts written by an attack runner.

    `tier` restricts to videos of that tier or below, so the paper-faithful
    one-video-per-class numbers stay recoverable after the set is extended.
    """
    ok = [r for r in records if r.get("status") == "done"]
    if tier is not None:
        ok = [r for r in ok if (r.get("tier") or 1) <= tier]
    n_clean = len(ok)
    if n_clean == 0:
        return {"n_clean": 0, "n_succ": 0, "FR": None, "QS": None, "QT": None, "MAP": None}

    succ = [r for r in ok if r.get("success")]
    qs_vals = [r["queries_to_success"] for r in succ
               if r.get("queries_to_success") is not None]

    def q_total(r):
        v = r.get("queries_used")
        if v is None:
            v = budget if budget is not None else 0
        return v

    # Methods that minimise L1/L2 (H-Opt, CLVA) routinely produce a
    # misclassified video whose PEAK perturbation exceeds a shared L_inf budget.
    # Reporting only the eps-constrained FR would hide that they succeeded at
    # what they optimise for, so the unconstrained misclassification rate and the
    # distortion actually needed are reported alongside it.
    mis = [r for r in ok if r.get("final_pred") is not None
           and r["final_pred"] != r.get("true_label")]
    mis_map = [r["map"] for r in mis if r.get("map") is not None]
    mis_linf = [r["linf"] for r in mis if r.get("linf") is not None]

    out = {
        "n_clean": n_clean,
        "n_succ": len(succ),
        "FR_unconstrained": 100.0 * len(mis) / n_clean,
        "MAP_unconstrained": mean(mis_map) if mis_map else None,
        "LINF_unconstrained": mean(mis_linf) if mis_linf else None,
        "FR": 100.0 * len(succ) / n_clean,
        "QS": mean(qs_vals) if qs_vals else None,
        "QT": mean([q_total(r) for r in ok]),
        "MAP": (mean([r["map"] for r in succ if r.get("map") is not None])
                if any(r.get("map") is not None for r in succ) else None),
        "LINF_mean": (mean([r["linf"] for r in succ if r.get("linf") is not None])
                      if any(r.get("linf") is not None for r in succ) else None),
        "wall_seconds_total": sum(r.get("wall_seconds") or 0 for r in ok),
        "wall_seconds_mean": mean([r.get("wall_seconds") or 0 for r in ok]),
    }
    return out


def fr_vs_budget(records, budgets=QUERY_BUDGETS, tier=None):
    """Fooling rate achievable if the attack had been stopped at each budget.

    A video counts as fooled at budget B iff it succeeded and did so within B
    queries.  Denominator is always |V_clean| so the curves are comparable.
    """
    ok = [r for r in records if r.get("status") == "done"]
    if tier is not None:
        ok = [r for r in ok if (r.get("tier") or 1) <= tier]
    n_clean = len(ok)
    curve = {}
    for b in budgets:
        if n_clean == 0:
            curve[b] = None
            continue
        hit = sum(1 for r in ok
                  if r.get("success") and r.get("queries_to_success") is not None
                  and r["queries_to_success"] <= b)
        curve[b] = 100.0 * hit / n_clean
    return curve


def format_table(rows):
    """rows: list of (method, metrics-dict). Returns a markdown table."""
    hdr = "| Method | FR (%) ↑ | QS ↓ | QT ↓ | MAP ↓ | n_clean | n_succ |"
    sep = "| --- | --- | --- | --- | --- | --- | --- |"
    out = [hdr, sep]
    for name, m in rows:
        def f(x, p=2):
            return "-" if x is None else f"{x:,.{p}f}"
        out.append(f"| {name} | {f(m.get('FR'))} | {f(m.get('QS'), 1)} | "
                   f"{f(m.get('QT'), 1)} | {f(m.get('MAP'), 4)} | "
                   f"{m.get('n_clean', '-')} | {m.get('n_succ', '-')} |")
    return "\n".join(out)
