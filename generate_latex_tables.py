"""
Generate publication-ready LaTeX tables from the saved policies/*.pth results.

Produces:
  results/rh_results_table.tex   — main-body summary (mean +/- std over 3 puzzles per difficulty)
  results/rh_appendix_table.tex  — per-puzzle breakdown (one row per puzzle), for the appendix

Both use booktabs (\\toprule/\\midrule/\\bottomrule, no vertical rules), matching
common NeurIPS style. Best solve rate (then best reward as tie-break) per
difficulty column is bolded.

Run:
  python3 generate_latex_tables.py
"""
import os

from plot_rh_results import load_data, summary, DIFFICULTIES, METHODS, LABELS

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')

CAPTION_MAIN = (
    r"Rush Hour solve rate and mean evaluation reward by state representation, "
    r"averaged over 3 puzzles per difficulty (mean $\pm$ std)."
)
CAPTION_APPENDIX = (
    r"Per-puzzle Rush Hour results underlying Table~\ref{tab:rush-hour-results}. "
    r"Each eval is fully deterministic (greedy policy, deterministic environment), "
    r"so std across the 50 eval episodes is always 0; the only variation is across puzzles."
)


def _fmt(mean, std, is_pct):
    if is_pct:
        return f"{mean:.0f} $\\pm$ {std:.0f}"
    return f"{mean:+.2f} $\\pm$ {std:.2f}"


def build_summary_table(solve_mean, solve_std, reward_mean, reward_std):
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{" + CAPTION_MAIN + "}")
    lines.append(r"\label{tab:rush-hour-results}")
    lines.append(r"\begin{tabular}{l" + "cc" * len(DIFFICULTIES) + "}")
    lines.append(r"\toprule")

    header1 = [""] + [rf"\multicolumn{{2}}{{c}}{{Difficulty {D}}}" for D in DIFFICULTIES]
    lines.append(" & ".join(header1) + r" \\")

    cmid = " ".join(
        rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(DIFFICULTIES))
    )
    lines.append(cmid)

    header2 = ["Method"] + [x for _ in DIFFICULTIES for x in ("Solve (\\%)", "Reward")]
    lines.append(" & ".join(header2) + r" \\")
    lines.append(r"\midrule")

    # find the winner (highest solve rate, reward as tie-break) per difficulty
    best = {}
    for D in DIFFICULTIES:
        best[D] = max(METHODS, key=lambda M: (solve_mean[D][M], reward_mean[D][M]))

    for M, L in zip(METHODS, LABELS):
        row = [L]
        for D in DIFFICULTIES:
            s = _fmt(solve_mean[D][M], solve_std[D][M], is_pct=True)
            r = _fmt(reward_mean[D][M], reward_std[D][M], is_pct=False)
            if M == best[D]:
                s, r = rf"\textbf{{{s}}}", rf"\textbf{{{r}}}"
            row += [s, r]
        lines.append(" & ".join(row) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def build_appendix_table(data):
    """
    Uses longtable (not table+tabular) since this can be 45+ rows once
    n_puzzles is scaled up -- a fixed-height table environment would
    overflow the page instead of breaking across pages.
    """
    col_spec = "ll" + "c" * len(METHODS)
    header_row = "Difficulty & Puzzle & " + " & ".join(LABELS) + r" \\"

    lines = []
    lines.append(rf"\begin{{longtable}}{{{col_spec}}}")
    lines.append(r"\caption{" + CAPTION_APPENDIX + "}")
    lines.append(r"\label{tab:rush-hour-per-puzzle}\\")
    lines.append(r"\toprule")
    lines.append(header_row)
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(
        r"\multicolumn{" + str(len(METHODS) + 2)
        + r"}{c}{\tablename\ \thetable{} -- continued from previous page} \\"
    )
    lines.append(r"\toprule")
    lines.append(header_row)
    lines.append(r"\midrule")
    lines.append(r"\endhead")
    lines.append(r"\midrule")
    lines.append(
        r"\multicolumn{" + str(len(METHODS) + 2)
        + r"}{r}{Continued on next page} \\"
    )
    lines.append(r"\endfoot")
    lines.append(r"\bottomrule")
    lines.append(r"\endlastfoot")

    for D in DIFFICULTIES:
        n_puzzles = max((len(data[D][M]) for M in METHODS), default=0)
        for p in range(n_puzzles):
            cells = []
            for M in METHODS:
                entries = data[D][M]
                if p < len(entries):
                    e = entries[p]
                    solved = e['solve_rate'] >= 0.5
                    mark = r"\checkmark" if solved else r"\ding{55}"
                    cells.append(f"{mark} ({e['mean_reward']:+.2f})")
                else:
                    cells.append("--")
            d_label = str(D) if p == 0 else ""
            lines.append(f"{d_label} & {p + 1} & " + " & ".join(cells) + r" \\")
        if D != DIFFICULTIES[-1]:
            lines.append(r"\midrule")

    lines.append(r"\end{longtable}")
    return "\n".join(lines)


def main():
    data = load_data()
    solve_mean, solve_std, reward_mean, reward_std = summary(data)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    main_tex = build_summary_table(solve_mean, solve_std, reward_mean, reward_std)
    main_path = os.path.join(RESULTS_DIR, 'rh_results_table.tex')
    with open(main_path, 'w') as f:
        f.write(main_tex + "\n")
    print(f"  -> {main_path}")

    appendix_tex = build_appendix_table(data)
    appendix_path = os.path.join(RESULTS_DIR, 'rh_appendix_table.tex')
    with open(appendix_path, 'w') as f:
        f.write(appendix_tex + "\n")
    print(f"  -> {appendix_path}")
    print(
        "\nNote: the appendix table needs \\usepackage{amssymb}, \\usepackage{pifont} "
        "(for \\checkmark / \\ding{55}), and \\usepackage{longtable} (it now spans "
        "multiple pages once n_puzzles is scaled up)."
    )


if __name__ == '__main__':
    main()
