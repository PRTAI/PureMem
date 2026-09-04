#!/usr/bin/env python3
"""Applicability-gate counterfactual interaction figure (manuscript Table 2).

    python figure_script/plot_gate_counterfactual.py            # -> ./gate_counterfactual.{pdf,png}
    python figure_script/plot_gate_counterfactual.py --width 7.0
    python figure_script/plot_gate_counterfactual.py --outdir latex

A diverging horizontal bar (forest-style) chart: one row per frozen gate subset,
bar length = the paired accuracy change from inserting the retrieved Top-1 memory
INSTEAD OF the neutral placebo, i.e.

    Delta = acc(Real Top-1) - acc(Placebo)

so a bar to the right means the retrieved memory helped that subset and a bar to
the left means it hurt. The two directions are the whole point of the figure, so
the axis is forced symmetric — an auto-scaled asymmetric axis would make a +2.81
bar and a -3.23 bar look comparable in length.

DATA PROVENANCE — read before regenerating
------------------------------------------
``SUBSETS`` / ``SELECTIVITY`` below are the ``manifest`` block of
``bbeh/work/table2_recomputed_notrunc.json``, produced by
``bbeh/work/recompute_table2.py`` from the committed run artifacts
(``runs/memory_zpd_counterfactual_{real_top1,placebo}_gemini-3.5-flash``).

The ``notrunc`` variant is authoritative: it keeps only those queries where
BOTH arms produced an untruncated completion, so N=178/433 rather than the
192/498 of the all-queries variant. A truncated generation scores a false 0
regardless of what sat in the memory slot, which injects noise that is unrelated
to the intervention under test; restricting to complete pairs is what makes the
paired comparison actually paired. Both variants agree in sign and rough
magnitude of the selectivity contrast (+6.04 pp here, +5.38 pp with truncated
pairs retained); the notrunc variant is the significant one (p=0.0065 vs
p=0.0720), which is precisely because the discarded pairs were noise.

Do not mix cells across the two variants — an earlier draft of the manuscript
carried N from one and the deltas from the other, which corresponds to no
computation that was actually run.

CONFIDENCE INTERVALS
--------------------
``ci`` holds the paired bootstrap 95% intervals reported in the appendix, taken
verbatim from the same JSON. The discordant counts ``b`` / ``c`` are supplied
alongside them so the exact McNemar p is derived here rather than transcribed,
and so ``resolve()`` can assert that the counts and the stated delta describe
the same data. An explicit ``ci`` takes precedence over the normal-approximation
interval the counts would imply, because the appendix promises a bootstrap
interval and the two do not coincide.

Do NOT transplant an interval computed from one set of point estimates onto a
different set — a CI that is not centred on its own estimate misstates both.

Suggested caption
-----------------
\\caption{\\textbf{The applicability verifier separates helpful from harmful
memory transfer on BBEH.}
Bars report the paired accuracy change of inserting the frozen retrieved Top-1
memory rather than a neutral placebo, within subsets determined by the frozen
gate decision. Retrieved memory is beneficial on accepted queries but harmful
when counterfactually injected into rejected queries; their difference is
significant under a gate-label permutation test ($p=0.0065$).}
"""

import argparse
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ═════════════════════════════════════════════════════════════════════
#  Data — manuscript Table 2. Replace this block when the numbers settle.
# ═════════════════════════════════════════════════════════════════════

SUBSETS = [
    # delta is in percentage points, Real Top-1 minus Placebo.
    # b / c are the paired discordant counts: b = real right & placebo wrong,
    # c = real wrong & placebo right. They drive the exact McNemar p and are
    # cross-checked against delta. ci=(lo, hi) is the paired bootstrap 95%
    # interval and takes precedence over the counts' normal approximation.
    dict(key='accept', label='Accept', n=178, delta=+2.81, b=8, c=3,
         ci=(-0.56, +6.74)),
    dict(key='reject', label='Reject', n=433, delta=-3.23, b=6, c=20,
         ci=(-5.54, -1.15)),
]

SELECTIVITY = dict(value=+6.04, p=0.0065, p_label='p', ci=(+1.82, +10.47))

# ═════════════════════════════════════════════════════════════════════
#  Palette — dataviz reference instance. blue <-> orange is the endorsed
#  warm/cool diverging pair; validated all-pairs on a white surface
#  (CVD Delta-E 24.7, normal-vision 33.6, both marks >= 3:1 contrast).
#  Sign is also carried by bar direction and by a signed value label, so
#  the encoding never rests on hue alone.
# ═════════════════════════════════════════════════════════════════════

C_POS = '#2a78d6'     # helped  — categorical slot 1
C_NEG = '#eb6834'     # hurt    — categorical slot 2
C_INK = '#0b0b0b'
C_INK_2 = '#52514e'
C_MUTED = '#898781'
C_GRID = '#e1e0d9'
C_AXIS = '#c3c2b7'


def paired_ci(b, c, n, z=1.96):
    """95% CI (in pp) for Delta = (b - c) / n on paired binary outcomes.

    Standard paired-difference variance: only the discordant pairs carry
    information, which is why concordant counts never enter.
    """
    d = (b - c) / float(n)
    var = (b + c - (b - c) ** 2 / float(n)) / float(n) ** 2
    se = math.sqrt(max(var, 0.0))
    return 100.0 * (d - z * se), 100.0 * (d + z * se), 100.0 * se


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p. Same implementation as bbeh/analyze.py."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n))


def resolve(subsets, selectivity):
    """Fill in derived CIs / p-values where the inputs allow it."""
    ses = []
    for s in subsets:
        s['se'] = None
        s['mcnemar'] = None
        has_counts = s.get('b') is not None and s.get('c') is not None
        if has_counts:
            # The consistency check runs whenever the counts exist, not only
            # when they are also the CI source. Counts that disagree with the
            # stated delta mean the block was assembled from two different
            # computations, which is worth failing on even if the drawn
            # interval happens to come from elsewhere.
            derived = 100.0 * (s['b'] - s['c']) / float(s['n'])
            if abs(derived - s['delta']) > 0.02:
                raise SystemExit(
                    'subset %r: delta=%+.2f pp but b=%d, c=%d, n=%d imply %+.2f pp. '
                    'The counts and the delta describe different data; fix the '
                    'data block rather than drawing a mismatched error bar.'
                    % (s['key'], s['delta'], s['b'], s['c'], s['n'], derived))
            s['mcnemar'] = mcnemar_exact(s['b'], s['c'])
            lo, hi, se = paired_ci(s['b'], s['c'], s['n'])
            s['se'] = se
            if s['ci'] is None:
                s['ci'] = (lo, hi)
        if s['se'] is not None:
            ses.append(s['se'])

    # Disjoint subsets, so the variances of the two deltas add.
    if selectivity.get('ci') is None and len(ses) == len(subsets) == 2:
        se_s = math.sqrt(ses[0] ** 2 + ses[1] ** 2)
        selectivity['ci'] = (selectivity['value'] - 1.96 * se_s,
                             selectivity['value'] + 1.96 * se_s)
    return subsets, selectivity


def build(width, height, font_size, show_direction_hints):
    subsets, sel = resolve([dict(s) for s in SUBSETS], dict(SELECTIVITY))

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'font.size': font_size,
        'axes.linewidth': 0.6,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
    })

    fig, ax = plt.subplots(figsize=(width, height))

    rows = list(range(len(subsets)))[::-1]        # first subset on top
    ypos = {s['key']: y for s, y in zip(subsets, rows)}

    # Symmetric limits so the two directions are visually comparable. This is a
    # provisional span: the value labels sit outside the data ends, so after the
    # first draw we measure them and widen if any overflows (see below). Picking
    # a fixed multiplier instead means re-tuning it every time the numbers
    # change — at 1.75x the longest bar's label collided with its y tick label,
    # at 2.5x the bars were swimming in dead space.
    reach = [abs(s['delta']) for s in subsets]
    reach += [abs(v) for s in subsets if s['ci'] for v in s['ci']]
    span = max(reach) * 1.45
    ax.set_xlim(-span, span)
    ax.set_ylim(-0.58, len(subsets) - 1 + 0.78)

    # ── chrome ───────────────────────────────────────────────────────
    ax.set_axisbelow(True)
    ax.grid(axis='x', color=C_GRID, linewidth=0.6, zorder=0)   # solid hairline
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(C_AXIS)
    ax.tick_params(axis='x', colors=C_MUTED, labelsize=font_size - 1,
                   length=3, width=0.6)
    ax.tick_params(axis='y', length=0)

    # The zero line is the reference the whole chart is read against, so it is
    # one step stronger than the grid — still a hairline, still solid.
    ax.axvline(0, color=C_AXIS, linewidth=1.0, zorder=2)

    # The per-subset McNemar p rides in the tick label rather than out beyond
    # the whisker. It is the longest string either label could carry, and the
    # axis is symmetric and auto-widened to fit whatever sits outside the data,
    # so leaving it out there forces the span wide enough to shrink both bars
    # into the middle fifth of the chart.
    ax.set_yticks(rows)
    tick_text = []
    for s in subsets:
        line = '%s\n(n = %d)' % (s['label'], s['n'])
        if s['mcnemar'] is not None:
            line += '\nMcNemar $p=%.3f$' % s['mcnemar']
        tick_text.append(line)
    ax.set_yticklabels(tick_text, fontsize=font_size, color=C_INK,
                       linespacing=1.35)
    ax.set_xlabel('Paired accuracy change vs. placebo (pp)',
                  color=C_INK, fontsize=font_size, labelpad=4)
    for lbl in ax.get_xticklabels():
        lbl.set_color(C_INK_2)

    # ── bars ─────────────────────────────────────────────────────────
    edge_labels = []
    for s in subsets:
        y = ypos[s['key']]
        colour = C_POS if s['delta'] >= 0 else C_NEG
        ax.barh(y, s['delta'], height=0.40, color=colour, zorder=4,
                edgecolor='white', linewidth=0.8)
        if s['ci']:
            lo, hi = s['ci']
            ax.errorbar(s['delta'], y, xerr=[[s['delta'] - lo], [hi - s['delta']]],
                        fmt='none', ecolor=C_INK_2, elinewidth=0.9,
                        capsize=2.6, capthick=0.9, zorder=6)

        # Value label just past the data end, on the bar's own side. The anchor
        # is the outermost drawn element on that side, not the bar tip: with an
        # error bar present the whisker reaches past the bar, and anchoring on
        # the tip puts the label underneath its own cap.
        out = 0.055 * span
        if s['delta'] >= 0:
            anchor = max(s['delta'], s['ci'][1]) if s['ci'] else s['delta']
        else:
            anchor = min(s['delta'], s['ci'][0]) if s['ci'] else s['delta']
        edge_labels.append(ax.text(
            anchor + (out if s['delta'] >= 0 else -out), y,
            '%+.2f pp' % s['delta'],
            ha='left' if s['delta'] >= 0 else 'right', va='center',
            fontsize=font_size - 0.6, color=colour, fontweight='bold',
            zorder=7))
        # (McNemar p is rendered in the y tick label; see set_yticklabels above.)

    # ── the interaction the figure exists to report ──────────────────
    # Two lines rather than one: the contrast and its test are separate facts,
    # and one line of mathtext at column width runs past the axis.
    line1 = ('$\\Delta_{\\mathrm{accept}}-\\Delta_{\\mathrm{reject}} = %+.2f$ pp'
             % sel['value'])
    if sel.get('ci'):
        line1 += '  [%+.2f, %+.2f]' % (sel['ci'][0], sel['ci'][1])
    line2 = 'permutation $%s = %.3f$' % (sel.get('p_label', 'p'), sel['p'])
    # Corner-anchored in AXES coordinates, not data coordinates, so the
    # auto-widening below cannot drag it off the corner.
    ax.text(0.995, 0.99, line1 + '\n' + line2, transform=ax.transAxes,
            ha='right', va='top', fontsize=font_size - 0.8, color=C_INK,
            linespacing=1.35, zorder=8)

    # ── which way is which ───────────────────────────────────────────
    if show_direction_hints:
        ax.text(0.005, 0.045, 'memory hurts', transform=ax.transAxes,
                ha='left', va='center', fontsize=font_size - 2.2,
                color=C_NEG, style='italic')
        ax.text(0.995, 0.045, 'memory helps', transform=ax.transAxes,
                ha='right', va='center', fontsize=font_size - 2.2,
                color=C_POS, style='italic')

    fig.tight_layout(pad=0.55)

    # ── widen until nothing overflows ────────────────────────────────
    # Measure the rendered value labels and grow the axis to fit them, rather
    # than hardcoding headroom. Growing shrinks each label's footprint in data
    # units, so this converges immediately; three passes is belt and braces.
    for _ in range(3):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        inv = ax.transData.inverted()
        need = max(abs(v) for v in ax.get_xlim()) * 0.0
        for t in edge_labels:
            bb = t.get_window_extent(renderer=renderer)
            x0 = inv.transform((bb.x0, 0))[0]
            x1 = inv.transform((bb.x1, 0))[0]
            need = max(need, abs(x0), abs(x1))
        limit = max(abs(v) for v in ax.get_xlim())
        if need <= limit * 0.97:
            break
        ax.set_xlim(-need * 1.05, need * 1.05)

    return fig, subsets, sel


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--outdir', default=root,
                    help='default: repo root, alongside puremem.pdf')
    ap.add_argument('--name', default='gate_counterfactual')
    ap.add_argument('--width', type=float, default=3.4,
                    help='inches; 3.4 = one WACV column, ~7.0 = full width')
    ap.add_argument('--height', type=float, default=2.2)
    ap.add_argument('--font-size', type=float, default=8.0)
    ap.add_argument('--no-direction-hints', action='store_true',
                    help='drop the "memory hurts / helps" axis-end cues')
    ap.add_argument('--dpi', type=int, default=400)
    args = ap.parse_args()

    fig, subsets, sel = build(args.width, args.height, args.font_size,
                              show_direction_hints=not args.no_direction_hints)

    os.makedirs(args.outdir, exist_ok=True)
    made = []
    for ext in ('pdf', 'png'):
        path = os.path.join(args.outdir, '%s.%s' % (args.name, ext))
        fig.savefig(path, dpi=args.dpi, bbox_inches='tight', pad_inches=0.02,
                    facecolor='white')
        made.append(path)
    plt.close(fig)

    for s in subsets:
        ci = ('  95%% CI [%+.2f, %+.2f]' % s['ci']) if s['ci'] else ''
        print('%-7s n=%-4d delta=%+.2f pp%s' % (s['label'], s['n'], s['delta'], ci))
    print('selectivity = %+.2f pp, p = %.4f' % (sel['value'], sel['p']))
    if not all(s['ci'] for s in subsets):
        print('\nNOTE: no confidence intervals were supplied, so the figure has no\n'
              '      error bars. Appendix A.5 states that paired bootstrap 95% CIs\n'
              '      were computed; add them by giving each subset its discordant\n'
              '      counts (b=..., c=...) or an explicit ci=(lo, hi) in SUBSETS.')
    for path in made:
        print('wrote %s' % path)


if __name__ == '__main__':
    main()
