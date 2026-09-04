#!/usr/bin/env python3
"""BBEH performance-vs-memory-size Pareto figure (replaces/supplements Table 1).

    python figure_script/plot_bbeh_pareto.py                 # -> ./bbeh_pareto.{pdf,png}
    python figure_script/plot_bbeh_pareto.py --width 7.0     # full-width (two-column)
    python figure_script/plot_bbeh_pareto.py --outdir latex

Design decisions worth knowing before editing:

**Why a single linear x-axis and not a log axis or a broken axis.** The three
size-matched banks sit at 26.4 / 26.6 / 27.4 % — within one percentage point of
each other. No horizontal transform separates them, because they are *designed*
to be the same size; a log axis or a 0-30 % zoom would spend the whole width
trying to pull apart points whose x-coordinates are supposed to coincide. They
separate cleanly on **y** instead (71.30 / 73.91 / 74.78), so the linear axis
renders them as one vertical triplet at "~27 % memory" — which is exactly what
the size-matched controls mean. The wide empty gap between 27 % and 100 % is not
wasted space either: it is the 3.8x memory cost of Full Memory, and the dashed
connector spans it.

**Why the dominated region is shaded.** Random Matched and Stratified Matched are
not merely "worse" — they are Pareto-dominated: PureMem uses *less* memory and is
*more* accurate than both. The shaded quadrant (x >= PureMem's size, y <= PureMem's
accuracy) makes that a geometric fact rather than something the reader has to
verify from the numbers.

**Color.** One categorical hue (blue #2a78d6) for PureMem; everything else is
neutral ink. This is the emphasis pattern — highlight one, gray the rest — which
is the right form when the story is one point rather than five series. Identity
never rests on hue alone: every point is direct-labeled and PureMem also differs
in marker shape and size, so the figure survives grayscale printing and CVD.
Checked with the palette validator on a white surface: blue vs muted gray is
Delta-E 15.9 (CVD) / 17.8 (normal), all marks >= 3:1 contrast.

**Gridlines are solid hairlines.** Dashing is reserved here for the one line that
carries meaning (the PureMem -> Full Memory trade-off); a dashed grid would
compete with it.
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ═════════════════════════════════════════════════════════════════════
#  Data — manuscript Table 1 (BBEH, 690 shared scorable instances)
#  Single source of truth for this figure. Update here if the runs change.
# ═════════════════════════════════════════════════════════════════════

#            label                 demos  chunks  rel_size%  correct  acc%
POINTS = [
    dict(key='no_memory',  label='No Memory',          demos=0,   chunks=0,
         size=0.0,   correct=502, acc=72.75),
    dict(key='random',     label='Random Matched',     demos=232, chunks=962,
         size=26.6,  correct=492, acc=71.30),
    dict(key='stratified', label='Stratified Matched', demos=226, chunks=990,
         size=27.4,  correct=510, acc=73.91),
    dict(key='puremem',    label='PureMem',            demos=200, chunks=955,
         size=26.4,  correct=516, acc=74.78),
    dict(key='full',       label='Full Memory',        demos=856, chunks=3616,
         size=100.0, correct=535, acc=77.54),
]
N_TOTAL = 690

# The Pareto frontier for (minimize memory, maximize accuracy). Computed in
# `pareto_frontier()` below and asserted against this, so a data edit that
# changes which points are dominated fails loudly instead of silently drawing
# the old story.
EXPECTED_FRONTIER = ['no_memory', 'puremem', 'full']

# ═════════════════════════════════════════════════════════════════════
#  Palette — dataviz reference instance, light mode on a white page
# ═════════════════════════════════════════════════════════════════════

C_PRIMARY = '#2a78d6'   # categorical slot 1 — PureMem only
C_INK = '#0b0b0b'       # primary ink
C_INK_2 = '#52514e'     # secondary ink — Full Memory (the reference ceiling)
C_MUTED = '#898781'     # muted — baselines and axis labels
C_GRID = '#e1e0d9'      # hairline gridline
C_AXIS = '#c3c2b7'      # baseline / axis rule
C_SHADE = '#eceae2'     # dominated-region wash


def pareto_frontier(points):
    """Keys on the (min size, max accuracy) frontier, ordered by size.

    A point is on the frontier when nothing else is at least as small AND at
    least as accurate, with at least one of the two strict.
    """
    keep = []
    for p in points:
        dominated = any(
            q is not p and q['size'] <= p['size'] and q['acc'] >= p['acc']
            and (q['size'] < p['size'] or q['acc'] > p['acc'])
            for q in points
        )
        if not dominated:
            keep.append(p)
    return [p['key'] for p in sorted(keep, key=lambda r: r['size'])]


def build(width, height, font_size, annotate_values, shade):
    by_key = {p['key']: p for p in POINTS}
    frontier = pareto_frontier(POINTS)
    assert frontier == EXPECTED_FRONTIER, (
        'the Pareto frontier changed: got %s, expected %s. The figure\'s '
        'annotations describe the expected frontier — update both together.'
        % (frontier, EXPECTED_FRONTIER))

    pm, full, nomem = by_key['puremem'], by_key['full'], by_key['no_memory']

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'font.size': font_size,
        'axes.linewidth': 0.6,
        'pdf.fonttype': 42,      # embed TrueType, not Type 3 — required by most
        'ps.fonttype': 42,       # camera-ready checkers
        'svg.fonttype': 'none',
    })

    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(-7, 114)
    ax.set_ylim(69.6, 78.9)

    # ── chrome ───────────────────────────────────────────────────────
    ax.set_axisbelow(True)
    ax.grid(axis='y', color=C_GRID, linewidth=0.6, zorder=0)   # solid hairline
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(C_AXIS)
    ax.tick_params(colors=C_MUTED, labelsize=font_size - 1, length=3, width=0.6)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(C_INK_2)

    ax.set_xlabel('Relative memory size (% of Full Memory)',
                  color=C_INK, fontsize=font_size, labelpad=4)
    ax.set_ylabel('Accuracy (%)', color=C_INK, fontsize=font_size, labelpad=4)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_yticks([70, 72, 74, 76, 78])

    # ── dominated quadrant ───────────────────────────────────────────
    if shade:
        ax.add_patch(plt.Rectangle(
            (pm['size'], ax.get_ylim()[0]),
            ax.get_xlim()[1] - pm['size'], pm['acc'] - ax.get_ylim()[0],
            facecolor=C_SHADE, edgecolor='none', alpha=0.55, zorder=1))
        ax.text(97, 70.25, 'dominated by PureMem', ha='right', va='bottom',
                fontsize=font_size - 2.2, color=C_MUTED, style='italic', zorder=2)

    # ── frontier: solid up to PureMem, dashed across the gap ─────────
    ax.plot([nomem['size'], pm['size']], [nomem['acc'], pm['acc']],
            color=C_PRIMARY, linewidth=1.1, alpha=0.45, zorder=3,
            solid_capstyle='round')
    ax.plot([pm['size'], full['size']], [pm['acc'], full['acc']],
            color=C_PRIMARY, linewidth=1.1, alpha=0.55, zorder=3,
            linestyle=(0, (4, 2.6)), dash_capstyle='round')

    # Each frontier segment states what it buys, so the two gains read as a
    # pair: what the curated bank wins, then what 3.8x more memory adds on top.
    ratio = full['chunks'] / pm['chunks']
    ax.text(10.5, 74.25, '+%.2f pp' % (pm['acc'] - nomem['acc']),
            ha='center', va='bottom', fontsize=font_size - 1.6,
            color=C_PRIMARY, zorder=6)
    ax.text(61, 76.55, '+%.2f pp\nfor %.1f$\\times$ memory'
            % (full['acc'] - pm['acc'], ratio),
            ha='center', va='bottom', fontsize=font_size - 1.6,
            color=C_PRIMARY, linespacing=1.25, zorder=6)

    # ── marks ────────────────────────────────────────────────────────
    style = {
        'no_memory':  dict(marker='o', s=34, c=C_MUTED,  zorder=4),
        'random':     dict(marker='o', s=34, c=C_MUTED,  zorder=4),
        'stratified': dict(marker='o', s=34, c=C_MUTED,  zorder=4),
        'full':       dict(marker='D', s=34, c=C_INK_2,  zorder=5),
        'puremem':    dict(marker='*', s=225, c=C_PRIMARY, zorder=7),
    }
    for p in POINTS:
        st = dict(style[p['key']])
        # 2px surface ring rather than an outline drawn to separate marks.
        ax.scatter([p['size']], [p['acc']], linewidths=0.9,
                   edgecolors='white', **st)

    # ── direct labels ────────────────────────────────────────────────
    # Every point is named, so identity never rests on color. Placement is
    # explicit because the three size-matched banks share an x-coordinate.
    place = {
        'no_memory':  dict(dx=2.6,  dy=-0.14, ha='left',  va='top'),
        'random':     dict(dx=3.0,  dy=-0.12, ha='left',  va='top'),
        # Below-right, not above-right: above-right runs into the star.
        'stratified': dict(dx=3.2,  dy=-0.10, ha='left',  va='top'),
        'puremem':    dict(dx=-2.6, dy=0.30,  ha='right', va='bottom'),
        'full':       dict(dx=-3.4, dy=0.20,  ha='right', va='bottom'),
    }
    for p in POINTS:
        q = place[p['key']]
        hero = p['key'] == 'puremem'
        text = p['label']
        if annotate_values:
            text += '\n%.2f%%' % p['acc']
        ax.text(p['size'] + q['dx'], p['acc'] + q['dy'], text,
                ha=q['ha'], va=q['va'],
                fontsize=font_size - (0.6 if hero else 1.6),
                color=C_PRIMARY if hero else C_INK_2,
                fontweight='bold' if hero else 'normal',
                linespacing=1.2, zorder=8)

    fig.tight_layout(pad=0.55)
    return fig


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--outdir', default=root,
                    help='default: repo root, so \\includegraphics{bbeh_pareto.pdf} '
                         'resolves the same way puremem.pdf already does')
    ap.add_argument('--name', default='bbeh_pareto')
    ap.add_argument('--width', type=float, default=3.4,
                    help='inches; 3.4 = one WACV column, ~7.0 = full width')
    ap.add_argument('--height', type=float, default=2.75)
    ap.add_argument('--font-size', type=float, default=8.0)
    ap.add_argument('--no-values', action='store_true',
                    help='label method names only, without the accuracy values')
    ap.add_argument('--no-shade', action='store_true',
                    help='drop the dominated-region wash')
    ap.add_argument('--dpi', type=int, default=400)
    args = ap.parse_args()

    fig = build(args.width, args.height, args.font_size,
                annotate_values=not args.no_values, shade=not args.no_shade)

    os.makedirs(args.outdir, exist_ok=True)
    made = []
    for ext in ('pdf', 'png'):
        path = os.path.join(args.outdir, '%s.%s' % (args.name, ext))
        fig.savefig(path, dpi=args.dpi, bbox_inches='tight', pad_inches=0.02,
                    facecolor='white')
        made.append(path)
    plt.close(fig)

    print('Pareto frontier: %s' % ' -> '.join(pareto_frontier(POINTS)))
    print('dominated      : %s' % ', '.join(
        p['label'] for p in POINTS if p['key'] not in pareto_frontier(POINTS)))
    for path in made:
        print('wrote %s' % path)


if __name__ == '__main__':
    main()
