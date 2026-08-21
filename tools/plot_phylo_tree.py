"""
python -m tools.plot_phylo_tree

Renders a circular phylogeny diagram for each tree in TREE_PATHS into tools/plots/
(named <dataset>_<stem>.png); trees not on disk are skipped with a message.

For lepid trees, non-Nymphalidae species are collapsed to one representative
tip per genus (Nymphalidae tips are all kept).
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from utils.utils import load_pickle, DATASET_ALIAS2NAME


TREE_PATHS = [
    Path("metadata/bryo/tree_prepoly.pkl"),
    Path("metadata/cub/tree_prepoly.pkl"),
    Path("metadata/lepid/tree_prepoly.pkl"),
]

DPATH_OUT = Path("tools/plots")


def induced_subtree(clade, keep):
    """
    Reduce the clade in place to the subtree induced by the tips in `keep`,
    compressing unary chains by merging branch lengths (kept tips retain their
    root-to-tip depths). Returns None if no kept tip lies under the clade.
    """
    if clade.is_terminal():
        return clade if clade.name in keep else None
    children = [sub for sub in (induced_subtree(child, keep) for child in clade.clades) if sub is not None]
    if not children:
        return None
    if len(children) == 1:
        child = children[0]
        child.branch_length = (child.branch_length or 0.0) + (clade.branch_length or 0.0)
        return child
    clade.clades = children
    return clade

def collapse_non_nymphalidae_genera(tree, class_data):
    """
    Keep all Nymphalidae tips; collapse every other genus to a single
    representative tip (first cid alphabetically).
    """
    keep = set()
    rep = {}
    for tip in tree.get_terminals():
        if class_data[tip.name]["family"] == "nymphalidae":
            keep.add(tip.name)
        else:
            genus = class_data[tip.name]["genus"]
            if genus not in rep or tip.name < rep[genus]:
                rep[genus] = tip.name
    keep |= set(rep.values())
    tree.root = induced_subtree(tree.root, keep)
    return tree

def layout(tree):
    """
    Circular layout: tips evenly spaced around the circle, internal nodes at the mean
    angle of their children, radius = root-to-node depth (sum of branch lengths).
    Returns (angle, radius) dicts keyed by clade.
    """
    tips = tree.get_terminals()
    angle = {tip: 2.0 * np.pi * i / len(tips) for i, tip in enumerate(tips)}

    radius = {tree.root: 0.0}
    postorder = []
    stack = [tree.root]
    while stack:
        node = stack.pop()
        postorder.append(node)
        for child in node.clades:
            radius[child] = radius[node] + child.branch_length
            stack.append(child)

    for node in reversed(postorder):
        if node.clades:
            angle[node] = np.mean([angle[child] for child in node.clades])

    return angle, radius

def build_segments(tree, angle, radius):
    segments = []
    stack = [tree.root]
    while stack:
        node = stack.pop()
        if not node.clades:
            continue
        # arc at the parent's radius spanning its children's angles
        angles_children = [angle[child] for child in node.clades]
        arc = np.linspace(min(angles_children), max(angles_children), 64)
        segments.append(np.column_stack([radius[node] * np.cos(arc), radius[node] * np.sin(arc)]))
        # radial segment out to each child
        for child in node.clades:
            a = angle[child]
            segments.append(np.array([
                [radius[node] * np.cos(a), radius[node] * np.sin(a)],
                [radius[child] * np.cos(a), radius[child] * np.sin(a)],
            ]))
            stack.append(child)
    return segments

def plot_tree(fpath_tree: Path, fpath_out: Path) -> None:
    tree = load_pickle(fpath_tree)
    if fpath_tree.parent.name == "lepid":
        class_data = load_pickle(fpath_tree.parent / "class_data.pkl")
        tree = collapse_non_nymphalidae_genera(tree, class_data)
    angle, radius = layout(tree)
    segments = build_segments(tree, angle, radius)

    # scale figure size up and line width down with tip count so dense trees stay legible
    n_tips = tree.count_terminals()
    size = float(np.clip(n_tips / 300, 10, 30))
    lw = float(np.clip(600 / n_tips, 0.15, 0.7))

    fig, ax = plt.subplots(figsize=(size, size))
    ax.add_collection(LineCollection(segments, colors="black", linewidths=lw))
    ax.set_aspect("equal")
    ax.autoscale()
    ax.axis("off")
    # title centered on the root node (origin), above the tree
    ymin, ymax = ax.get_ylim()
    ax.text(0.0, ymax + 0.03 * (ymax - ymin), DATASET_ALIAS2NAME[fpath_tree.parent.name],
            ha="center", va="bottom", fontsize=2 * size)
    fig.savefig(fpath_out, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {fpath_out}")

def main() -> None:
    DPATH_OUT.mkdir(exist_ok=True)
    for fpath_tree in TREE_PATHS:
        if not fpath_tree.is_file():
            print(f"skipping {fpath_tree} (not found)")
            continue
        plot_tree(fpath_tree, DPATH_OUT / f"{fpath_tree.parent.name}_{fpath_tree.stem}.png")


if __name__ == "__main__":
    main()
