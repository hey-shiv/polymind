from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def build_metrics_plot(cards: list[dict]) -> plt.Figure:
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    figure.patch.set_facecolor("#f6efe6")

    for axis in axes:
        axis.set_facecolor("#fffaf4")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    if not cards:
        for axis in axes:
            axis.axis("off")
        figure.text(
            0.5,
            0.5,
            "Metrics will appear after the first query.",
            ha="center",
            va="center",
            fontsize=12,
            color="#5c544a",
        )
        return figure

    labels = [card["display_name"].split()[0] for card in cards]
    colors = [card.get("accent_color", "#666666") for card in cards]
    word_counts = [card["metrics"]["word_count"] for card in cards]
    grounding_scores = [card["metrics"]["grounding_score"] for card in cards]

    axes[0].bar(labels, word_counts, color=colors)
    axes[0].set_title("Answer Length")
    axes[0].set_ylabel("Words")

    axes[1].bar(labels, grounding_scores, color=colors)
    axes[1].set_title("Grounding Score")
    axes[1].set_ylim(0, 1.05)

    return figure
