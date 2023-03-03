import matplotlib.pyplot as plt


def legend_grid(
    handles: list[plt.Line2D],
    labels: list[str],
    row_names: list[str],
    col_names: list[str],
    ax: plt.Axes,
    loc: str = "center",
) -> None:
    r = len(row_names)
    c = len(col_names)

    if (r * c) != len(handles):
        raise RuntimeError(
            "Names and number of legend handles need to match! "
            f"Got {r*c} names and {len(handles)} handles."
        )

    dummy_handles = [plt.plot([], marker="", ls="")[0]] * (r + c + 1)
    all_labels = [""] + row_names + col_names
    new_handles, new_labels = [], []
    for idx in range(c):
        new_handles.extend([dummy_handles[r + 1 + idx]] + handles[idx::c])
        new_labels.extend([all_labels[r + 1 + idx]] + labels[idx::c])
    lgd = ax.legend(
        dummy_handles[: r + 1] + new_handles,
        all_labels[: r + 1] + new_labels,
        loc=loc,
        ncol=c + 1,
        frameon=False,
        columnspacing=0,
    )
    for line in lgd.get_lines():
        line.set_linewidth(2.0)
    for text in lgd.get_texts():
        x, y = text.get_position()
        t = text.get_text()
        if t in col_names:
            text.set_fontweight("bold")
            text.set_position([x - 40, y])
        elif t in row_names:
            text.set_position([x - 30, y])
    ax.axis("off")
