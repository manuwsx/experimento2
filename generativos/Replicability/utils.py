import os
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.transforms import blended_transform_factory
from matplotlib.ticker import MaxNLocator
try:
    from IPython.display import HTML, display
except ImportError:
    HTML = None

def load_all_data_from_json(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data

def load_values_from_json_without_normalization(file_path):
    with open(file_path, "r") as file:
        data_org = json.load(file)
    return list(data_org.values())

def load_values_from_json(file_path):
    with open(file_path, "r") as file:
        data_org = json.load(file)
    return [v for k, v in data_org.items() if "normalized" in k]

def generate_histogram_and_save_with_means(
    data,
    bins,
    alpha,
    title,
    class_labels,
    means,
    colors,
    output_path,
    mean_labels=None
):
    text_col = "#0b1f3a"
    axis_col = "#5f6f7a"
    grid_col = "#d7e0e8"

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],

        "font.size": 18,
        "axes.titlesize": 20,
        "axes.labelsize": 20,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 15,

        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.facecolor": "white",

        "text.color": text_col,
        "axes.labelcolor": text_col,
        "xtick.color": text_col,
        "ytick.color": text_col,

        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    if mean_labels is None:
        mean_labels = class_labels

    # =========================================================
    # Figure
    # =========================================================
    fig, ax = plt.subplots(figsize=(9.2, 5.6), facecolor="white")

    all_data = np.concatenate(data)
    bin_edges = np.histogram_bin_edges(all_data, bins=bins)

    # =========================================================
    # Histograms
    # =========================================================
    for i in range(len(data)):
        ax.hist(
            data[i],
            bins=bin_edges,
            alpha=alpha,
            color=colors[i],
            edgecolor="white",
            linewidth=0.35,
            histtype="stepfilled",
            label=class_labels[i],
            zorder=2
        )

    # =========================================================
    # Mean line
    # =========================================================
    ymax = ax.get_ylim()[1]

    # Vertical positions to avoid overlap
    base_positions = [0.92, 0.81, 0.70, 0.59, 0.48]

    for i in range(len(data)):
        y_pos = base_positions[i] if i < len(base_positions) else 0.92 - i * 0.10

        ax.axvline(
            means[i],
            color=colors[i],
            linestyle="--",
            linewidth=2.3,
            alpha=0.95,
            zorder=4
        )

        ax.text(
            means[i],
            ymax * y_pos,
            f"{mean_labels[i]} = {means[i]:.4f}",
            color=colors[i],
            fontsize=16,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.24",
                facecolor="white",
                edgecolor=colors[i],
                linewidth=0.8,
                alpha=0.96
            ),
            zorder=5
        )

    # =========================================================
    # Axis and titles
    # =========================================================
    ax.set_xlabel("NSCD", labelpad=10)
    ax.set_ylabel("Frequency", labelpad=10)

    ax.set_title(
        title,
        pad=14,
        loc="center",
        x=0.44
    )

    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.75,
        color=grid_col,
        alpha=0.70,
        zorder=0
    )

    ax.set_axisbelow(True)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(axis_col)
        ax.spines[spine].set_linewidth(1.0)

    fig.subplots_adjust(
        top=0.88,
        bottom=0.17,
        left=0.13,
        right=0.96
    )

    # =========================================================
    # Save
    # =========================================================
    if output_path.lower().endswith(".svg"):
        pdf_path = output_path[:-4] + ".pdf"

        plt.savefig(
            pdf_path,
            format="pdf",
            transparent=False,
            bbox_inches="tight",
            pad_inches=0.06
        )

        plt.savefig(
            output_path,
            format="svg",
            transparent=False,
            bbox_inches="tight",
            pad_inches=0.06
        )

        print(f"✅ Saved PDF: {pdf_path}")
        print(f"✅ Saved SVG: {output_path}")

    else:
        plt.savefig(
            output_path,
            transparent=False,
            bbox_inches="tight",
            pad_inches=0.06,
            dpi=400
        )

        print(f"✅ Saved figure: {output_path}")

    plt.show()
    plt.close()

def generate_heatmap_and_save(
    results,
    models,
    title,
    output_path,
    threshold_line=True,
    show_title=False,
    show_colorbar=True
):
    # =========================================================
    # Histogram with common bins
    # =========================================================
    all_values = np.concatenate([
        np.asarray(results[m], dtype=float) for m in models
    ])

    n_bins = 34
    bins = np.linspace(all_values.min(), all_values.max(), n_bins + 1)

    heatmap_data = []
    means = []

    for model in models:
        values = np.asarray(results[model], dtype=float)
        counts, _ = np.histogram(values, bins=bins)
        heatmap_data.append(counts)
        means.append(values.mean())

    heatmap_data = np.array(heatmap_data)
    means = np.array(means)

    threshold = means[0]

    # =========================================================
    # Style
    # =========================================================
    base_teal = "#0f2f33"
    panel_bg = "#f4f8f9"
    text_col = "#0b1f3a"
    axis_col = "#5f6f7a"
    line_col = "#1d4e89"

    mean_point_color = "#3f4a5a"
    mean_point_edge = "#ffffff"

    teal_cmap = LinearSegmentedColormap.from_list(
        "teal_custom",
        [
            panel_bg,
            "#c7dcdf",
            "#8fbfc4",
            "#4a9aa4",
            base_teal
        ]
    )

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],

        "font.size": 18,
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 15,

        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.facecolor": "white",

        "text.color": text_col,
        "axes.labelcolor": text_col,
        "xtick.color": text_col,
        "ytick.color": text_col,

        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    if show_colorbar:
        fig = plt.figure(figsize=(7.2, 4.85), facecolor="white")

        gs = fig.add_gridspec(
            1, 4,
            # heatmap | Mean CD | espacio mínimo | colorbar
            width_ratios=[53.0, 14, 0, 1.5],
            wspace=0.045
        )

        ax = fig.add_subplot(gs[0, 0])
        ax_mean = fig.add_subplot(gs[0, 1])

        ax_space = fig.add_subplot(gs[0, 2])
        cax = fig.add_subplot(gs[0, 3])

        ax_space.axis("off")

    else:
        fig = plt.figure(figsize=(5.65, 4.25), facecolor="white")

        gs = fig.add_gridspec(
            1, 2,
            width_ratios=[23.0, 4.6],
            wspace=0.08
        )

        ax = fig.add_subplot(gs[0, 0])
        ax_mean = fig.add_subplot(gs[0, 1])

        cax = None

    ax.set_facecolor(panel_bg)
    ax_mean.set_facecolor("white")

    # =========================================================
    # Normalization
    # =========================================================
    vmax = np.percentile(heatmap_data, 99.5)

    if vmax <= 0:
        vmax = heatmap_data.max() if heatmap_data.max() > 0 else 1

    norm = PowerNorm(
        gamma=0.9,
        vmin=0,
        vmax=vmax
    )

    # =========================================================
    # Heatmap
    # =========================================================
    x_edges = bins
    y_edges = np.arange(len(models) + 1)

    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        heatmap_data,
        cmap=teal_cmap,
        norm=norm,
        shading="flat",
        edgecolors="none"
    )

    ax.set_ylim(0, len(models))
    ax.invert_yaxis()

    y_centers = np.arange(len(models)) + 0.5
    ax.set_yticks(y_centers)
    ax.set_yticklabels(models, fontsize=15)

    if threshold_line:
        ax.axvline(
            threshold,
            color=line_col,
            linestyle=(0, (4, 3)),
            linewidth=1.8,
            alpha=0.95,
            zorder=3
        )

    ax.scatter(
        means,
        y_centers,
        s=48,
        color=mean_point_color,
        edgecolor=mean_point_edge,
        linewidth=1.0,
        zorder=5
    )

    for y in range(1, len(models)):
        ax.hlines(
            y,
            bins[0],
            bins[-1],
            color="#f7fbfc",
            linewidth=0.9,
            alpha=0.50,
            zorder=2
        )

    # =========================================================
    # Axis
    # =========================================================
    ax.set_xlabel(
        "NSCD",
        labelpad=9,
        fontsize=17
    )

    ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=15)

    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))

    if show_title:
        ax.set_title(
            title,
            pad=9,
            fontsize=17,
            color=text_col
        )

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(axis_col)
        ax.spines[spine].set_linewidth(0.9)

    # =========================================================
    # Mean CD column without texts
    # =========================================================
    ax_mean.set_xlim(0, 1)
    ax_mean.set_ylim(ax.get_ylim())

    ax_mean.set_xticks([])
    ax_mean.set_yticks([])
    ax_mean.tick_params(
        left=False,
        right=False,
        labelleft=False,
        labelright=False,
        bottom=False,
        top=False,
        labelbottom=False,
        labeltop=False
    )

    for spine in ax_mean.spines.values():
        spine.set_visible(False)

    ax_mean.set_facecolor("white")

    mean_transform = blended_transform_factory(ax_mean.transAxes, ax_mean.transData)

    dot_x = 0.12
    value_x = 0.25
    header_x = 0.50

    ax_mean.text(
        header_x,
        1.025,
        "Mean NSCD",
        transform=ax_mean.transAxes,
        ha="center",
        va="bottom",
        fontsize=16.5,
        fontweight="bold",
        color=text_col,
        clip_on=False
    )

    for y, mean_val in zip(y_centers, means):
        ax_mean.scatter(
            dot_x,
            y,
            s=38,
            color=mean_point_color,
            edgecolor=mean_point_edge,
            linewidth=0.85,
            transform=mean_transform,
            zorder=4
        )

        ax_mean.text(
            value_x,
            y,
            f"{mean_val:.4f}",
            ha="left",
            va="center",
            fontsize=15.5,
            family="monospace",
            color=text_col,
            transform=mean_transform
        )

    # =========================================================
    # Separated colorbar
    # =========================================================
    if show_colorbar:
        cbar = fig.colorbar(mesh, cax=cax)

        cbar.set_label(
            "Frequency",
            color=text_col,
            labelpad=9,
            fontsize=14
        )

        cbar.ax.tick_params(
            colors=text_col,
            labelsize=13,
            width=0.7,
            length=3
        )

        cbar.outline.set_linewidth(0.6)
        cbar.outline.set_edgecolor(axis_col)

    fig.subplots_adjust(
        top=0.91 if not show_title else 0.84,
        bottom=0.18,
        left=0.13,
        right=0.98
    )

    # =========================================================
    # Save
    # =========================================================
    if output_path.lower().endswith(".svg"):
        pdf_path = output_path[:-4] + ".pdf"

        plt.savefig(
            pdf_path,
            format="pdf",
            transparent=False,
            bbox_inches="tight",
            pad_inches=0.04
        )

        plt.savefig(
            output_path,
            format="svg",
            transparent=False,
            bbox_inches="tight",
            pad_inches=0.04
        )

        print(f"✅ Saved PDF: {pdf_path}")
        print(f"✅ Saved SVG: {output_path}")

    else:
        plt.savefig(
            output_path,
            transparent=False,
            bbox_inches="tight",
            pad_inches=0.04,
            dpi=400
        )

        print(f"✅ Saved figure: {output_path}")

    plt.show()
    plt.close()

def generate_multiline_plot_per_model(
    model_file_templates,
    eval_points,
    title,
    output_path,
    ylabel="Mean Chamfer Distance",
    xlabel="Training progress (%)",
    highlight_model=None,
    annotate_last=True
):
    text_col = "#0b1f3a"
    axis_col = "#5f6f7a"
    grid_col = "#d7e0e8"

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],

        "font.size": 22,
        "axes.titlesize": 20,
        "axes.labelsize": 22,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 17,

        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.facecolor": "white",

        "text.color": text_col,
        "axes.labelcolor": text_col,
        "xtick.color": text_col,
        "ytick.color": text_col,

        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    color_map = {
        "DPM":    "#3366cc",
        "PVD":    "#e69500",
        "DiT-3D": "#009E73",
        "LION":   "#8e44ad",
    }

    marker_map = {
        "DPM": "o",
        "PVD": "s",
        "DiT-3D": "D",
        "LION": "^",
    }

    linestyle_map = {
        "DPM": "-",
        "PVD": "--",
        "DiT-3D": "-.",
        "LION": ":",
    }

    eval_points = np.asarray(eval_points)

    # =========================================================
    # Load means
    # =========================================================
    model_means = {}

    for model_name, checkpoint_paths in model_file_templates.items():
        means = []

        for point in eval_points:
            point_key = int(point)

            if point_key not in checkpoint_paths:
                raise ValueError(f"Missing {point_key}% checkpoint for model '{model_name}'")

            file_path = checkpoint_paths[point_key]

            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            values = load_values_from_json(file_path)
            mean_val = np.mean(values)
            means.append(mean_val)

            #print(f"[{model_name}] {point_key}% -> mean = {mean_val:.6f}")

        model_means[model_name] = np.array(means, dtype=float)

    fig, ax = plt.subplots(figsize=(9.6, 6.0), facecolor="white")

    # =========================================================
    # Plot curves
    # =========================================================
    for model_name, y_values in model_means.items():
        is_highlight = model_name == highlight_model

        color = color_map.get(model_name, "#333333")
        marker = marker_map.get(model_name, "o")
        linestyle = linestyle_map.get(model_name, "-")

        ax.plot(
            eval_points,
            y_values,
            linestyle=linestyle,
            color=color,
            linewidth=3.2 if is_highlight else 2.5,
            alpha=0.98,
            zorder=3
        )

        ax.scatter(
            eval_points,
            y_values,
            s=92 if is_highlight else 72,
            marker=marker,
            color=color,
            edgecolor="white",
            linewidth=1.2,
            zorder=4
        )

        ax.scatter(
            eval_points,
            y_values,
            s=14 if is_highlight else 11,
            marker="o",
            color="white",
            edgecolor="none",
            zorder=5,
            alpha=0.85
        )

    all_y = np.concatenate(list(model_means.values()))
    y_min, y_max = np.min(all_y), np.max(all_y)
    y_range = y_max - y_min

    if y_range == 0:
        y_range = abs(y_max) * 0.1 if y_max != 0 else 1.0

    y_pad = y_range * 0.16
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    x_last = eval_points[-1]

    ax.axvspan(
        x_last - 3.5,
        x_last + 3.5,
        color="#66adff",
        alpha=0.32,
        zorder=0
    )

    ax.axvline(
        x_last,
        color="#0086c4",
        linewidth=1.4,
        alpha=0.75,
        zorder=1
    )

    ax.set_xlim(eval_points[0] - 4, eval_points[-1] + 14)

    if annotate_last:
        last_values = {
            model_name: y_values[-1]
            for model_name, y_values in model_means.items()
        }

        sorted_models = sorted(last_values.keys(), key=lambda m: last_values[m])

        min_gap = y_range * 0.075
        adjusted_y = {}

        previous_y = None
        for model_name in sorted_models:
            y = last_values[model_name]

            if previous_y is None:
                adjusted_y[model_name] = y
            else:
                adjusted_y[model_name] = max(y, previous_y + min_gap)

            previous_y = adjusted_y[model_name]

        upper_limit = y_max + y_pad * 0.80
        overflow = max(adjusted_y.values()) - upper_limit

        if overflow > 0:
            for model_name in adjusted_y:
                adjusted_y[model_name] -= overflow

        x_text = x_last + 1.8

        for model_name, y_values in model_means.items():
            color = color_map.get(model_name, "#333333")
            y_last = y_values[-1]
            y_text = adjusted_y[model_name]
            is_highlight = model_name == highlight_model

            if abs(y_text - y_last) > y_range * 0.015:
                ax.plot(
                    [x_last + 0.35, x_text - 0.35],
                    [y_last, y_text],
                    color=color,
                    linewidth=1.0,
                    alpha=0.65,
                    zorder=2
                )

            ax.text(
                x_text,
                y_text,
                model_name,
                color=color,
                fontsize=19,
                fontweight="bold" if is_highlight else "normal",
                va="center",
                ha="left",
                zorder=6
            )

    ax.set_xlabel(xlabel, labelpad=12)
    ax.set_ylabel(ylabel, labelpad=12)
    ax.set_xticks(eval_points)

    ax.grid(
        True,
        which="major",
        linestyle="--",
        linewidth=0.75,
        color=grid_col,
        alpha=0.90
    )

    ax.set_axisbelow(True)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(axis_col)
        ax.spines[spine].set_linewidth(1.0)

    fig.subplots_adjust(
        top=0.88,
        bottom=0.18,
        left=0.16,
        right=0.86
    )

    if output_path.lower().endswith(".svg"):
        pdf_path = output_path[:-4] + ".pdf"
        plt.savefig(pdf_path, format="pdf", bbox_inches="tight", pad_inches=0.06)
        print(f"✅ Saved PDF: {pdf_path}")

"""
Tables generation
"""

TABLE_STYLE = """
<style>
.paper-table {border-collapse: collapse; font-family: serif; font-size: 14px; margin: 12px 0;}
.paper-table th, .paper-table td {padding: 4px 10px; text-align: right; border-bottom: 1px solid #ddd;}
.paper-table th:first-child, .paper-table td:first-child {text-align: left; font-weight: 600;}
.paper-table thead tr:first-child th {border-bottom: 2px solid #222;}
.paper-table caption {caption-side: top; text-align: left; font-weight: 700; font-size: 16px; margin-bottom: 6px;}
</style>
"""

CATEGORIES = ["airplane", "car", "chair"]
CATEGORY_LABELS = {"airplane": "Airplane", "car": "Car", "chair": "Chair"}
MODELS = ["DPM", "PVD", "LION", "DiT-3D", "XCube", "SLIDE 3D", "SPVD-S", "SPVD-L"]

def fmt4(value):
    return f"{float(value):.4f}"

def fmt2(value):
    return f"{float(value):.2f}"

def render_table(title, header_groups, rows):
    html = [TABLE_STYLE, "<table class='paper-table'>", f"<caption>{title}</caption>"]
    html.append("<thead><tr><th rowspan='2'>Model</th>")
    for group, subheaders in header_groups:
        html.append(f"<th colspan='{len(subheaders)}'>{group}</th>")
    html.append("</tr><tr>")
    for _, subheaders in header_groups:
        for subheader in subheaders:
            html.append(f"<th>{subheader}</th>")
    html.append("</tr></thead><tbody>")
    for model, values in rows:
        html.append(f"<tr><td>{model}</td>")
        for value in values:
            html.append(f"<td>{value}</td>")
        html.append("</tr>")
    html.append("</tbody></table>")
    html = "".join(html)
    if HTML is not None:
        display(HTML(html))
    else:
        print(title)
        print(["Model"] + [f"{g} {s}" for g, ss in header_groups for s in ss])
        for row in rows:
            print(row)
    return {"title": title, "header_groups": header_groups, "rows": rows}