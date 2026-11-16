import ipywidgets as widgets
from IPython.display import display, clear_output
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def make_injection_plotter(data, var_labels):
    """Plot numeric variables with optional overlays for cluster populations."""
    numeric_columns = data.select_dtypes(include="number").columns
    columns = [column for column in var_labels if column in numeric_columns]
    if not columns:
        raise ValueError("No dictionary keys match numeric injection columns.")

    # Mathtext renders the expressions inside $...$. Remove the outer \rm
    # prefix so it is not displayed as literal text.
    labels = {
        column: var_labels[column].removeprefix(r"\rm ").strip()
        for column in columns
    }

    # Normalize the catalog labels without modifying the input DataFrame.
    # Both the literal "none" and missing/blank values represent isolated systems.
    cluster_values = None
    if "cluster" in data.columns:
        cluster_values = (
            data["cluster"].astype("string").str.strip().str.lower()).to_numpy()
    cluster_colors = { 
        "globular": "#ff7f0e",
        "young": "#1f77b4",
        "nuclear": "#2ca02c",
        "isolated": "#9467bd",
    }
    cluster_overlays = {
        cluster: widgets.Checkbox(
            value=False, description=cluster.title(), indent=False,
            disabled=cluster_values is None, layout=widgets.Layout(width="auto")
        )
        for cluster in cluster_colors
    }

    default_x = "m1_source" if "m1_source" in columns else columns[0]
    default_y = "m2_source" if "m2_source" in columns else columns[min(1, len(columns) - 1)]
    plot_type = widgets.ToggleButtons(
        options=["Histogram", "Scatter"], description="Plot type:"
    )
    x_variable = widgets.Dropdown(
        options=columns, value=default_x, description="Variable:"
    )
    y_variable = widgets.Dropdown(
        options=columns, value=default_y, description="Y:"
    )
    x_scale = widgets.ToggleButtons(
        options=[("Linear", "linear"), ("Log", "log")],
        value="linear", description="X scale:"
    )
    y_scale = widgets.ToggleButtons(
        options=[("Linear", "linear"), ("Log", "log")],
        value="linear", description="Y scale:"
    )
    bins = widgets.IntSlider(
        value=40, min=5, max=150, step=5,
        description="Bins:", continuous_update=False
    )
    density = widgets.Checkbox(
    value=True, description="Density", indent=False
    )
    plot_button = widgets.Button(
        description="Plot", button_style="primary", icon="bar-chart"
    )
    output = widgets.Output()

    def update_controls(_=None):
        histogram = plot_type.value == "Histogram"
        x_variable.description = "Variable:" if histogram else "X:"
        y_variable.layout.display = "none" if histogram else ""
        bins.layout.display = "" if histogram else "none"
        density.layout.display = "" if histogram else "none"

    def redraw(_=None):
        histogram = plot_type.value == "Histogram"
        selected_clusters = [
            cluster for cluster, control in cluster_overlays.items()
            if control.value and cluster_values is not None
        ]
        overlay_counts = {}
        with output:
            clear_output(wait=True)
            x = data[x_variable.value].to_numpy(dtype=float, na_value=np.nan)
            valid = np.isfinite(x)
            if not histogram:
                y = data[y_variable.value].to_numpy(dtype=float, na_value=np.nan)
                # Use the same rows on both axes.
                valid &= np.isfinite(y)

            finite_count = int(valid.sum())
            if x_scale.value == "log":
                valid &= x > 0
            if not histogram and y_scale.value == "log":
                valid &= y > 0
            omitted_for_log = finite_count - int(valid.sum())
            if omitted_for_log:
                print(
                    f"Omitted {omitted_for_log:,} injections with nonpositive "
                    "values on a logarithmic axis."
                )

            if not valid.any():
                print("No values can be plotted with the selected variables and scales.")
                return

            fig, ax = plt.subplots(figsize=(10, 6))
            try:
                base_label = rf"\rm All injections ({valid.sum():,})" if selected_clusters else None
                base_color = "0.65" if selected_clusters else None
                if histogram:
                    # Reuse the full selection's bin edges for every cluster overlay.
                    hist_bins = np.histogram_bin_edges(x[valid], bins=bins.value)
                    if x_scale.value == "log":
                        # Equal-width bins on a logarithmic X axis.
                        lower, upper = x[valid].min(), x[valid].max()
                        if lower == upper:
                            lower, upper = lower / 1.1, upper * 1.1
                        hist_bins = np.geomspace(lower, upper, bins.value + 1)
                    ax.hist(
                        x[valid], bins=hist_bins, density=density.value,
                        alpha=0.6 if selected_clusters else 0.8,
                        color=base_color, label=base_label,
                        edgecolor="black", linewidth=0.5,
                        log=(y_scale.value == "log")
                    )
                    ax.set_ylabel(
                        r"\rm Probability density" if density.value else r"\rm Number of injections",
                        fontsize=16
                    )
                else:
                    ax.scatter(x[valid], y[valid], s=12,
                               alpha=0.25 if selected_clusters else 0.5,
                               color=base_color, label=base_label,
                               edgecolors="none", rasterized=True)
                    ax.set_ylabel(labels[y_variable.value], fontsize=16)

                for cluster in selected_clusters:
                    # Keep the same finite/log-axis filtering as the current plot.
                    overlay_valid = valid & (cluster_values == cluster)
                    count = int(overlay_valid.sum())
                    overlay_counts[cluster] = count
                    if not count:
                        continue
                    label = rf"\rm {cluster.title()} ({count:,})"
                    if histogram:
                        ax.hist(
                        x[overlay_valid], bins=hist_bins, density=density.value,
                        histtype="step", color=cluster_colors[cluster],
                        linewidth=2, label=label,
                        log=(y_scale.value == "log"), zorder=3
                    )
                    else:
                        ax.scatter(x[overlay_valid], y[overlay_valid], s=18, alpha=0.8,
                                   color=cluster_colors[cluster], label=label,
                                   edgecolors="none", rasterized=True, zorder=3)

                ax.set_xlabel(labels[x_variable.value], fontsize=16)
                ax.set_xscale(x_scale.value)
                ax.set_yscale(y_scale.value)
                ax.grid(alpha=0.2)
                if selected_clusters:
                    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                              borderaxespad=0, fontsize=14)
                fig.tight_layout()
                display(fig)
            finally:
                # Avoid duplicate output and accumulating open figures.
                plt.close(fig)

            print(f"Plotted {valid.sum():,} of {len(data):,} injections.")
            if overlay_counts:
                print("Cluster overlays: " + ", ".join(
                    f"{cluster.title()}: {count:,}"
                    for cluster, count in overlay_counts.items()
                ))

    # Wait for the button so large scatter plots redraw only when requested.
    plot_type.observe(update_controls, names="value")
    plot_button.on_click(redraw)

    update_controls()
    cluster_note = (
        "Cluster overlays (none/missing = isolated):"
        if cluster_values is not None else "Cluster overlays require a 'cluster' column."
    )
    panel = widgets.VBox([
        widgets.Label("Choose the plot type, variables, scales and overlays, then click Plot."),
        plot_type,
        widgets.HBox([x_variable, y_variable]),
        widgets.HBox([x_scale, y_scale]),
        widgets.Label(cluster_note),
        widgets.HBox(list(cluster_overlays.values())),
        widgets.HBox([bins, density, plot_button]),
        output,
    ])
    display(panel)
    redraw()
    return panel





def _normalise_et_band(values):
    """Accept Boolean flags or their True/False and 1/0 representations."""
    text = values.astype("string").str.strip().str.lower()
    true_values = ["true", "1", "1.0"]
    false_values = ["false", "0", "0.0", ""]
    recognised = text.isna() | text.isin(true_values + false_values)
    if not recognised.all():
        unexpected = text[~recognised].unique().tolist()
        raise ValueError(
            "ET in-band flags must be True/False or 1/0; "
            f"unexpected values: {unexpected[:5]}"
        )
    return text.isin(true_values).to_numpy(dtype=bool)


def _et_selection_mask(n_rows, in_band, snr, enabled, minimum_snr):
    """Return the shared ET/SNR selection for every plotted population."""
    if not enabled:
        return np.ones(n_rows, dtype=bool)
    if in_band is None:
        raise ValueError("The ET in-band column is required for this filter.")
    selected = in_band.copy()
    if snr is not None:
        selected &= np.isfinite(snr) & (snr >= minimum_snr)
    return selected


def make_injection_plotter_detection(
    data, var_labels, *, in_band_column="ET_in_band", snr_column="SNR"
):
    """Display a histogram/scatter widget and return its VBox panel.

    Check "ET in band only" to highlight events and enable the minimum-SNR
    slider. The cutoff is inclusive: SNR >= the slider value. All events
    remain in gray behind the filtered selection. Events without a finite
    SNR are excluded from the selection whenever the SNR filter is active.
    Changing controls does not redraw until Plot is clicked.

    Density is enabled by default for histograms. Cluster overlays use the
    same filtered selection and axes filtering as the colored foreground.
    All histograms use bin edges from the full gray background; each
    histogram is normalised independently when density is enabled.
    The input DataFrame is not modified.
    """

    numeric_columns = data.select_dtypes(include="number").columns
    columns = [column for column in var_labels if column in numeric_columns]
    if not columns:
        raise ValueError("No dictionary keys match numeric injection columns.")

    labels = {
        column: str(var_labels[column]).removeprefix(r"\rm ").strip()
        for column in columns
    }

    cluster_values = None
    if "cluster" in data.columns:
        cluster_values = (
            data["cluster"].astype("string").str.strip().str.lower()
            .fillna("isolated").replace({"none": "isolated", "": "isolated"})
            .to_numpy(dtype=str)
        )
    cluster_colors = {
        r"globular": "#ff7f0e",
        r"young": "#1f77b4",
        r"nuclear": "#2ca02c",
        r"isolated": "#9467bd",
    }
    cluster_overlays = {
        cluster: widgets.Checkbox(
            value=False, description=cluster.title(), indent=False,
            disabled=cluster_values is None, layout=widgets.Layout(width="auto")
        )
        for cluster in cluster_colors
    }

    in_band_values = (
        _normalise_et_band(data[in_band_column])
        if in_band_column in data.columns else None
    )
    snr_values = (
        pd.to_numeric(data[snr_column], errors="coerce")
        .to_numpy(dtype=float, na_value=np.nan)
        if snr_column in data.columns else None
    )
    snr_max = 1.0
    if in_band_values is not None and snr_values is not None:
        finite_snrs = snr_values[in_band_values & np.isfinite(snr_values)]
        if finite_snrs.size:
            snr_max = max(1.0, float(np.ceil(finite_snrs.max())))

    in_band_only = widgets.Checkbox(
        value=False, description="ET in band only", indent=False,
        disabled=in_band_values is None, layout=widgets.Layout(width="auto")
    )
    minimum_snr = widgets.FloatSlider(
        value=0.0, min=0.0, max=50, step=1,
        description="Min SNR:", continuous_update=False,
        readout_format=".1f", disabled=True,
        layout=widgets.Layout(width="450px")
    )

    default_x = next(
        (name for name in ("m1_source", "mass_1_source") if name in columns),
        columns[0]
    )
    default_y = next(
        (name for name in ("m2_source", "mass_2_source") if name in columns),
        columns[min(1, len(columns) - 1)]
    )
    plot_type = widgets.ToggleButtons(
        options=["Histogram", "Scatter"], description="Plot type:"
    )
    x_variable = widgets.Dropdown(
        options=columns, value=default_x, description="Variable:"
    )
    y_variable = widgets.Dropdown(
        options=columns, value=default_y, description="Y:"
    )
    x_scale = widgets.ToggleButtons(
        options=[("Linear", "linear"), ("Log", "log")],
        value="linear", description="X scale:"
    )
    y_scale = widgets.ToggleButtons(
        options=[("Linear", "linear"), ("Log", "log")],
        value="linear", description="Y scale:"
    )
    bins = widgets.IntSlider(
        value=40, min=5, max=150, step=5,
        description="Bins:", continuous_update=False
    )
    density = widgets.Checkbox(
        value=True, description="Density", indent=False,
        layout=widgets.Layout(width="auto")
    )
    plot_button = widgets.Button(
        description="Plot", button_style="primary", icon="bar-chart"
    )
    output = widgets.Output()

    def update_controls(_=None):
        histogram = plot_type.value == "Histogram"
        x_variable.description = "Variable:" if histogram else "X:"
        y_variable.layout.display = "none" if histogram else ""
        bins.layout.display = "" if histogram else "none"
        density.layout.display = "" if histogram else "none"
        minimum_snr.disabled = not (
            in_band_only.value and snr_values is not None
        )

    def redraw(_=None):
        histogram = plot_type.value == "Histogram"
        selected_clusters = [
            cluster for cluster, control in cluster_overlays.items()
            if control.value and cluster_values is not None
        ]
        overlay_counts = {}
        with output:
            clear_output(wait=True)
            event_selection = _et_selection_mask(
                len(data), in_band_values, snr_values,
                in_band_only.value, minimum_snr.value
            )
            if in_band_only.value:
                selection_description = r"ET in-band"
                if snr_values is not None:
                    selection_description += f", SNR $\ge$ {minimum_snr.value:g}"
                    missing_snr_count = int((
                        in_band_values & ~np.isfinite(snr_values)
                    ).sum())
                    if missing_snr_count:
                        print(
                            f"Excluded {missing_snr_count:,} in-band events "
                            "without a finite SNR from the colored selection."
                        )
                print(
                    f"Selection: {selection_description} "
                    f"({event_selection.sum():,} of {len(data):,} injections)."
                )

            x = data[x_variable.value].to_numpy(dtype=float, na_value=np.nan)
            # The background depends only on the chosen axes, never on ET/SNR.
            background_valid = np.isfinite(x)
            if not histogram:
                y = data[y_variable.value].to_numpy(dtype=float, na_value=np.nan)
                background_valid &= np.isfinite(y)

            finite_count = int(background_valid.sum())
            if x_scale.value == "log":
                background_valid &= x > 0
            if not histogram and y_scale.value == "log":
                background_valid &= y > 0
            valid = background_valid & event_selection
            omitted_for_log = finite_count - int(background_valid.sum())
            if omitted_for_log:
                print(
                    f"Omitted {omitted_for_log:,} injections with nonpositive "
                    "values on a logarithmic axis."
                )
            if not background_valid.any():
                print("No events can be plotted with these variables and scales.")
                return
            if in_band_only.value and not valid.any():
                print("No events pass the ET/SNR filters on these axes; showing the gray background.")

            fig, ax = plt.subplots(figsize=(10, 6))
            try:
                show_background = in_band_only.value or bool(selected_clusters)
                base_label = (
                    rf"\rm All injections ({background_valid.sum():,})"
                    if show_background else None
                )
                base_color = "0.65" if show_background else None
                if histogram:
                    hist_bins = np.histogram_bin_edges(x[background_valid], bins=bins.value)
                    if x_scale.value == "log":
                        lower, upper = x[background_valid].min(), x[background_valid].max()
                        if lower == upper:
                            lower, upper = lower / 1.1, upper * 1.1
                        hist_bins = np.geomspace(lower, upper, bins.value + 1)
                    ax.hist(
                        x[background_valid], bins=hist_bins, density=density.value,
                        alpha=0.45 if in_band_only.value else (0.6 if selected_clusters else 0.8),
                        color=base_color, label=base_label,
                        edgecolor="black", linewidth=0.5,
                        log=(y_scale.value == "log"), zorder=1
                    )
                    ax.set_ylabel(
                        r"\rm Probability density" if density.value else r"\rm Number of injections",
                        fontsize=16
                    )
                else:
                    ax.scatter(
                        x[background_valid], y[background_valid], s=12,
                        alpha=0.2 if in_band_only.value else (0.25 if selected_clusters else 0.5),
                        color=base_color, label=base_label,
                        edgecolors="none", rasterized=True, zorder=1
                    )
                    ax.set_ylabel(labels[y_variable.value], fontsize=16)

                # Draw the entire ET/SNR selection even when no clusters are checked.
                if in_band_only.value and valid.any():
                    filtered_label = f"{selection_description} ({valid.sum():,})"
                    if histogram:
                        ax.hist(
                            x[valid], bins=hist_bins, density=density.value,
                            histtype="step", color="#d62728", linewidth=2,
                            label=filtered_label, log=(y_scale.value == "log"),
                            zorder=2
                        )
                    else:
                        ax.scatter(
                            x[valid], y[valid], s=16,
                            alpha=0.4 if selected_clusters else 0.8,
                            color="#d62728", label=filtered_label,
                            edgecolors="none", rasterized=True, zorder=2
                        )

                for cluster in selected_clusters:
                    overlay_valid = valid & (cluster_values == cluster)
                    count = int(overlay_valid.sum())
                    overlay_counts[cluster] = count
                    if not count:
                        continue
                    label = f"{cluster.title()} ({count:,})"
                    if histogram:
                        ax.hist(
                            x[overlay_valid], bins=hist_bins, density=density.value,
                            histtype="step", color=cluster_colors[cluster],
                            linewidth=2, label=label,
                            log=(y_scale.value == "log"), zorder=3
                        )
                    else:
                        ax.scatter(
                            x[overlay_valid], y[overlay_valid], s=18, alpha=0.8,
                            color=cluster_colors[cluster], label=label,
                            edgecolors="none", rasterized=True, zorder=3
                        )

                ax.set_xlabel(labels[x_variable.value], fontsize=16)
                ax.set_xscale(x_scale.value)
                ax.set_yscale(y_scale.value)
                ax.grid(alpha=0.2)
                if show_background:
                    ax.legend(
                        loc="center left", bbox_to_anchor=(1.02, 0.5),
                        borderaxespad=0, fontsize=14
                    )
                fig.tight_layout()
                display(fig)
            finally:
                plt.close(fig)

            if in_band_only.value:
                print(f"Gray background: {background_valid.sum():,} of {len(data):,} injections.")
                print(f"Colored ET/SNR selection: {valid.sum():,} injections.")
            else:
                print(f"Plotted {background_valid.sum():,} of {len(data):,} injections.")
            if overlay_counts:
                print("Cluster overlays: " + ", ".join(
                    f"{cluster.title()}: {count:,}"
                    for cluster, count in overlay_counts.items()
                ))

    plot_type.observe(update_controls, names="value")
    in_band_only.observe(update_controls, names="value")
    plot_button.on_click(redraw)
    update_controls()

    cluster_note = (
        "Cluster overlays (none/missing = isolated):"
        if cluster_values is not None
        else "Cluster overlays require a 'cluster' column."
    )
    if in_band_values is None:
        et_note = (
            f"'{in_band_column}' is missing; add the ET result columns "
            "to the plotting DataFrame."
        )
    elif snr_values is None:
        et_note = f"'{snr_column}' is missing; only the in-band filter is available."
    else:
        et_note = "ET/SNR filters select the colored foreground; all events remain in gray."

    panel = widgets.VBox([
        widgets.Label("Choose the plot type, variables, scales and filters, then click Plot."),
        plot_type,
        widgets.HBox([x_variable, y_variable]),
        widgets.HBox([x_scale, y_scale]),
        widgets.HBox([in_band_only, minimum_snr]),
        widgets.Label(et_note),
        widgets.Label(cluster_note),
        widgets.HBox(list(cluster_overlays.values())),
        widgets.HBox([bins, density, plot_button]),
        output,
    ])
    display(panel)
    redraw()
    return panel
