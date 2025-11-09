import io

import numpy as np
import pandas as pd
import panel as pn

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool

pn.extension('tabulator')


class AnomalyDashboard:
    def __init__(self):
        # Data
        self.df = None
        self.results_df = None

        # Widgets
        self.file_input = pn.widgets.FileInput(
            name="Upload CSV",
            accept=".csv",
            sizing_mode="stretch_width"
        )

        self.model_select = pn.widgets.RadioButtonGroup(
            name="Model",
            options=["Isolation Forest", "OCSVM", "Both"],
            value="Isolation Forest",
            button_type="success"
        )

        self.feature_selector = pn.widgets.MultiSelect(
            name="Feature columns (used for training)",
            options=[],
            size=10,
            sizing_mode="stretch_width"
        )

        self.x_axis = pn.widgets.Select(
            name="X axis (for scatter plot)",
            options=[],
            sizing_mode="stretch_width"
        )

        self.y_axis = pn.widgets.Select(
            name="Y axis (for scatter plot)",
            options=[],
            sizing_mode="stretch_width"
        )

        self.if_contamination = pn.widgets.FloatSlider(
            name="Isolation Forest: contamination",
            start=0.001,
            end=0.30,
            step=0.001,
            value=0.05
        )

        self.oc_nu = pn.widgets.FloatSlider(
            name="OCSVM: ν (expected anomalies proportion)",
            start=0.001,
            end=0.30,
            step=0.001,
            value=0.05
        )

        self.oc_gamma = pn.widgets.Select(
            name="OCSVM: gamma",
            options=["scale", "auto"],
            value="scale"
        )

        self.run_button = pn.widgets.Button(
            name="Run Anomaly Detection",
            button_type="primary",
            sizing_mode="stretch_width"
        )

        # Output panes
        self.status_pane = pn.pane.Markdown(
            "### Status\nUpload a CSV file to get started.",
            sizing_mode="stretch_width"
        )

        self.table = pn.widgets.Tabulator(
            name="Detected anomalies",
            pagination="remote",
            page_size=20,
            show_index=False,
            sizing_mode="stretch_both"
        )

        self.plot_pane = pn.pane.Bokeh(
            sizing_mode="stretch_both",
            height=500
        )

        # Callbacks
        self.file_input.param.watch(self._on_file_upload, "value")
        self.run_button.on_click(self.run_anomaly_detection)

    # ---------- Data loading ----------

    def _on_file_upload(self, event):
        if not event.new:
            return

        try:
            raw = io.BytesIO(event.new)
            df = pd.read_csv(raw)
        except UnicodeDecodeError:
            raw = io.BytesIO(event.new)
            df = pd.read_csv(raw, encoding="latin1")
        except Exception as e:
            self.status_pane.object = f"### Status\nError reading file: `{e}`"
            return

        self.df = df
        self.results_df = None

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_options = numeric_cols if numeric_cols else list(df.columns)

        self.feature_selector.options = feature_options
        self.feature_selector.value = feature_options[: min(5, len(feature_options))]

        # Axis options
        self.x_axis.options = feature_options
        self.y_axis.options = feature_options

        if feature_options:
            self.x_axis.value = feature_options[0]
            self.y_axis.value = feature_options[1] if len(feature_options) > 1 else feature_options[0]

        self.status_pane.object = (
            f"### Status\nCSV loaded with **{df.shape[0]}** rows and **{df.shape[1]}** columns.\n\n"
            f"Select feature columns and click **Run Anomaly Detection**."
        )

    # ---------- Core logic ----------

    def run_anomaly_detection(self, _=None):
        if self.df is None:
            self.status_pane.object = "### Status\n⚠️ Please upload a CSV file first."
            return

        features = list(self.feature_selector.value)
        if not features:
            self.status_pane.object = "### Status\n⚠️ Please select at least one feature column."
            return

        # Ensure selected columns exist
        for col in features:
            if col not in self.df.columns:
                self.status_pane.object = f"### Status\n⚠️ Column `{col}` not found in data."
                return

        # Prepare data
        X = self.df[features].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model_choice = self.model_select.value
        res_df = self.df.copy()

        # Isolation Forest
        if model_choice in ("Isolation Forest", "Both"):
            try:
                iforest = IsolationForest(
                    contamination=self.if_contamination.value,
                    random_state=42
                )
                if_labels = iforest.fit_predict(X_scaled)
                if_scores = iforest.decision_function(X_scaled)

                res_df["IF_score"] = if_scores
                res_df["IF_label"] = if_labels
                res_df["IF_anomaly"] = (if_labels == -1)
            except Exception as e:
                self.status_pane.object = f"### Status\nError in Isolation Forest: `{e}`"
                return

        # OCSVM
        if model_choice in ("OCSVM", "Both"):
            try:
                ocsvm = OneClassSVM(
                    nu=self.oc_nu.value,
                    gamma=self.oc_gamma.value
                )
                oc_labels = ocsvm.fit_predict(X_scaled)
                oc_scores = ocsvm.decision_function(X_scaled)

                res_df["OCSVM_score"] = oc_scores
                res_df["OCSVM_label"] = oc_labels
                res_df["OCSVM_anomaly"] = (oc_labels == -1)
            except Exception as e:
                self.status_pane.object = f"### Status\nError in OCSVM: `{e}`"
                return

        # Combined anomaly if both
        if model_choice == "Both":
            res_df["Combined_anomaly"] = (
                res_df.get("IF_anomaly", False) | res_df.get("OCSVM_anomaly", False)
            )

        self.results_df = res_df

        # Update table + plot
        self._update_views()

    # ---------- Views ----------

    def _update_views(self):
        if self.results_df is None:
            return

        # Decide which anomaly flag is "active"
        model_choice = self.model_select.value
        df = self.results_df.copy()

        if model_choice == "Isolation Forest":
            active_col = "IF_anomaly"
            label = "Isolation Forest"
        elif model_choice == "OCSVM":
            active_col = "OCSVM_anomaly"
            label = "OCSVM"
        else:
            active_col = "Combined_anomaly"
            label = "IF OR OCSVM"

        if active_col not in df.columns:
            # Safety fallback
            df["Active_anomaly"] = False
        else:
            df["Active_anomaly"] = df[active_col]

        # Put anomaly flag early in the table for clarity
        anomaly_cols = [c for c in ["Active_anomaly", "IF_anomaly", "OCSVM_anomaly", "Combined_anomaly"] if c in df.columns]
        other_cols = [c for c in df.columns if c not in anomaly_cols]
        ordered = anomaly_cols + other_cols

        self.table.value = df[ordered]

        n_total = len(df)
        n_anom = int(df["Active_anomaly"].sum())

        self.status_pane.object = (
            f"### Status\n"
            f"Model: **{label}**  |  Total points: **{n_total}**  |  Detected anomalies: **{n_anom}**\n\n"
            f"Use the table to filter/sort and the scatter plot to visually inspect anomalies."
        )

        self.plot_pane.object = self._build_plot(df)

    def _build_plot(self, df: pd.DataFrame):
        x = self.x_axis.value
        y = self.y_axis.value

        if not x or not y:
            return None
        if x not in df.columns or y not in df.columns:
            return None

        # Only plot numeric; if not numeric, skip
        if not np.issubdtype(df[x].dtype, np.number) or not np.issubdtype(df[y].dtype, np.number):
            return None

        anomalies = df["Active_anomaly"].fillna(False)

        color = np.where(anomalies, "red", "gray")
        label = np.where(anomalies, "Anomaly", "Normal")

        source = ColumnDataSource(data=dict(
            x=df[x],
            y=df[y],
            anomaly=label,
            color=color,
        ))

        p = figure(
            height=500,
            sizing_mode="stretch_both",
            x_axis_label=x,
            y_axis_label=y,
            title=f"{x} vs {y} (red = anomalies)"
        )
        p.circle(
            x="x",
            y="y",
            source=source,
            size=6,
            color="color",
            line_alpha=0.2,
            fill_alpha=0.7,
        )

        hover = HoverTool(
            tooltips=[
                (x, "@x"),
                (y, "@y"),
                ("Label", "@anomaly"),
            ]
        )
        p.add_tools(hover)
        p.toolbar.autohide = True

        return p

    # ---------- Public entrypoint ----------

    def panel(self):
        controls = pn.Column(
            "## Controls",
            self.file_input,
            pn.layout.Divider(),
            pn.pane.Markdown("### Models"),
            self.model_select,
            pn.pane.Markdown("### Isolation Forest"),
            self.if_contamination,
            pn.pane.Markdown("### OCSVM"),
            self.oc_nu,
            self.oc_gamma,
            pn.layout.Divider(),
            pn.pane.Markdown("### Features & Axes"),
            self.feature_selector,
            self.x_axis,
            self.y_axis,
            pn.layout.Spacer(height=10),
            self.run_button,
            sizing_mode="fixed",
            width=320,
        )

        main_tabs = pn.Tabs(
            ("Status", self.status_pane),
            ("Table", self.table),
            ("Scatter Plot", self.plot_pane),
            sizing_mode="stretch_both"
        )

        return pn.Row(
            controls,
            pn.layout.HSpacer(width=10),
            pn.Column(main_tabs, sizing_mode="stretch_both"),
            sizing_mode="stretch_both",
        )


# Serveable app
anomaly_dashboard = AnomalyDashboard()
anomaly_dashboard.panel().servable()
