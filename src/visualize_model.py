import numpy as np
import plotly.graph_objects as go


def create_confusion_matrix_figure(cm: list, labels: list, title: str) -> go.Figure:
    """
    Creates an interactive Plotly confusion matrix heatmap.

    Parameters:
        cm (list or np.ndarray): 2D confusion matrix array.
        labels (list): Class label strings (['REST', 'LEFT_FIST', 'RIGHT_FIST']).
        title (str): Plot title.

    Returns:
        go.Figure: Interactive heatmap figure.
    """
    cm_arr = np.array(cm)
    
    # Text annotations inside heatmap cells
    annotations = []
    for i, row in enumerate(cm_arr):
        for j, val in enumerate(row):
            annotations.append(
                dict(
                    x=labels[j],
                    y=labels[i],
                    text=str(val),
                    font=dict(size=14, color="#FFFFFF" if val > np.max(cm_arr)/2 else "#CBD5E1"),
                    showarrow=False
                )
            )

    fig = go.Figure(data=go.Heatmap(
        z=cm_arr,
        x=labels,
        y=labels,
        colorscale="Blues",
        showscale=True,
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>"
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#1E293B",
        title=dict(text=title, font=dict(size=16, color="#F8FAFC")),
        xaxis=dict(title="Predicted Class", gridcolor="#334155"),
        yaxis=dict(title="Actual Class", gridcolor="#334155", autorange="reversed"),
        annotations=annotations,
        margin=dict(l=60, r=40, t=50, b=50),
        height=380
    )

    return fig


def create_model_comparison_chart(svm_metrics: dict, rf_metrics: dict) -> go.Figure:
    """
    Creates a grouped bar chart comparing test performance between SVM and Random Forest.

    Parameters:
        svm_metrics (dict): Dict containing accuracy, precision, recall, f1.
        rf_metrics (dict): Dict containing accuracy, precision, recall, f1.

    Returns:
        go.Figure: Grouped bar chart figure.
    """
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    svm_vals = [
        svm_metrics.get("accuracy", 0.0),
        svm_metrics.get("precision", 0.0),
        svm_metrics.get("recall", 0.0),
        svm_metrics.get("f1", 0.0)
    ]
    rf_vals = [
        rf_metrics.get("accuracy", 0.0),
        rf_metrics.get("precision", 0.0),
        rf_metrics.get("recall", 0.0),
        rf_metrics.get("f1", 0.0)
    ]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=metrics,
        y=svm_vals,
        name="SVM",
        marker_color="#38BDF8",
        text=[f"{v*100:.1f}%" for v in svm_vals],
        textposition="auto"
    ))

    fig.add_trace(go.Bar(
        x=metrics,
        y=rf_vals,
        name="Random Forest",
        marker_color="#10B981",
        text=[f"{v*100:.1f}%" for v in rf_vals],
        textposition="auto"
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#1E293B",
        title=dict(text="Model Performance Comparison on Unseen Test Subjects", font=dict(size=16, color="#F8FAFC")),
        xaxis=dict(title="Metric", gridcolor="#334155"),
        yaxis=dict(title="Score (0 - 1.0)", range=[0, 1.05], gridcolor="#334155"),
        barmode="group",
        height=380,
        margin=dict(l=50, r=30, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig


def create_feature_importance_chart(feature_importances: dict) -> go.Figure:
    """
    Creates an interactive horizontal bar chart showing Gini feature importances.
    """
    if not feature_importances:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0F172A",
            plot_bgcolor="#1E293B",
            title="Feature Importance (Linear / Logistic Model Selected)",
            height=320
        )
        return fig

    sorted_feats = sorted(feature_importances.items(), key=lambda x: x[1], reverse=False)
    names = [x[0] for x in sorted_feats]
    vals = [x[1] for x in sorted_feats]

    fig = go.Figure(go.Bar(
        x=vals,
        y=names,
        orientation='h',
        marker=dict(
            color=vals,
            colorscale="Viridis",
            showscale=False
        ),
        text=[f"{v:.4f}" for v in vals],
        textposition="auto"
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#1E293B",
        title=dict(text="Mathematical EEG Feature Importance (Random Forest)", font=dict(size=16, color="#F8FAFC")),
        xaxis=dict(title="Importance Score", gridcolor="#334155"),
        yaxis=dict(title="Feature Name", gridcolor="#334155"),
        height=450,
        margin=dict(l=140, r=40, t=50, b=40)
    )

    return fig

