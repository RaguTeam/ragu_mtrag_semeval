# type: ignore

import dataclasses
from dash import Dash, html, dcc, Input, Output
import plotly.graph_objects as go
from typing import Literal, Any

# Required import
from src.data.utils import GenerationTaskAnalysis

def run_dashboard(predictions: dict[str, list[GenerationTaskAnalysis]]):
    """Run a dashboard to analyze how well an agent continues a dialog.

    The `predictions` argument is a dict from model name (such as "qwen_7b") to
    the list of `GenerationTaskAnalysis` that consists of the input data,
    the prediction and the metrics.
    """

    # --- 1. Data Preprocessing ---

    all_task_ids_set = set()
    data_map: dict[str, dict[str, GenerationTaskAnalysis]] = {}

    available_metrics_set = set()
    available_analysis_keys = set()

    models = sorted(list(predictions.keys()))

    for model_name, tasks in predictions.items():
        data_map[model_name] = {}
        for task in tasks:
            all_task_ids_set.add(task.task_id)
            data_map[model_name][task.task_id] = task

            # Find float metrics
            if task.metrics:
                for field_name, value in task.metrics.__dict__.items():
                    if isinstance(value, float):
                        available_metrics_set.add(field_name)

            # Find analysis keys
            if task.analysis:
                for item in task.analysis:
                    title, _, analysis_model, _ = item
                    key = f"{title} | {analysis_model}"
                    available_analysis_keys.add(key)

    sorted_ids = sorted(list(all_task_ids_set))
    idx_to_id = {i: t_id for i, t_id in enumerate(sorted_ids)}

    metric_options = sorted(list(available_metrics_set))
    analysis_options = sorted(list(available_analysis_keys))

    default_metric = metric_options[0] if metric_options else ""
    default_analysis = analysis_options[0] if analysis_options else ""

    # --- 2. Styles ---

    # Main container: Full viewport height, flex column
    STYLE_CONTAINER = {
        'fontFamily': 'sans-serif',
        'position': 'fixed', # Forces the container to ignore body margins
        'top': 0,
        'left': 0,
        'bottom': 0,
        'right': 0,
        'display': 'flex',
        'flexDirection': 'column',
        'overflow': 'hidden',
        'padding': '10px',
        'boxSizing': 'border-box',
    }

    STYLE_ROW_CONTROL = {'display': 'flex', 'gap': '15px', 'paddingBottom': '10px', 'flexShrink': 0}
    STYLE_CONTROL_GROUP = {'display': 'flex', 'flexDirection': 'column', 'minWidth': '200px'}
    STYLE_LABEL = {'fontSize': '12px', 'fontWeight': 'bold', 'color': '#555', 'marginBottom': '2px'}

    STYLE_PLOT_AREA = {'height': '16vh', 'minHeight': '150px', 'marginBottom': '10px', 'flexShrink': 0}

    # Details Area: Fills remaining space
    STYLE_DETAILS_AREA = {
        'flex': '1',
        'display': 'flex',
        'flexDirection': 'column',
        'minHeight': '0', # Crucial for flex scrolling
        'border': '1px solid #ccc',
        'borderRadius': '4px',
    }

    STYLE_DETAILS_HEADER = {
        'backgroundColor': '#f1f1f1',
        'padding': '8px',
        'borderBottom': '1px solid #ccc',
        'display': 'flex',
        'gap': '20px',
        'flexWrap': 'wrap',
        'fontSize': '14px',
        'flexShrink': 0,
    }

    # Body: containing Left and Right panels
    STYLE_DETAILS_BODY = {'flex': '1', 'display': 'flex', 'overflow': 'hidden', 'minHeight': '0'}

    # Left Panel: Joint scrollbar for all fields
    STYLE_PANEL_LEFT = {
        'flex': '1',
        'overflowY': 'auto',
        'padding': '15px',
        'borderRight': '1px solid #eee',
        'backgroundColor': '#fafafa',
        'display': 'flex',
        'flexDirection': 'column',
        'gap': '20px',
    }

    # Right Panel: Container for vertically split sections
    STYLE_PANEL_RIGHT = {
        'flex': '1',
        'display': 'flex',
        'flexDirection': 'column',
        'height': '100%',
        'backgroundColor': '#fff',
    }

    # Right Top (Documents): Flex 2
    STYLE_RIGHT_TOP = {
        'flex': '2',
        'overflowY': 'auto',
        'padding': '15px',
        'borderBottom': '1px solid #eee',
    }

    # Right Bottom (Analysis): Flex 1
    STYLE_RIGHT_BOTTOM = {
        'flex': '2',
        'overflowY': 'auto',
        'padding': '15px',
        'backgroundColor': '#fcfcfc',
    }

    STYLE_SECTION_TITLE = {
        'fontSize': '11px',
        'textTransform': 'uppercase',
        'color': '#888',
        'fontWeight': 'bold',
        'marginBottom': '6px',
        'borderBottom': '2px solid #eee',
        'paddingBottom': '2px',
    }

    STYLE_TEXT_BLOCK = {'whiteSpace': 'pre-wrap', 'fontSize': '13px', 'lineHeight': '1.5'}
    STYLE_DOC_BLOCK = {
        'padding': '8px',
        'backgroundColor': '#f8f9fa',
        'border': '1px solid #eee',
        'marginBottom': '8px',
        'fontSize': '12px',
        'borderRadius': '3px',
        'whiteSpace': 'pre-wrap',
    }

    STYLE_FOOTER = {
        'padding': '10px',
        'backgroundColor': '#f1f1f1',
        'borderTop': '1px solid #ccc',
        'fontSize': '13px',
        'display': 'flex',
        'flexWrap': 'wrap',
        'gap': '15px',
        'flexShrink': 0,
    }

    # --- 3. Helper Functions ---

    def format_dialog(text: str) -> list:
        if not text:
            return []
        lines = text.split('\n')
        formatted_content = []
        for line in lines:
            if "User:" in line:
                parts = line.split("User:", 1)
                formatted_content.append(html.Span([parts[0], html.B("User:"), parts[1]]))
            elif "Assistant:" in line:
                parts = line.split("Assistant:", 1)
                formatted_content.append(html.Span([parts[0], html.B("Assistant:"), parts[1]]))
            else:
                formatted_content.append(html.Span(line))
            formatted_content.append(html.Br())
        return formatted_content

    # --- 4. Layout ---

    app = Dash(__name__)

    app.layout = html.Div(
        style=STYLE_CONTAINER, children=[
            # 1. Selectors
            html.Div(
                style=STYLE_ROW_CONTROL, children=[
                    html.Div(
                        style=STYLE_CONTROL_GROUP, children=[
                            html.Label('Model', style=STYLE_LABEL),
                            dcc.Dropdown(
                                id='sel-model', options=[{'label': m, 'value': m} for m in models],
                                value=models[0] if models else None, clearable=False,
                            ),
                        ],
                    ),
                    html.Div(
                        style=STYLE_CONTROL_GROUP, children=[
                            html.Label('Task Index (ID)', style=STYLE_LABEL | {'width': '400px'}),
                            dcc.Dropdown(
                                id='sel-task', options=[{'label': f"#{i} ({t_id})", 'value': i} for i, t_id in enumerate(sorted_ids)],
                                value=0 if sorted_ids else None, clearable=False,
                            ),
                        ],
                    ),
                    html.Div(
                        style=STYLE_CONTROL_GROUP, children=[
                            html.Label('Plot Metric', style=STYLE_LABEL),
                            dcc.Dropdown(
                                id='sel-metric', options=[{'label': m, 'value': m} for m in metric_options],
                                value=default_metric, clearable=False,
                            ),
                        ],
                    ),
                    html.Div(
                        style=STYLE_CONTROL_GROUP, children=[
                            html.Label('Analysis Type', style=STYLE_LABEL | {'width': '500px'}),
                            dcc.Dropdown(
                                id='sel-analysis', options=[{'label': a, 'value': a} for a in analysis_options],
                                value=default_analysis, clearable=True, placeholder="Select analysis...",
                            ),
                        ],
                    ),
                ],
            ),

            # 2. Plots
            html.Div(
                style=STYLE_PLOT_AREA, children=[
                    dcc.Graph(id='main-heatmap', style={'height': '100%'}, config={'displayModeBar': False}),
                ],
            ),

            # 3. Details (Fills the rest)
            html.Div(id='details-container', style=STYLE_DETAILS_AREA),
        ],
    )

    # --- 5. Callback ---

    @app.callback(
        [
            Output('main-heatmap', 'figure'),
            Output('details-container', 'children'),
        ],
        [
            Input('sel-model', 'value'),
            Input('sel-task', 'value'),
            Input('sel-metric', 'value'),
            Input('sel-analysis', 'value'),
        ],
    )
    def update_view(selected_model, task_idx, selected_metric, selected_analysis):

        # --- A. Build Heatmap ---
        z_values = []
        hover_texts = []

        for m_name in models:
            row_z = []
            row_hover = []
            for t_idx, t_id in enumerate(sorted_ids):
                task_obj = data_map[m_name].get(t_id)
                val = float('nan')
                txt = "N/A"
                if task_obj and task_obj.metrics and selected_metric:
                    m_val = getattr(task_obj.metrics, selected_metric, None)
                    if isinstance(m_val, float):
                        val = m_val
                        txt = f"{m_val:.4f}"
                row_z.append(val)
                row_hover.append(f"Model: {m_name}<br>Task: #{t_idx} ({t_id})<br>{selected_metric}: {txt}")
            z_values.append(row_z)
            hover_texts.append(row_hover)

        fig = go.Figure(
            data=go.Heatmap(
                z=z_values, x=[f"#{i}" for i in range(len(sorted_ids))], y=models,
                text=hover_texts, hoverinfo='text', colorscale='Portland', showscale=True,
            ),
        )

        fig.update_layout(
            margin=dict(l=50, r=50, t=10, b=30),
            xaxis=dict(title='Task Index', tickmode='auto'),
            yaxis=dict(title='', automargin=True),
        )

        if selected_model and task_idx is not None:
             try:
                y_idx = models.index(selected_model)
                fig.add_shape(
                    type="rect", x0=task_idx-0.5, x1=task_idx+0.5, y0=y_idx-0.5, y1=y_idx+0.5,
                    line=dict(color="red", width=2), fillcolor="rgba(0,0,0,0)",
                )
             except ValueError: pass

        # --- B. Build Details ---

        if not selected_model or task_idx is None:
            return fig, html.Div("Please select a model and task.", style={'padding': '20px'})

        task_id = idx_to_id[task_idx]
        task_data = data_map.get(selected_model, {}).get(task_id)

        if not task_data:
            return fig, html.Div(
                html.H3(f"No prediction found for Model '{selected_model}' on Task '{task_id}'"),
                style={'padding': '20px', 'textAlign': 'center', 'color': '#777'},
            )

        # 1. Header
        header = html.Div(
            style=STYLE_DETAILS_HEADER, children=[
                html.Span([html.B("Task ID: "), task_data.task_id]),
                html.Span([html.B("Answerability: "), task_data.answerability]),
                html.Span([html.B("Multi Turn: "), task_data.multi_turn]),
                html.Span([html.B("Type: "), task_data.question_type]),
            ],
        )

        # 2. Content Sections

        # --- Left Side: Dialog, Reference, Prediction ---
        pred_content = task_data.prediction if task_data.prediction else html.I("None", style={'color': '#999'})

        left_panel = html.Div(
            style=STYLE_PANEL_LEFT, children=[
                html.Div([
                    html.Div("Dialog", style=STYLE_SECTION_TITLE),
                    html.Div(format_dialog(task_data.dialog), style=STYLE_TEXT_BLOCK),
                ]),
                html.Div([
                    html.Div("Reference", style=STYLE_SECTION_TITLE),
                    html.Div(task_data.reference, style=STYLE_TEXT_BLOCK),
                ]),
                html.Div([
                    html.Div("Prediction", style=STYLE_SECTION_TITLE),
                    html.Div(pred_content, style=STYLE_TEXT_BLOCK),
                ]),
            ],
        )

        # --- Right Side: Documents (Top), Analysis (Bottom) ---

        # Documents
        doc_elements = []
        if task_data.documents:
            for i, doc in enumerate(task_data.documents):
                doc_elements.append(html.Div(doc, style=STYLE_DOC_BLOCK))
        else:
            doc_elements.append(html.Div("No documents.", style={'fontStyle': 'italic', 'color': '#999'}))

        right_docs = html.Div(
            style=STYLE_RIGHT_TOP, children=[
                html.Div("Documents", style=STYLE_SECTION_TITLE),
                html.Div(doc_elements),
            ],
        )

        # Analysis
        analysis_divs = []
        if selected_analysis and task_data.analysis:
            found = False
            for (t, p, m, a) in task_data.analysis:
                if f"{t} | {m}" == selected_analysis:
                    found = True
                    analysis_divs = [
                        html.Div([
                            html.Span("Prompt:", style={'fontWeight': 'bold', 'color': '#555'}),
                            html.Div(p, style={**STYLE_TEXT_BLOCK, 'marginBottom': '10px', 'paddingLeft': '10px', 'borderLeft': '3px solid #ddd'}),
                        ]),
                        html.Div([
                            html.Span("Answer:", style={'fontWeight': 'bold', 'color': '#555'}),
                            html.Div(a, style={**STYLE_TEXT_BLOCK, 'paddingLeft': '10px', 'borderLeft': '3px solid #ddd'}),
                        ]),
                    ]
                    break
            if not found:
                analysis_divs = [html.Div("Analysis not available for this task.", style={'color': '#999'})]
        elif selected_analysis:
             analysis_divs = [html.Div("No analysis data present.", style={'color': '#999'})]
        else:
            analysis_divs = [html.Div("Select an analysis type above.", style={'color': '#ccc', 'fontStyle': 'italic'})]

        right_analysis = html.Div(
            style=STYLE_RIGHT_BOTTOM, children=[
                html.Div(f"Analysis: {selected_analysis if selected_analysis else '(None)'}", style=STYLE_SECTION_TITLE),
                html.Div(analysis_divs),
            ],
        )

        # Assemble Right Panel
        right_panel = html.Div(style=STYLE_PANEL_RIGHT, children=[right_docs, right_analysis])

        # 3. Footer
        metric_spans = []
        if task_data.metrics:
            for k, v in task_data.metrics.__dict__.items():
                if isinstance(v, float):
                    style = {'fontWeight': 'bold', 'color': '#007bff'} if k == selected_metric else {}
                    metric_spans.append(
                        html.Span([
                            html.Span(f"{k}: ", style={'color': '#666'}),
                            html.Span(f"{v:.4f}", style=style),
                        ]),
                    )

        footer = html.Div(style=STYLE_FOOTER, children=metric_spans if metric_spans else "No metrics available.")

        # Final Assembly
        details_layout = html.Div(
            [
                header,
                html.Div(style=STYLE_DETAILS_BODY, children=[left_panel, right_panel]),
                footer,
            ], style={'display': 'flex', 'flexDirection': 'column', 'height': '100%'},
        )

        return fig, details_layout

    # --- 6. Run ---
    app.run(debug=True, host='0.0.0.0', port=8051)
