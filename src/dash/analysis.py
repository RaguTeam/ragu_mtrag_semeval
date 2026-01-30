import pandas as pd
import plotly.express as px
from dash import Dash, html, dcc, dash_table, Input, Output
from dataclasses import dataclass, asdict
from typing import List
import re

from src.data.utils import GenerationTaskAnalysis


def process_tasks_to_df(tasks: List[GenerationTaskAnalysis]) -> pd.DataFrame:
    data_records = []
    for i, task in enumerate(tasks):
        record = {
            "id": i,
            "dialog": task.dialog,
            "reference": task.reference,
            "documents_preview": "\n".join(task.documents)[:100] + "..." if task.documents else "",
            "prediction": task.prediction
        }
        if task.metrics:
            metrics_dict = asdict(task.metrics)
            for k, v in metrics_dict.items():
                if isinstance(v, list): continue 
                record[k] = v
        else:
            dummy_metrics = [f for f in GenerationTaskMetrics.__annotations__ 
                             if f not in ['BertKPrec', 'Extractiveness_RougeL']]
            for key in dummy_metrics:
                record[key] = None
        data_records.append(record)
    return pd.DataFrame(data_records)


def format_conversation(text):
    if not text:
        return ""
    
    # 1. Split text by keys, capturing the delimiters so they aren't lost.
    #    This regex finds 'User:' or 'Assistant:'
    parts = re.split(r'(User:|Assistant:)', text)
    
    # 2. Convert parts into Dash components
    children = []
    for part in parts:
        if part in ["User:", "Assistant:"]:
            children.append(html.B(part))  # Make these bold
        else:
            children.append(html.Span(part)) # Keep the rest as plain text
            
    return children


def run_dashboard(tasks: List[GenerationTaskAnalysis]):
    df = process_tasks_to_df(tasks)
    
    # Filter columns for dropdown (exclude text fields)
    metric_cols = [c for c in df.columns if c not in 
                   ['id', 'dialog', 'reference', 'documents_preview', 'prediction']]
    
    app = Dash(__name__)

    app.layout = html.Div([
        html.H1("Generation Task Evaluation Dashboard", 
                style={'textAlign': 'center', 'fontFamily': 'sans-serif'}),
        
        # Upper Control Panel
        # html.Div([
        #     html.Div([
        #         html.Label("Select Metric for Histogram:"),
        #         dcc.Dropdown(
        #             id='metric-dropdown',
        #             options=[{'label': m, 'value': m} for m in metric_cols],
        #             value=metric_cols[0] if metric_cols else None,
        #             clearable=False
        #         ),
        #     ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top'}),
            
        #     html.Div([
        #         dcc.Graph(id='metric-histogram', style={'height': '300px'})
        #     ], style={'width': '68%', 'display': 'inline-block'})
        # ], style={'padding': '20px', 'backgroundColor': '#f9f9f9', 'marginBottom': '20px'}),

        # Data Table
        html.H3("Task List (Select a row to view details)", style={'fontFamily': 'sans-serif'}),
        html.Div([
            dash_table.DataTable(
                id='datatable',
                columns=[{"name": i, "id": i} for i in ['id'] + metric_cols],
                data=df.to_dict('records'),
                sort_action="native",
                sort_mode="multi",
                filter_action="native",
                
                # --- FIXED LINE HERE ---
                row_selectable='single', 
                # -----------------------
                
                selected_rows=[0],
                page_action="native",
                page_current=0,
                page_size=10,
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left', 'minWidth': '80px', 'maxWidth': '180px', 'overflow': 'hidden'},
                style_header={'fontWeight': 'bold', 'backgroundColor': '#e8e8e8'}
            )
        ], style={'marginBottom': '30px'}),

        # Detailed View Container
        html.Div(id='detail-container', style={
            'border': '1px solid #ccc', 
            'padding': '20px', 
            'borderRadius': '5px',
            'fontFamily': 'sans-serif'
        })
    ], style={'padding': '20px'})

    # --- Callbacks ---

    # @app.callback(
    #     Output('metric-histogram', 'figure'),
    #     Input('metric-dropdown', 'value')
    # )
    # def update_graph(selected_metric):
    #     if not selected_metric or selected_metric not in df.columns:
    #         return px.histogram(title="No Metric Selected")
        
    #     fig = px.histogram(df, x=selected_metric, nbins=20, marginal="box", 
    #                        title=f"Distribution of {selected_metric}")
    #     fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    #     return fig

    @app.callback(
        Output('detail-container', 'children'),
        Input('datatable', 'selected_rows')
    )
    def update_details(selected_rows):
        if not selected_rows:
            return html.Div("Select a row in the table to view details.")
        
        # selected_rows returns the index relative to the generated dataframe 
        # (even when sorted, in standard 'native' mode, it preserves original index)
        row_idx = selected_rows[0]
        task = tasks[row_idx]
        
        box_style = {
            'width': '100%', 'height': '200px', 'padding': '10px', 
            'overflowY': 'scroll', 'border': '1px solid #ddd', 
            'backgroundColor': '#fcfcfc', 'whiteSpace': 'pre-wrap'
        }
        titles_style = {'fontWeight': 'bold', 'color': '#555', 'marginTop': '10px'}

        return html.Div([
            html.H4(f"Detailed Analysis (Task ID: {row_idx})"),
            
            html.Div([
                html.Div([
                    html.Div("Dialog / Context", style=titles_style),
                    html.Div(format_conversation(task.dialog), style=box_style),
                ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginRight': '2%'}),
                
                html.Div([
                    html.Div("Documents", style=titles_style),
                    html.Div("\n\n".join(task.documents), style=box_style),
                ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
            ]),
            
            html.Div([
                html.Div([
                    html.Div("Reference Answer", style=titles_style),
                    html.Div(task.reference, style=box_style),
                ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginRight': '2%'}),
                
                html.Div([
                    html.Div("Model Prediction", style=titles_style),
                    html.Div(str(task.prediction), style=box_style),
                ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
            ]),
            
            html.Div([
                html.Div("Specific Metrics", style=titles_style),
                html.Pre(str(task.metrics).replace("GenerationTaskMetrics", ""), 
                         style={'backgroundColor': '#eee', 'padding': '10px'})
            ])
        ])

    app.run(debug=True, host='0.0.0.0', port=8051)