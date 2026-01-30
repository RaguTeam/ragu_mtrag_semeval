import pandas as pd
import plotly.express as px
from dash import Dash, html, dcc, dash_table, Input, Output
from dataclasses import dataclass, asdict
from typing import List
import re

from src.data.utils import GenerationTaskAnalysis


def format_conversation(text):
    """Parses dialog text and returns a list of Dash components with bolded speakers."""
    if not text:
        return ""
    # Split by common speaker patterns
    parts = re.split(r'(User:|Assistant:|Bot:)', str(text))
    children = []
    for part in parts:
        if part.strip() in ["User:", "Assistant:", "Bot:"]:
            children.append(html.B(part))
        else:
            children.append(html.Span(part))
    return children

def process_tasks_to_df(tasks: List[GenerationTaskAnalysis]) -> pd.DataFrame:
    data_records = []
    for i, task in enumerate(tasks):
        # We explicitly cast categorical fields to str() to prevent Dash JSON errors
        # in case the input data contains Enums or None objects.
        record = {
            "id": i,
            "answerability": str(task.answerability) if task.answerability else "",
            "multi_turn": str(task.multi_turn) if task.multi_turn else "",
            "question_type": str(task.question_type) if task.question_type else "",
            "dialog": task.dialog,
            "reference": task.reference,
            "prediction": task.prediction,
            "documents_preview": "\n".join(task.documents)[:50] + "..." if task.documents else "",
        }
        
        # Flatten metrics
        if task.metrics:
            metrics_dict = asdict(task.metrics)
            for k, v in metrics_dict.items():
                if isinstance(v, list): continue 
                record[k] = v
        else:
            # Populate None for metrics if missing so columns align
            dummy_metrics = [f for f in GenerationTaskMetrics.__annotations__ 
                             if f not in ['BertKPrec', 'Extractiveness_RougeL']]
            for key in dummy_metrics:
                record[key] = None
                
        data_records.append(record)
        
    return pd.DataFrame(data_records)

# --- Main Dashboard Logic ---

def run_dashboard(tasks: List[GenerationTaskAnalysis]):
    df = process_tasks_to_df(tasks)
    
    # Define columns for the table
    text_cols = ['dialog', 'reference', 'prediction', 'documents_preview']
    meta_cols = ['id', 'answerability', 'multi_turn', 'question_type']
    metric_cols = [c for c in df.columns if c not in text_cols and c not in meta_cols]
    
    # Display Order: ID -> Metadata -> Metrics
    display_cols = meta_cols + metric_cols
    
    app = Dash(__name__)

    app.layout = html.Div([
        html.H1("Generation Task Evaluation Dashboard", 
                style={'textAlign': 'center', 'fontFamily': 'sans-serif'}),
        
        html.H3("Task List", style={'fontFamily': 'sans-serif'}),
        html.P("Sort and filter tasks using the table headers. Select a row to view full details.", 
               style={'fontFamily': 'sans-serif', 'marginBottom': '10px'}),

        # Data Table
        html.Div([
            dash_table.DataTable(
                id='datatable',
                columns=[{"name": i, "id": i} for i in display_cols],
                data=df.to_dict('records'),
                sort_action="native",
                sort_mode="multi",
                filter_action="native",
                row_selectable='single',
                selected_rows=[0],
                page_action="native",
                page_current=0,
                page_size=15,
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left', 'minWidth': '90px', 'maxWidth': '180px', 'overflow': 'hidden'},
                style_header={'fontWeight': 'bold', 'backgroundColor': '#e8e8e8'},
                # Highlights for metadata columns
                style_data_conditional=[
                    {
                        'if': {'column_id': c},
                        'backgroundColor': '#f0f8ff'
                    } for c in ['answerability', 'multi_turn', 'question_type']
                ]
            )
        ], style={'marginBottom': '30px'}),

        # Detailed View Container
        html.Div(id='detail-container', style={
            'border': '1px solid #ccc', 
            'padding': '20px', 
            'borderRadius': '5px',
            'fontFamily': 'sans-serif',
            'backgroundColor': '#fff'
        })
    ], style={'padding': '20px', 'backgroundColor': '#fafafa'})

    # --- Callbacks ---

    @app.callback(
        Output('detail-container', 'children'),
        Input('datatable', 'selected_rows')
    )
    def update_details(selected_rows):
        if not selected_rows:
            return html.Div("Select a row in the table to view details.")
        
        row_idx = selected_rows[0]
        # Ensure we don't go out of bounds if filters are applied weirdly
        if row_idx >= len(tasks):
            return html.Div("Selection out of bounds.")
            
        task = tasks[row_idx]
        
        box_style = {
            'width': '100%', 
            'height': '200px', 
            'padding': '10px', 
            'overflowY': 'scroll', 
            'border': '1px solid #ddd', 
            'backgroundColor': '#fcfcfc',
            'whiteSpace': 'pre-wrap'
        }
        
        titles_style = {'fontWeight': 'bold', 'color': '#555', 'marginTop': '15px'}
        
        def meta_badge(label, value):
            return html.Span([
                html.B(f"{label}: "), 
                html.Span(str(value)) # Ensure value is string
            ], style={'marginRight': '20px', 'padding': '5px 10px', 'backgroundColor': '#e6f2ff', 'borderRadius': '15px', 'border': '1px solid #b3d9ff'})

        return html.Div([
            html.H4(f"Detailed Analysis (Task ID: {row_idx})"),
            
            # Metadata Strip
            html.Div([
                meta_badge("Answerability", task.answerability),
                meta_badge("Multi-turn", task.multi_turn),
                meta_badge("Type", task.question_type),
            ], style={'padding': '10px 0', 'marginBottom': '10px', 'borderBottom': '1px solid #eee'}),

            # Main Content
            html.Div([
                html.Div([
                    html.Div("Dialog / Context", style=titles_style),
                    html.Div(format_conversation(task.dialog), style=box_style),
                    
                    html.Div("Reference Answer", style=titles_style),
                    html.Div(task.reference, style=box_style),
                ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginRight': '2%'}),
                
                html.Div([
                    html.Div("Documents", style=titles_style),
                    html.Div("\n\n".join(task.documents), style=box_style),
                    
                    html.Div("Model Prediction", style=titles_style),
                    html.Div(str(task.prediction), style=box_style),
                ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
            ]),
            
            # Metrics
            html.Div([
                html.Div("Specific Metrics", style=titles_style),
                html.Pre(str(task.metrics).replace("GenerationTaskMetrics", ""), 
                         style={'backgroundColor': '#eee', 'padding': '10px', 'marginTop': '5px'})
            ])
        ])

    app.run(debug=True, host='0.0.0.0', port=8051)