import pandas as pd
import plotly.express as px
from dash import Dash, html, dcc, dash_table, Input, Output
from dataclasses import dataclass, asdict
from typing import List
import re

from src.data.utils import GenerationTaskAnalysis


def format_conversation(text):
    """Parses dialog text and returns a list of Dash components with bolded speakers."""
    if not text: return ""
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
        record = {
            "id": i,
            # Ensure categorical fields are strings for Dash DataTable
            "answerability": str(task.answerability) if task.answerability else "",
            "multi_turn": str(task.multi_turn) if task.multi_turn else "",
            "question_type": str(task.question_type) if task.question_type else "",
            "dialog": task.dialog,
            "reference": task.reference,
            "prediction": task.prediction,
            "documents_preview": "\n".join(task.documents)[:50] + "..." if task.documents else "",
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

# --- Main Dashboard Logic ---

def run_dashboard(tasks: List[GenerationTaskAnalysis]):
    df = process_tasks_to_df(tasks)
    
    # Define columns: Metadata -> Metrics
    # Exclude heavy text fields from the overview table
    exclude_cols = ['dialog', 'reference', 'prediction', 'documents_preview']
    meta_cols = ['id', 'answerability', 'multi_turn', 'question_type']
    metric_cols = [c for c in df.columns if c not in exclude_cols and c not in meta_cols]
    
    display_cols = meta_cols + metric_cols
    
    app = Dash(__name__)

    # Minimalist Layout
    app.layout = html.Div([
        # Table Section
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
                page_size=8, # Compact limit
                style_table={'overflowX': 'auto'},
                style_cell={
                    'textAlign': 'left', 
                    'minWidth': '80px', 'maxWidth': '150px', 
                    'overflow': 'hidden', 'textOverflow': 'ellipsis',
                    'fontFamily': 'sans-serif', 'fontSize': '12px'
                },
                style_header={'fontWeight': 'bold', 'backgroundColor': '#f0f0f0'},
                style_data_conditional=[
                    {'if': {'column_id': c}, 'backgroundColor': '#f9fbfd'} 
                    for c in ['answerability', 'multi_turn', 'question_type']
                ]
            )
        ], style={'marginBottom': '10px'}),

        # Detailed View Section
        html.Div(id='detail-container', style={
            'border': '1px solid #ccc', 'padding': '15px', 
            'borderRadius': '4px', 'fontFamily': 'sans-serif', 'backgroundColor': '#fff'
        })
    ], style={'padding': '10px', 'backgroundColor': '#fafafa', 'height': '100vh'})

    # --- Callback ---

    @app.callback(
        Output('detail-container', 'children'),
        Input('datatable', 'selected_rows')
    )
    def update_details(selected_rows):
        if not selected_rows:
            return html.Div("Select a row.")
        
        row_idx = selected_rows[0]
        # Safety check for index
        if row_idx >= len(tasks): return html.Div("Data mismatch.")
        task = tasks[row_idx]
        
        # Styles
        box_style = {
            'width': '100%', 'height': '150px', 'padding': '8px', 
            'overflowY': 'scroll', 'border': '1px solid #ddd', 
            'backgroundColor': '#f7f7f7', 'whiteSpace': 'pre-wrap', 'fontSize': '13px'
        }
        titles_style = {'fontWeight': 'bold', 'color': '#333', 'fontSize': '13px', 'marginBottom': '4px', 'marginTop': '10px'}
        
        def meta_badge(label, value):
            return html.Span([
                html.Span(f"{label}: ", style={'color': '#666'}), 
                html.B(str(value)) 
            ], style={
                'marginLeft': '15px', 'padding': '2px 8px', 
                'backgroundColor': '#eef', 'borderRadius': '4px', 
                'fontSize': '12px', 'border': '1px solid #dde'
            })

        return html.Div([
            # HEADER: Title Left, Metadata Right
            html.Div([
                html.H4(f"Detailed Analysis (Task ID: {row_idx})", style={'margin': 0, 'color': '#222'}),
                html.Div([
                    meta_badge("Ans", task.answerability),
                    meta_badge("Turn", task.multi_turn),
                    meta_badge("Type", task.question_type),
                ])
            ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'borderBottom': '1px solid #eee', 'paddingBottom': '8px'}),

            # CONTENT: 2 Columns
            html.Div([
                # Left Column
                html.Div([
                    html.Div("Dialog / Context", style=titles_style),
                    html.Div(format_conversation(task.dialog), style=box_style),
                    
                    html.Div("Reference Answer", style=titles_style),
                    html.Div(task.reference, style=box_style),
                ], style={'width': '49%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginRight': '2%'}),
                
                # Right Column
                html.Div([
                    html.Div("Documents", style=titles_style),
                    html.Div("\n\n".join(task.documents), style=box_style),
                    
                    html.Div("Model Prediction", style=titles_style),
                    html.Div(str(task.prediction), style=box_style),
                ], style={'width': '49%', 'display': 'inline-block', 'verticalAlign': 'top'}),
            ]),
            
            # FOOTER: Metrics
            html.Div([
                html.Div("Specific Metrics", style=titles_style),
                html.Pre(str(task.metrics).replace("GenerationTaskMetrics", ""), 
                         style={'backgroundColor': '#eee', 'padding': '8px', 'borderRadius': '4px', 'fontSize': '11px', 'margin': 0})
            ])
        ])

    app.run(debug=True, host='0.0.0.0', port=8051)