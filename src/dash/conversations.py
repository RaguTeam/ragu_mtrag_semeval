# type: ignore

import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import json
from dataclasses import asdict
from typing import List

from ..data.utils import Conversation, UserMessage, AgentMessage

def run_dashboard(conversations: List[Conversation]):
    """
    Launches a Dash application to visualize the provided list of Conversations.
    """
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    # --- Helper Layout Functions ---

    def render_enrichments(enrichments):
        """Renders UserMessage enrichments as badges."""
        badges = []

        # Multi-turn
        mt_color = "info" if enrichments.multi_turn != "N/A" else "secondary"
        badges.append(dbc.Badge(f"Turn: {enrichments.multi_turn}", color=mt_color, className="me-1"))

        # Answerability
        ans_color = "success" if enrichments.answerability == "ANSWERABLE" else "warning" if enrichments.answerability == "PARTIAL" else "danger"
        badges.append(dbc.Badge(enrichments.answerability, color=ans_color, className="me-1"))

        # Question Types
        for qt in enrichments.question_type:
            # Fixed: removed 'outline' argument, used text_color and light background to distinguish
            badges.append(dbc.Badge(qt, color="light", text_color="dark", className="me-1 border border-secondary"))

        return html.Div(badges, className="mb-2")

    def render_context(ctx, index):
        """Renders a single RAG context retrieval item."""

        # Handle feedback (Relevance Judgments)
        judgements = []
        if ctx.feedback:
            for fb in ctx.feedback:
                color = "success" if fb.value == "yes" else "danger"
                judgements.append(dbc.Badge(f"{fb.annotator}: {fb.value}", color=color, className="me-1"))

        feedback_div = html.Div(judgements, className="mb-1")

        # Score formatting
        score_badge = dbc.Badge(f"Score: {ctx.score:.4f}", color="primary", className="me-2")

        # Content
        # Handle cases where title or url might be None
        title_text = ctx.title if ctx.title else f"Doc ID: {ctx.document_id}"

        return dbc.AccordionItem(
            [
                html.Div([score_badge, feedback_div]),
                html.H6(title_text, className="card-subtitle text-muted mt-2"),
                html.Small(html.A(ctx.url, href=ctx.url, target="_blank"), className="d-block mb-2") if ctx.url else None,
                html.Pre(ctx.text, style={"whiteSpace": "pre-wrap", "maxHeight": "200px", "overflowY": "auto", "fontSize": "0.85rem", "backgroundColor": "#f8f9fa", "padding": "10px"}),
            ],
            title=f"Doc: {ctx.document_id} (Score: {ctx.score:.2f})",
            item_id=f"ctx-{index}",
        )

    def render_message(msg, idx):
        """Renders a single User or Agent message."""
        if isinstance(msg, UserMessage):
            header = html.Div([
                html.Strong("User"),
                html.Span(f" (TS: {msg.timestamp})", className="text-muted small ms-2"),
            ])
            body = html.Div([
                render_enrichments(msg.enrichments),
                html.Div(msg.text, style={"whiteSpace": "pre-wrap"}),
            ])
            return dbc.Card(
                dbc.CardBody([header, body]),
                className="mb-3 ms-5 shadow-sm border-primary",
                style={"backgroundColor": "#f0f8ff"}, # Light aliceblue
            )

        elif isinstance(msg, AgentMessage):
            header = html.Div([
                html.Strong("Agent"),
                html.Span(f" (TS: {msg.timestamp})", className="text-muted small ms-2"),
            ])

            # Text Content
            text_content = html.Div(msg.text, style={"whiteSpace": "pre-wrap", "marginBottom": "15px"})

            # Contexts (Accordion)
            ctx_items = [render_context(ctx, i) for i, ctx in enumerate(msg.contexts)]
            context_accordion = dbc.Accordion(ctx_items, start_collapsed=True, flush=True) if ctx_items else html.Em("No contexts retrieved.")

            # Original text if modified
            original = html.Div([
                html.Hr(),
                html.Small("Original Text (before edit):", className="text-muted"),
                html.Pre(msg.original_text, className="small text-muted"),
            ]) if msg.original_text and msg.original_text != msg.text else None

            return dbc.Card(
                dbc.CardBody([
                    header,
                    text_content,
                    html.Hr(),
                    html.H6("Retrieved Contexts", className="msg-header"),
                    context_accordion,
                    original,
                ]),
                className="mb-3 me-5 shadow-sm",
                style={"backgroundColor": "#ffffff"},
            )
        return html.Div()

    def render_config_json(obj, title):
        """Helper to render complex config objects as JSON trees."""
        try:
            # dataclasses.asdict is recursive
            data_dict = asdict(obj)
            return dbc.Card(
                [
                    dbc.CardHeader(title),
                    dbc.CardBody(
                        html.Pre(json.dumps(data_dict, indent=2), style={"fontSize": "0.75rem", "maxHeight": "300px", "overflow": "auto"}),
                    ),
                ], className="mb-3",
            )
        except Exception as e:
            return dbc.Alert(f"Error parsing {title}: {str(e)}", color="danger")

    # --- Layout ---

    app.layout = dbc.Container(
        [
            dbc.Row([
                dbc.Col(html.H2("MTRAG Conversation Viewer"), width=12, className="my-3 text-center"),
            ]),

            dbc.Row([
                # Sidebar: Conversation Selector
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader("Select Conversation"),
                                dbc.CardBody([
                                    dcc.Dropdown(
                                        id='conv-selector',
                                        options=[
                                            {'label': f"#{i} {c.messages[0].text}", 'value': i}
                                            for i, c in enumerate(conversations)
                                        ],
                                        value=0 if conversations else None,
                                        placeholder="Select a conversation...",
                                    ),
                                    html.Hr(),
                                    html.Div(id='conv-meta-sidebar'),
                                ]),
                            ], style={"height": "85vh", "overflowY": "auto"},
                        ),
                    ], width=3,
                ),

                # Main: Chat Display
                dbc.Col(
                    [
                        dbc.Spinner(html.Div(id='chat-display'), color="primary"),
                    ], width=6, style={"height": "85vh", "overflowY": "auto"},
                ),

                # Right Sidebar: Configs & Details
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader("System Configuration"),
                                dbc.CardBody(id='system-config-display'),
                            ], style={"height": "85vh", "overflowY": "auto"},
                        ),
                    ], width=3,
                ),
            ]),
        ], fluid=True, className="bg-light vh-100 p-3",
    )

    # --- Callbacks ---

    @app.callback(
        [
            Output('chat-display', 'children'),
            Output('system-config-display', 'children'),
            Output('conv-meta-sidebar', 'children'),
        ],
        [Input('conv-selector', 'value')],
    )
    def update_view(conv_index):
        if conv_index is None or not conversations:
            return (
                html.Div("Please select a conversation.", className="text-center mt-5"),
                html.Div(),
                html.Div(),
            )

        conv = conversations[conv_index]

        # 1. Render Chat
        chat_bubbles = [render_message(msg, i) for i, msg in enumerate(conv.messages)]

        # 2. Render Configurations (Right Sidebar)
        retriever_viz = render_config_json(conv.retriever, "Retriever Config")
        generator_viz = render_config_json(conv.generator, "Generator Config")

        config_viz = html.Div([
            html.H5("Details", className="mb-3"),
            retriever_viz,
            generator_viz,
        ])

        # 3. Render Metadata (Left Sidebar)
        status_badges = [
            dbc.Badge(
                f"{h.status} by {h.author} (@{h.timestamp})",
                color="secondary",
                className="d-block mb-1 text-wrap",
                style={"textAlign": "left"},
            )
            for h in conv.status_history
        ]

        meta_viz = html.Div([
            html.Strong("Domain: "), html.Span(conv.domain), html.Br(),
            html.Strong("Author: "), html.Span(conv.author), html.Br(),
            html.Strong("Editor: "), html.Span(conv.editor), html.Br(),
            html.Strong("Reviewer: "), html.Span(conv.reviewer), html.Br(),
            html.Hr(),
            html.Strong("Status History:"),
            html.Div(status_badges, className="mt-2"),
        ])

        return chat_bubbles, config_viz, meta_viz

    app.run(debug=True, host='0.0.0.0', port=8051)
