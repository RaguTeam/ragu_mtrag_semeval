# type: ignore

import dash
from dash import html, dcc, Input, Output, State, ALL, ctx
import dash_bootstrap_components as dbc
from collections import defaultdict
import json
from typing import List, Dict

# Assuming the import path provided in the prompt
from ..data.utils import GenerationTask, GenerationTaskMessage, AgentMessageContext


def run_dashboard(generation_tasks: List[GenerationTask]):
    """
    Launches a Dash application to visualize GenerationTask conversations.
    """

    # --- Data Processing ---
    # Group tasks by conversation_id and sort by turn
    conversations: Dict[str, List[GenerationTask]] = defaultdict(list)
    dataset_options = set()
    collection_options = set()

    for task in generation_tasks:
        conversations[task.conversation_id].append(task)
        dataset_options.add(task.dataset)
        collection_options.add(task.collection)

    # Sort turns within conversations
    for conv_id in conversations:
        conversations[conv_id].sort(key=lambda x: x.turn)

    # --- Styles ---
    # Custom CSS for chat interface
    CHAT_STYLE = {
        "user": {
            "backgroundColor": "#dcf8c6",
            "color": "black",
            "padding": "10px",
            "borderRadius": "15px",
            "marginBottom": "10px",
            "maxWidth": "80%",
            "marginLeft": "auto",
            "textAlign": "left",
        },
        "agent": {
            "backgroundColor": "#f1f0f0",
            "color": "black",
            "padding": "10px",
            "borderRadius": "15px",
            "marginBottom": "10px",
            "maxWidth": "80%",
            "marginRight": "auto",
            "textAlign": "left",
        },
        "meta_box": {
            "fontSize": "0.85em",
            "color": "#666",
            "marginBottom": "5px",
            "borderBottom": "1px solid #eee",
            "paddingBottom": "5px",
        },
    }

    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    # --- Components ---

    def render_message_metadata(task: GenerationTask):
        """Renders badges for task metadata."""
        badges = [
            dbc.Badge(f"Turn: {task.turn}", color="dark", className="me-1"),
            dbc.Badge(task.answerability, color="info" if task.answerability == "ANSWERABLE" else "warning", className="me-1"),
            dbc.Badge(task.question_type, color="secondary", className="me-1"),
        ]

        if task.multi_turn and task.multi_turn != "N/A":
            badges.append(dbc.Badge(task.multi_turn, color="primary", className="me-1"))

        if task.rewritten_query:
            badges.append(dbc.Badge("Query Rewritten", color="success", className="me-1"))

        return html.Div(badges, className="mb-2")

    def render_contexts(contexts: List[AgentMessageContext], task_id: str):
        """Renders the retrieved contexts in an accordion."""
        if not contexts:
            return html.Div("No contexts retrieved.", className="text-muted small")

        accordion_items = []
        for idx, ctx_item in enumerate(contexts):

            # Format Reference Judgements
            feedback_display = []
            if ctx_item.feedback:
                for fb in ctx_item.feedback:
                    color = "success" if fb.value == 'yes' else "danger"
                    feedback_display.append(dbc.Badge(f"{fb.annotator}: {fb.value}", color=color, className="me-1"))

            header = f"[{ctx_item.score:.4f}] {ctx_item.title or 'Doc ' + str(ctx_item.document_id)}"

            content = html.Div([
                html.Div(feedback_display, className="mb-2") if feedback_display else None,
                html.P(ctx_item.text, style={"whiteSpace": "pre-wrap", "fontSize": "0.9em"}),
                html.Hr(),
                html.Small(f"Doc ID: {ctx_item.document_id} | URL: {ctx_item.url}", className="text-muted"),
                html.Details([
                    html.Summary("Raw Query JSON"),
                    html.Pre(json.dumps(ctx_item.query, indent=2), className="bg-light p-2 small"),
                ]),
            ])

            accordion_items.append(
                dbc.AccordionItem(content, title=header, item_id=f"ctx-{task_id}-{idx}"),
            )

        return dbc.Accordion(accordion_items, start_collapsed=True, flush=True)

    def render_turn(task: GenerationTask):
        """Renders a single turn (Latest User Message -> Agent Response)."""

        # 1. Identify the latest user message from input history
        # In a generic conversation structure, the last message in 'input' allows us to see what triggered 'target'
        last_input_msg = task.input[-1] if task.input else None

        elements = []

        # Render User Message (if it exists and is user - usually index -1 is user in RAG)
        if last_input_msg and last_input_msg.speaker == 'user':
            elements.append(
                dbc.Row([
                     dbc.Col(
                         [
                            html.Div(
                                [
                                    html.Strong("User"),
                                    html.Div(last_input_msg.text),
                                ], style=CHAT_STYLE["user"],
                            ),
                         ], width=12,
                     ),
                ]),
            )

        # Render Agent Response (Target) + Metadata + Contexts

        # Prepare Context/Analysis Section
        analysis_card = dbc.Card(
            [
                dbc.CardHeader("RAG Analysis"),
                dbc.CardBody([
                    html.Div(
                        [
                            html.Strong("Rewritten Query: "),
                            html.Span(task.rewritten_query if task.rewritten_query else "None", className="text-info"),
                        ], className="mb-2 small",
                    ),
                    html.Strong("Retrieved Contexts:", className="small"),
                    render_contexts(task.contexts, task.task_id),
                ]),
            ], className="mb-3 shadow-sm",
        )

        elements.append(
            dbc.Row([
                dbc.Col(
                    [
                        html.Div(
                            [
                                html.Div(render_message_metadata(task), style=CHAT_STYLE["meta_box"]),
                                html.Strong("Agent"),
                                html.Div(task.target.text),
                                # Embed the analysis inside or below the agent bubble?
                                # Putting it below is cleaner for the "Chat" feel.
                            ], style=CHAT_STYLE["agent"],
                        ),

                        # Context Accordion rendered below the agent response
                        analysis_card,
                    ], width=12,
                ),
            ]),
        )

        return html.Div(elements, className="mb-4")

    # --- Layout ---

    app.layout = dbc.Container(
        [
            dbc.Row([
                dbc.Col(html.H2("RAG Conversation Viewer"), width=12, className="my-3 text-center"),
            ]),

            dbc.Row([
                # --- Sidebar ---
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader("Filters"),
                                dbc.CardBody([
                                    html.Label("Collection"),
                                    dcc.Dropdown(
                                        id='filter-collection',
                                        options=[{'label': c, 'value': c} for c in sorted(collection_options)],
                                        value=list(collection_options)[0] if collection_options else None,
                                    ),
                                    html.Br(),
                                    html.Label("Conversations"),
                                    dcc.Dropdown(
                                        id='conversation-selector',
                                        placeholder="Select ID...",
                                        optionHeight=50,
                                        # Options populated by callback
                                    ),
                                ]),
                            ], className="mb-3",
                        ),

                        html.Div(id='conversation-stats', className="small text-muted"),
                    ], width=3,
                ),

                # --- Main Content ---
                dbc.Col(
                    [
                        dbc.Card([
                            dbc.CardHeader(html.Div(id='chat-header', children="Select a conversation")),
                            dbc.CardBody(
                                html.Div(id='chat-content', style={"height": "75vh", "overflowY": "scroll"}),
                            ),
                        ]),
                    ], width=9,
                ),
            ]),
        ], fluid=True,
    )

    # --- Callbacks ---

    @app.callback(
        [
            Output('conversation-selector', 'options'),
            Output('conversation-selector', 'value'),
        ],
        [Input('filter-collection', 'value')],
    )
    def update_conversation_list(selected_collection):
        if not selected_collection:
            return [], None

        # Filter conversations that belong to this collection (checking the first task is sufficient usually)
        valid_ids = []
        for conv_id, tasks in conversations.items():
            if tasks[0].collection == selected_collection:
                # Add some info to the label
                turn_count = len(tasks)
                label = f"{conv_id} ({turn_count} turns)"
                valid_ids.append({'label': label, 'value': conv_id})

        valid_ids.sort(key=lambda x: x['label'])

        first_val = valid_ids[0]['value'] if valid_ids else None
        return valid_ids, first_val

    @app.callback(
        [
            Output('chat-content', 'children'),
            Output('chat-header', 'children'),
            Output('conversation-stats', 'children'),
        ],
        [Input('conversation-selector', 'value')],
    )
    def render_conversation(conv_id):
        if not conv_id or conv_id not in conversations:
            return html.Div("Please select a conversation."), "No Conversation Selected", ""

        tasks = conversations[conv_id]

        # Build the view
        chat_flow = []
        for task in tasks:
            chat_flow.append(render_turn(task))

        header = html.Div([
            html.Span(f"ID: {conv_id}", className="fw-bold"),
            dbc.Badge(tasks[0].dataset, color="light", text_color="dark", className="ms-2"),
        ])

        # Stats for sidebar
        stats = [
            html.P(f"Total Turns: {len(tasks)}"),
            html.P(f"Questions: {[t.question_type for t in tasks]}"),
            html.P(f"Stand-alone: {tasks[0].standalone_type}"),
        ]

        return chat_flow, header, stats

    # --- Run ---
    app.run(debug=True, host='0.0.0.0', port=8051)
