from __future__ import annotations

from functools import lru_cache

import gradio as gr

from config import SETTINGS, load_eval_queries
from rag.pipeline import build_pipeline
from rag.types import QueryRequest
from ui.charts import build_metrics_plot
from ui.components import render_persona_card, render_placeholder_card


@lru_cache(maxsize=1)
def get_pipeline():
    return build_pipeline()


def run_ui_pipeline(
    query: str,
    style_strength: float,
    top_k: int,
    debate_mode: bool,
    show_trace: bool,
):
    query = (query or "").strip()
    if not query:
        empty = render_placeholder_card(
            "Awaiting Question",
            "Ask one question to compare grounded persona answers side by side.",
        )
        return (
            empty,
            empty,
            empty,
            build_metrics_plot([]),
            gr.update(value=None, visible=show_trace),
        )

    result = get_pipeline().run(
        QueryRequest(
            query=query,
            style_strength=float(style_strength),
            top_k=int(top_k),
            debate_mode=bool(debate_mode),
        )
    )
    cards_by_id = {card["personality_id"]: card for card in result["cards"]}

    return (
        render_persona_card(cards_by_id["elon_musk"]),
        render_persona_card(cards_by_id["robert_greene"]),
        render_persona_card(cards_by_id["steve_jobs"]),
        build_metrics_plot(result["cards"]),
        gr.update(value=result["trace"], visible=show_trace),
    )


examples = [[item["query"]] for item in load_eval_queries()]


with gr.Blocks(
    title="PolyMind",
    css_paths=[str(SETTINGS.ui_theme_path)],
) as demo:
    gr.Markdown(
        """
        # PolyMind Comparison Lab
        Ask one question and compare grounded answers across three personalities.

        This build already includes retrieval, routing, trace logging, and a bootstrap local fallback.
        If you add persona books and local 7B model directories later, the same pipeline will pick them up.
        """
    )

    with gr.Row():
        query = gr.Textbox(
            lines=3,
            placeholder="What does success require in the long term?",
            label="Question",
        )
        ask_btn = gr.Button("Ask", elem_classes=["primary-action"])

    gr.Examples(examples=examples, inputs=query)

    with gr.Row():
        style_strength = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=SETTINGS.default_style_strength,
            label="Style Strength",
        )
        top_k = gr.Slider(
            minimum=1,
            maximum=6,
            value=SETTINGS.default_top_k,
            step=1,
            label="Top K",
        )
        debate_mode = gr.Checkbox(
            label="Debate Mode",
            value=False,
            info="Scaffolded now; standard comparison remains active in this build.",
        )
        show_trace = gr.Checkbox(label="Show Trace", value=False)

    with gr.Row(equal_height=True):
        elon_card = gr.Markdown(
            value=render_placeholder_card(
                "Elon Musk",
                "Mission-driven, first-principles, engineering-heavy answers will appear here.",
            ),
            elem_classes=["persona-card", "elon"],
        )
        robert_card = gr.Markdown(
            value=render_placeholder_card(
                "Robert Greene",
                "Strategic, power-aware, long-game answers will appear here.",
            ),
            elem_classes=["persona-card", "robert"],
        )
        steve_card = gr.Markdown(
            value=render_placeholder_card(
                "Steve Jobs",
                "Taste-driven, focused, product-minded answers will appear here.",
            ),
            elem_classes=["persona-card", "steve"],
        )

    metrics_plot = gr.Plot(value=build_metrics_plot([]), label="Comparison Metrics")
    trace_json = gr.JSON(label="Trace", visible=False)

    ask_btn.click(
        fn=run_ui_pipeline,
        inputs=[query, style_strength, top_k, debate_mode, show_trace],
        outputs=[elon_card, robert_card, steve_card, metrics_plot, trace_json],
    )
    query.submit(
        fn=run_ui_pipeline,
        inputs=[query, style_strength, top_k, debate_mode, show_trace],
        outputs=[elon_card, robert_card, steve_card, metrics_plot, trace_json],
    )

demo.queue()
