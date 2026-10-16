"""`hr-app`: a small web page to try a trained model.

    hr-app --checkpoint checkpoints/run/best.pt      # then open http://127.0.0.1:7860

1. Click "Random test employee" (an employee the model never saw) or fill the form.
2. Click "Predict": all four answers appear, next to the true values for test employees.
"""

import argparse
import random
from pathlib import Path
from typing import Any

from hr_gpt.config import CLASSIFICATION, DataConfig
from hr_gpt.data import frame_to_records, load_split
from hr_gpt.features import FEATURES, default_record, record_to_text
from hr_gpt.predict import Predictor, actual_values

TITLES = {
    "Attrition": "Will they leave?",
    "JobLevel": "Job level",
    "JobSatisfaction": "Job satisfaction",
    "MonthlyIncome": "Monthly income",
}

CSS = """
.gradio-container {max-width: 1100px !important; margin: auto;}
#results table {width: 100%; font-size: 1.05em;}
"""


def _show(value: Any) -> str:
    return f"${value:,.0f}" if isinstance(value, (int, float)) else str(value)


def results_table(result: dict[str, dict[str, Any]], actual: dict[str, Any] | None) -> str:
    """Markdown table: task | prediction (confidence) | actual (✅ / ❌ / how far off)."""
    lines = [
        "| | Prediction | Actual |" if actual else "| | Prediction |",
        "|---|---|---|" if actual else "|---|---|",
    ]
    for name, res in result.items():
        if "label" in res:
            pred = f"**{res['label']}** · {res['probabilities'][res['label']]:.0%}"
        else:
            pred = f"**{_show(res['value'])}**"
        row = f"| {TITLES.get(name, name)} | {pred} |"
        if actual and name in actual:
            truth = actual[name]
            if "label" in res:
                mark = "✅" if res["label"] == truth else "❌"
            else:
                mark = f"(off by {abs(res['value'] - truth) / truth:.0%})"
            row += f" {_show(truth)} {mark} |"
        lines.append(row)
    return "\n".join(lines)


def build_app(predictor: Predictor, test_records: list[dict[str, Any]]):
    import gradio as gr

    names = [f.name for f in FEATURES]
    defaults = default_record()

    def to_record(values: tuple) -> dict[str, Any]:
        return dict(zip(names, values))

    def predict(actual: dict | None, *values):
        try:
            record = to_record(values)
            result = predictor.predict(record)
        except ValueError as err:  # e.g. an empty field: say so, keep the rest as it is
            return (f"⚠️ **Cannot predict:** {err}", *(gr.update() for _ in range(4)))
        probs = {
            name: res["probabilities"]
            for name, res in result.items()
            if predictor.tasks[name].kind == CLASSIFICATION
        }
        return (
            results_table(result, actual),
            *(probs.get(n, {}) for n in ("Attrition", "JobLevel", "JobSatisfaction")),
            record_to_text(record),
        )

    def load_random():
        index = random.randrange(len(test_records))
        row = test_records[index]
        actual = actual_values(row, predictor.tasks)
        status = (
            f"Loaded **test employee #{index}** of {len(test_records)} (never seen in training). "
            "True answers are shown next to the predictions."
        )
        return (*(row[n] for n in names), actual, status)

    def reset():
        return (
            *(defaults[n] for n in names),
            None,
            "Default employee. Edit any field, then Predict.",
        )

    with gr.Blocks(title="HR-GPT") as demo:
        gr.Markdown(
            "# HR-GPT\n"
            "Describe an employee, get four predictions from one GPT-2 model: "
            "**attrition**, **job level**, **job satisfaction** and **monthly income**."
        )
        actual_state = gr.State(None)
        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                with gr.Row():
                    random_btn = gr.Button("🎲 Random test employee", variant="secondary")
                    reset_btn = gr.Button("↺ Reset", variant="secondary")
                status = gr.Markdown("Default employee. Edit any field, then Predict.")
                inputs = []
                with gr.Tabs():
                    for group in dict.fromkeys(f.group for f in FEATURES):
                        with gr.Tab(group), gr.Row():
                            for f in (f for f in FEATURES if f.group == group):
                                if f.is_categorical:
                                    comp = gr.Dropdown(
                                        choices=[(lbl, val) for val, lbl in f.choices.items()],
                                        value=f.default,
                                        label=f.label,
                                        min_width=180,
                                        elem_id=f"in-{f.name}",
                                    )
                                else:
                                    comp = gr.Number(
                                        value=f.default,
                                        label=f.label,
                                        minimum=f.minimum,
                                        maximum=f.maximum,
                                        precision=0,
                                        min_width=180,
                                        elem_id=f"in-{f.name}",
                                    )
                                inputs.append(comp)
            with gr.Column(scale=2):
                predict_btn = gr.Button("Predict", variant="primary", size="lg")
                results = gr.Markdown("Click **Predict** to see the results.", elem_id="results")
                with gr.Accordion("Probabilities", open=False):
                    attrition = gr.Label(label="Attrition")
                    level = gr.Label(label="Job level", num_top_classes=5)
                    satisfaction = gr.Label(label="Job satisfaction")
                with gr.Accordion("What the model reads", open=False):
                    sentence = gr.Textbox(show_label=False, lines=6, interactive=False)

        outputs = [results, attrition, level, satisfaction, sentence]
        predict_btn.click(predict, [actual_state, *inputs], outputs)
        random_btn.click(load_random, None, [*inputs, actual_state, status]).then(
            predict, [actual_state, *inputs], outputs
        )
        reset_btn.click(reset, None, [*inputs, actual_state, status])
        # Editing a loaded test employee makes the true answers meaningless: drop them
        for comp in inputs:
            comp.input(
                lambda: (None, "Edited by hand: no true answers to compare."),
                None,
                [actual_state, status],
            )
    demo.css = CSS
    return demo


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Web app to try a trained HR-GPT model.")
    parser.add_argument("--checkpoint", default="checkpoints/run/best.pt")
    parser.add_argument("--data-dir", type=Path, default=DataConfig().data_dir)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="public gradio.live link")
    args = parser.parse_args(argv)

    predictor = Predictor.from_checkpoint(args.checkpoint)
    test = frame_to_records(load_split(DataConfig(data_dir=args.data_dir), "test"))
    build_app(predictor, test).launch(
        server_name=args.host, server_port=args.port, share=args.share
    )


if __name__ == "__main__":
    main()
