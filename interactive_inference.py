import gradio as gr
from model.model import HRGPTInterface

# parameters 
CKPT_PATH = "./checkpoints/hr_gpt_training/best_model_step_10000.pt.pt"

def create_gradio_interface():
    """Create the Gradio interface"""
    hr_interface = HRGPTInterface(ckpt_path=CKPT_PATH)

    with gr.Blocks(title="HR-GPT: Employee Analytics AI", theme=gr.themes.Soft()) as demo:

        # Centered header & description
        gr.HTML(
            """
            <div style="text-align:center; max-width:900px; margin: 0 auto 16px;">
              <h1 style="margin-bottom:8px;">🏢 HR-GPT: Employee Analytics AI</h1>
              <p style="margin:0 0 10px 0; font-size:16px;">
                <b>Professional AI-powered HR analytics for employee insights</b>
              </p>
              <div style="display:inline-block; text-align:left; font-size:15px;">
                <ul style="margin:0; padding-left:18px;">
                  <li><b>Employee Attrition</b>: Will the employee leave?</li>
                  <li><b>Job Level</b>: What is the appropriate job level?</li>
                  <li><b>Monthly Income</b>: What should be the salary?</li>
                  <li><b>Job Satisfaction</b>: How satisfied is the employee?</li>
                </ul>
              </div>
            </div>
            """
        )

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### 📋 Employee Information")

                # Sample selector (instant apply on change)
                with gr.Row():
                    with gr.Column(scale=3):
                        sample_dropdown = gr.Dropdown(
                            choices=[f"Sample {i+1}" for i in range(len(hr_interface.test_data))],
                            label="📁 Load Test Sample",
                            value=None
                        )

                # Single status box; we will update value & visibility with gr.update(...)
                load_info = gr.Textbox(label="Load Status", interactive=False, visible=False)

                # Create input fields dynamically
                input_components = {}

                with gr.Tabs():
                    with gr.TabItem("👤 Personal Info"):
                        with gr.Row():
                            input_components["Age"] = gr.Number(
                                label="Age", value=30, minimum=18, maximum=70
                            )
                            input_components["Gender"] = gr.Dropdown(
                                choices=["male", "female"], label="Gender", value="male"
                            )
                            input_components["MaritalStatus"] = gr.Dropdown(
                                choices=["single", "married", "divorced"], 
                                label="Marital Status", value="married"
                            )
                        with gr.Row():
                            input_components["Education"] = gr.Dropdown(
                                choices=["Below College", "College", "Bachelor", "Master", "Doctor"],
                                label="Education Level", value="Bachelor"
                            )
                            input_components["EducationField"] = gr.Dropdown(
                                choices=["life sciences", "medical", "marketing", "technical degree", "other", "human resources"],
                                label="Education Field", value="life sciences"
                            )
                        input_components["DistanceFromHome"] = gr.Number(
                            label="Distance from Home (km)", value=10, minimum=1, maximum=50
                        )

                    with gr.TabItem("💼 Job Details"):
                        with gr.Row():
                            input_components["Department"] = gr.Dropdown(
                                choices=["sales", "research & development", "human resources"],
                                label="Department", value="research & development"
                            )
                            input_components["JobRole"] = gr.Dropdown(
                                choices=["sales executive", "research scientist", "laboratory technician", 
                                       "manufacturing director", "healthcare representative", "manager", 
                                       "sales representative", "research director", "human resources"],
                                label="Job Role", value="research scientist"
                            )
                        with gr.Row():
                            input_components["BusinessTravel"] = gr.Dropdown(
                                choices=["non-travel", "travel_rarely", "travel_frequently"],
                                label="Business Travel", value="travel_rarely"
                            )
                            input_components["OverTime"] = gr.Dropdown(
                                choices=["no", "yes"], label="Works Overtime", value="no"
                            )
                        with gr.Row():
                            input_components["TotalWorkingYears"] = gr.Number(
                                label="Total Working Years", value=8, minimum=0, maximum=40
                            )
                            input_components["YearsAtCompany"] = gr.Number(
                                label="Years at Company", value=5, minimum=0, maximum=40
                            )
                            input_components["YearsInCurrentRole"] = gr.Number(
                                label="Years in Current Role", value=3, minimum=0, maximum=20
                            )
                        with gr.Row():
                            input_components["YearsSinceLastPromotion"] = gr.Number(
                                label="Years Since Last Promotion", value=1, minimum=0, maximum=15
                            )
                            input_components["YearsWithCurrManager"] = gr.Number(
                                label="Years with Current Manager", value=2, minimum=0, maximum=15
                            )
                            input_components["NumCompaniesWorked"] = gr.Number(
                                label="Number of Companies Worked", value=2, minimum=0, maximum=10
                            )

                    with gr.TabItem("💰 Compensation"):
                        with gr.Row():
                            input_components["HourlyRate"] = gr.Number(
                                label="Hourly Rate ($)", value=50, minimum=20, maximum=100
                            )
                            input_components["DailyRate"] = gr.Number(
                                label="Daily Rate ($)", value=800, minimum=100, maximum=2000
                            )
                            input_components["MonthlyRate"] = gr.Number(
                                label="Monthly Rate ($)", value=15000, minimum=2000, maximum=30000
                            )
                        with gr.Row():
                            input_components["PercentSalaryHike"] = gr.Number(
                                label="Percent Salary Hike (%)", value=15, minimum=10, maximum=25
                            )
                            input_components["StockOptionLevel"] = gr.Number(
                                label="Stock Option Level", value=1, minimum=0, maximum=3
                            )

                    with gr.TabItem("📊 Performance & Satisfaction"):
                        with gr.Row():
                            input_components["PerformanceRating"] = gr.Dropdown(
                                choices=["Low", "Good", "Excellent", "Outstanding"],
                                label="Performance Rating", value="Excellent"
                            )
                            input_components["JobInvolvement"] = gr.Dropdown(
                                choices=["Low", "Medium", "High", "Very High"],
                                label="Job Involvement", value="High"
                            )
                        with gr.Row():
                            input_components["EnvironmentSatisfaction"] = gr.Dropdown(
                                choices=["Low", "Medium", "High", "Very High"],
                                label="Environment Satisfaction", value="High"
                            )
                            input_components["RelationshipSatisfaction"] = gr.Dropdown(
                                choices=["Low", "Medium", "High", "Very High"],
                                label="Relationship Satisfaction", value="High"
                            )
                        with gr.Row():
                            input_components["WorkLifeBalance"] = gr.Dropdown(
                                choices=["Bad", "Good", "Better", "Best"],
                                label="Work-Life Balance", value="Better"
                            )
                            input_components["TrainingTimesLastYear"] = gr.Number(
                                label="Training Times Last Year", value=3, minimum=0, maximum=10
                            )

            with gr.Column(scale=1):
                gr.Markdown("### 🔮 AI Predictions")

                task_selection = gr.Radio(
                    choices=[
                        ("Employee Attrition 📤", "Attrition"),
                        ("Job Level 📊", "JobLevel"),
                        ("Monthly Income 💰", "MonthlyIncome"),
                        ("Job Satisfaction 😊", "JobSatisfaction"),
                    ],
                    label="Select Prediction Task",
                    value="Attrition",
                )

                predict_btn = gr.Button("🚀 Make Prediction", variant="primary", size="lg")

                with gr.Group():
                    gr.Markdown("### Results")
                    classification_result = gr.Label(label="Prediction", visible=False)
                    regression_result = gr.Textbox(label="Predicted Value", visible=False)
                    confidence_display = gr.Textbox(label="Confidence / Status", visible=False)

        # ===== Event handlers =====

        def load_sample(sample_name):
            if sample_name is None:
                # Keep current values; just show a message
                return (
                    *[gr.update() for _ in input_components],  # no change
                    gr.update(value="Please select a sample", visible=True),
                )
            try:
                sample_idx = int(sample_name.split()[-1]) - 1
                inputs, info = hr_interface.load_test_sample(sample_idx)

                # Return values for all input components in creation order
                updates = []
                for field in input_components.keys():
                    updates.append(inputs.get(field, hr_interface.input_fields[field].get("value", "")))

                # Set status message visible
                return (*updates, gr.update(value=info, visible=True))

            except Exception as e:
                # Don’t change inputs; show error
                return (
                    *[gr.update() for _ in input_components],
                    gr.update(value=f"Error loading sample: {e}", visible=True),
                )

        def make_prediction(task_name, *input_values):
            try:
                inputs = dict(zip(input_components.keys(), input_values))
                result = hr_interface.predict_task(task_name, inputs)

                if "error" in result:
                    # Hide value widgets, show error in status/confidence
                    return (
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(value=f"Error: {result['error']}", visible=True),
                    )

                task_spec = hr_interface.config.tasks[task_name]

                if task_spec.task_type in ("binary", "multiclass"):
                    # Show class probabilities and confidence
                    return (
                        gr.update(value=result["all_probabilities"], visible=True),
                        gr.update(visible=False),
                        gr.update(value=f"{result['confidence']:.1%}", visible=True),
                    )
                else:
                    # Regression
                    return (
                        gr.update(visible=False),
                        gr.update(value=result["prediction"], visible=True),
                        gr.update(visible=False),
                    )

            except Exception as e:
                return (
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(value=f"Exception: {e}", visible=True),
                )

        # Instant apply on change (no Load button needed)
        sample_dropdown.change(
            fn=load_sample,
            inputs=[sample_dropdown],
            outputs=list(input_components.values()) + [load_info],
        )

        # Predict click
        predict_btn.click(
            fn=make_prediction,
            inputs=[task_selection] + list(input_components.values()),
            outputs=[classification_result, regression_result, confidence_display],
        )

        # Reveal/hide the status box depending on content
        load_info.change(
            fn=lambda x: gr.update(visible=bool(x)),
            inputs=[load_info],
            outputs=[load_info],
        )

    return demo


# run the Gradio app
demo = create_gradio_interface()
demo.launch(
    server_name="127.0.0.1",
    server_port=7866,
    share=False,
    debug=True,
)
