import random
import torch
from flask import Flask, render_template, request, redirect, url_for
from huggingface_hub import login
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    AutoConfig
)
from peft import PeftModel, PeftConfig

# ----------------------------------------------------------------------------
# 1) Create the Flask app
# ----------------------------------------------------------------------------
app = Flask(__name__)


login("")

# ----------------------------------------------------------------------------
# 3) SETUP: Model-1 (Score Detection)
# ----------------------------------------------------------------------------
score_adapter_path = "scoringModel/"
score_peft_config = PeftConfig.from_pretrained(score_adapter_path)
num_labels = 1  # single regression-like output

base_config = AutoConfig.from_pretrained(score_peft_config.base_model_name_or_path)
base_config.num_labels = num_labels

score_tokenizer = AutoTokenizer.from_pretrained(score_peft_config.base_model_name_or_path)
score_base_model = AutoModelForSequenceClassification.from_pretrained(
    score_peft_config.base_model_name_or_path,
    config=base_config
)

score_model = PeftModel.from_pretrained(score_base_model, score_adapter_path)
score_model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
score_model.to(device)

def get_score(question_text, user_answer):
    """
    Run the 'score detection' model on combined question+answer text.
    Returns a float (logit or predicted value).
    """
    text = f"""Grade the following - Question: {question_text}
Answer: {user_answer}
"""
    inputs = score_tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = score_model(**inputs)
    logits = outputs.logits  # shape [batch, 1]
    return logits.squeeze().item()


# ----------------------------------------------------------------------------
# 4) SETUP: Model-2 (Feedback)
# ----------------------------------------------------------------------------
feedback_adapter_path = "feedbackModel/"
feedback_peft_config = PeftConfig.from_pretrained(feedback_adapter_path)

feedback_tokenizer = AutoTokenizer.from_pretrained(feedback_peft_config.base_model_name_or_path)
feedback_base_model = AutoModelForSeq2SeqLM.from_pretrained(feedback_peft_config.base_model_name_or_path)

feedback_model = PeftModel.from_pretrained(feedback_base_model, feedback_adapter_path)
feedback_model.eval()
feedback_model.to(device)

def get_feedback(question_text, user_answer):
    """
    Run the feedback model on a combined text of question+answer.
    Returns a generated feedback string.
    """
    input_text = f"""Grade the following - Question: {question_text}
Answer: {user_answer}
"""
    inputs = feedback_tokenizer(input_text, return_tensors="pt").to(device)
    with torch.no_grad():
        generated_tokens = feedback_model.generate(
            **inputs,
            max_new_tokens=50,
            num_beams=4,
            do_sample=False
        )
    return feedback_tokenizer.decode(generated_tokens[0], skip_special_tokens=True)


# ----------------------------------------------------------------------------
# 5) DATASET: Load random questions from HF "test_unseen_answers"
# ----------------------------------------------------------------------------
hf_dataset = load_dataset("Short-Answer-Feedback/saf_communication_networks_english")
test_ds = hf_dataset["test_unseen_answers"]

def get_random_questions(num=7):
    """
    Return `num` random question strings from 'test_unseen_answers'.
    """
    total_items = len(test_ds)
    random_indices = random.sample(range(total_items), num)
    questions = [test_ds[idx]["question"] for idx in random_indices]
    return questions

# We'll store 7 current questions globally
current_questions = get_random_questions(num=7)

# ----------------------------------------------------------------------------
# 6) Store user data for each question so it doesn't reset on navigation
# ----------------------------------------------------------------------------
# Key = question index (0..6), Value = dict with {user_answer, score, feedback}
answers_data = {}

# ----------------------------------------------------------------------------
# 7) FLASK ROUTES
# ----------------------------------------------------------------------------
@app.route('/')
def index():
    """Show the first question by default."""
    return redirect(url_for('question', qid=0))


@app.route('/refresh')
def refresh():
    """
    When user clicks 'Refresh', pick new random questions,
    CLEAR all saved answers/grades, and show question 0.
    """
    global current_questions, answers_data
    current_questions = get_random_questions(num=7)
    answers_data = {}  # Clear existing answers/grades
    return redirect(url_for('question', qid=0))


@app.route('/question/<int:qid>', methods=['GET', 'POST'])
def question(qid):
    """Show question at index qid, handle submission of answer."""
    if qid < 0 or qid >= len(current_questions):
        return "Question not found", 404

    question_text = current_questions[qid]

    # Check if we already have data for this question
    existing_data = answers_data.get(qid, {})
    score = existing_data.get('score')
    feedback = existing_data.get('feedback')
    user_answer = existing_data.get('user_answer', '')  # default empty

    if request.method == 'POST':
        user_answer = request.form.get('answer', '').strip()
        if not user_answer:
            # Empty or whitespace
            score = None
            feedback = "No response, please type an answer."
        else:
            # Re-run the models to update score & feedback
            raw_score = get_score(question_text, user_answer)
            score = round(raw_score, 2)
            feedback = get_feedback(question_text, user_answer)

        # Save the updated data back into answers_data
        answers_data[qid] = {
            'user_answer': user_answer,
            'score': score,
            'feedback': feedback
        }

    return render_template(
        'index.html',
        questions=current_questions,
        current_qid=qid,
        question_text=question_text,
        score=score,
        feedback=feedback,
        user_answer=user_answer
    )


# ----------------------------------------------------------------------------
# 8) Run the app
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
