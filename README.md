## PEFT Finetuning for subjective grading

![Logo](media/logo.png)

In this project we explore the possibility of customized grading assistants that can provide both numeric scores and feedbacks instantly. This is an effort to decrease the workload of teaching faculty and TAs. We focus on grading subjective/writing based questions mainly as there are many tools that exist for multiple-choice and coding questions. For this project we have finetuned 2 SLMs - Facebook's Roberta-base and BART as they are lightweight and can run locally as well as with compute restrained environments. We also propose a custom dataset curation method that can be used to generated fine-tuning datasets for different subject/domain requirements as required by the teaching faculty.

## Fine-tuning details

Aligning the models to the particular domain of interest using finetuning is called model adaptation. For this we need a dataset with labels and inputs. The labels are what the language model will predict. Essentially, this follows a similar objective to the pre-training task, which is to predict the next token given the previous context, just in this case the next token will be a numeric value for the scoring-model and the feedback model will predict the next words given the previous context (same as the original transformer model). 

For the purpose of demonstration, we have finetuned the models on the Short-Answer-Feedback (SAF) communication networks dataset [dataset link](https://huggingface.co/datasets/Short-Answer-Feedback/saf_communication_networks_english). We also provide a framework to generate custom datasets using state-of the art LLMs in the data curation section.

**PEFT Finetuning**
Parameter-Efficient Fine-Tuning (PEFT) is a method that enables fine-tuning large pre-trained models by updating only a small subset of parameters, significantly reducing computational cost and memory usage. Instead of modifying all weights, PEFT techniques like LoRA (Low-Rank Adaptation), Prefix Tuning, or Adapter Tuning insert lightweight modules or train low-rank matrices within the model, while keeping the core model frozen. This makes PEFT ideal for deploying large language models in resource-constrained settings or for quickly adapting models to new tasks with limited data, without the need for full retraining.

**Configuration**
The configuration for PEFT training chosen was the following:
Based on the two notebooks you provided, here’s a theoretical explanation of the **PEFT configurations** used for each model:

- **1. BART Model – Feedback Only**
- - **Model Type**: Encoder-Decoder (Seq2Seq)
- - **PEFT Method Used**: LoRA (Low-Rank Adaptation)
- - **Target Modules**: Mostly linear layers within attention blocks (`q`, `v`, `k`, `out` projections) of the encoder and decoder.
- - **LoRA Rank (r)**: Set to a low value (commonly 4 or 8), representing the rank of the low-rank decomposition.
- - **LoRA Alpha**: A scaling factor applied to the LoRA updates to control their contribution relative to the frozen weights.
- - **Dropout**: LoRA-specific dropout (typically around 0.05) to regularize the fine-tuned parameters.
- - **Bias**: Often set to “none”, meaning only LoRA-injected weights are trainable while biases remain frozen.
- - **Task Type**: Marked as `SEQ_2_SEQ_LM`, indicating a sequence generation task (generating feedback text).


- **2. RoBERTa Model – Score Only**
- - **Model Type**: Encoder-only (Transformer)
- - **PEFT Method Used**: LoRA
- - **Target Modules**: Typically attention-related layers (`query`, `value`, `key`, and `output` projections) inside RoBERTa's self-attention blocks.
- - **LoRA Rank (r)**: Set similar to BART (e.g., 4 or 8), to inject low-rank matrices into selected layers.
- - **LoRA Alpha**: Again used to scale the updates during training.
- - **Dropout**: Regularization of the LoRA paths to avoid overfitting.
- - **Bias**: Set to “none” to maintain the minimal-parameter strategy.
- - **Task Type**: `SEQ_CLS` or `REGRESSION`, meaning it performs score prediction on a continuous scale.


## Dataset curation
The finetuning has been for the SAF dataset but this finetuning architecture is meant for widespread application into multiple domains. the framework is as follows:

- collect a question answer dataset for the subject of interest. For example, if you want to create a math dataset, you can use the GSM8K benchmark dataset, that has highschool level math word problems.
- Shuffle the dataset to avoid bias.
- Take around 20% of the dataset and pass each example into a capable LLM (suggested: Llama 70B, GPT-4o, etc), provide a system prompt to change the answer by maintaining the structure but making calculation/minor errors.
- Then take another 10% of original dataset (make sure it does not intersect with the previous 20%) and provide a prompt to the model to keep the same answer but meddle with some of the steps/details.
- Finally we pass the modified dataset to the LLM and ask it to give scores and grading based on a RAG/system prompt based rubric. This is the most important part and determines the quality of fine-tuning later on.
  

We have built a small UI to demonstrate the finetuned grader assistant.

## Instructions on how to run the application
1) First clone the repository in your virtual/base environment<pre> ```git clone https://github.com/mani2001/FineTuningProjectNLP.git``` </pre>
2) Move into the folder <pre> ```cd FineTuningProjectNLP``` </pre>
3) Open the [Link](https://drive.google.com/drive/folders/1Ux9I9T5cI-MoK9orVutRuKr8PrR4dqZy?usp=sharing) and download both the folders (scoringModel and feedbackModel) - The files were slightly bigger than the accepted GitHub size.
4) Copy and paste/move the folders into the GraderApp folder.
5) Go to huggingface and create an access token by following this: [Huggingface access token](https://huggingface.co/docs/hub/en/security-tokens)
6) In GraderApp/app.py paste the token within the quotes of the login function <pre> ```login("")``` </pre>
7) Run app.py (from within the GraderApp folder)
8) go to the localhost url provided by Django (within your terminal)
9) Test the app!
    

