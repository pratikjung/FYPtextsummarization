from flask import Flask, render_template, request
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch

app = Flask(__name__)

model_name = "google/pegasus-large"

tokenizer = PegasusTokenizer.from_pretrained(model_name)


device = "cuda" if torch.cuda.is_available() else "cpu"

model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/text-summarization', methods=["POST"])
def summarize():

    if request.method == "POST":

        inputtext = request.form["inputtext_"]

        input_text = "summarize: " + inputtext

        tokenized_text = tokenizer.encode(input_text, return_tensors='pt', max_length=512).to(device)
        summary_ = model.generate(tokenized_text, min_length=90, max_length=300)
        summary = tokenizer.decode(summary_[0], skip_special_tokens=True)



    return render_template("output.html", data = {"summary": summary})

if __name__ == '__main__': # It Allows You to Execute Code When the File Runs as a Script
    app.run(port=5002)


'''model_path = "Users\Ripple\Desktop\FYP\Text-summarization-flask-huggingface-main"
model_name_or_path = "transformersbook/pegasus-samsum"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = PegasusTokenizer.from_pretrained(model_name_or_path, cache_dir=model_path)
model = PegasusForConditionalGeneration.from_pretrained(model_name_or_path, cache_dir=model_path).to(device)'''