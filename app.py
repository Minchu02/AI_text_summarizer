from flask import Flask, request, render_template_string
from readability import Document
from transformers import BartForConditionalGeneration, BartTokenizer
from utils import split_text_into_chunks

app = Flask(__name__)

model_name = "philschmid/bart-large-cnn-samsum"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

HTML_FORM = """
<!doctype html>
<html>
<head><title>AI Writer with Text Summarization</title></head>
<body>
    <h1>Text Summarization</h1>
    <form method="post">
        <textarea name="text" rows="20" cols="80" placeholder="Enter text to summarize"></textarea><br>
        <input type="submit" value="Summarize">
    </form>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form['text']
        chunks = split_text_into_chunks(text, tokenizer)
        summaries = []
        for chunk in chunks:
            inputs = tokenizer([chunk], max_length=1024, truncation=True, return_tensors="pt")
            summary_ids = model.generate(inputs["input_ids"], num_beams=4, min_length=30, max_length=150, early_stopping=True)
            summary_chunk = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            summaries.append(summary_chunk)

        summary = " ".join(summaries)
        original_word_count = len(text.split())
        summary_word_count = len(summary.split())

        original_doc = Document(f"<html><body>{text}</body></html>")
        summary_doc = Document(f"<html><body>{summary}</body></html>")

        try:
            original_readability = "Processed" if original_doc.summary() else "Could not process"
        except Exception as e:
            original_readability = f"Error: {e}"

        try:
            summary_readability = "Processed" if summary_doc.summary() else "Could not process"
        except Exception as e:
            summary_readability = f"Error: {e}"

        return render_template_string(HTML_FORM + f"""
        <h2>Original Text:</h2>
        <p>{text}</p>
        <p>Word Count: {original_word_count}</p>
        <p>Readability Score: {original_readability}</p>

        <h2>Summary:</h2>
        <p>{summary}</p>
        <p>Word Count: {summary_word_count}</p>
        <p>Readability Score: {summary_readability}</p>
        """)

    return render_template_string(HTML_FORM)
