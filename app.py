from flask import Flask, request, render_template_string
from readability import Document
from transformers import BartForConditionalGeneration, BartTokenizer
from utils import split_text_into_chunks
import re

app = Flask(__name__)

# Load the DistilBART model and tokenizer
model_name = "sshleifer/distilbart-cnn-12-6"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Simple HTML for the input form
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

@app.route('/')
def index():
    return render_template_string(HTML_FORM)

@app.route('/', methods=['POST'])
def summarize():
    text = request.form['text']

    # Use the split_text_into_chunks function
    chunks = split_text_into_chunks(text, tokenizer)

    summaries = []
    for chunk in chunks:
        # Generate summary for each chunk
        inputs = tokenizer([chunk], max_length=1024, truncation=True, return_tensors="pt")
        summary_ids = model.generate(inputs["input_ids"], num_beams=4, min_length=30, max_length=150, early_stopping=True)
        summary_chunk = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary_chunk)

    # Combine summaries if there are multiple chunks
    summary = " ".join(summaries)

    # Calculate word count
    original_word_count = len(text.split())
    summary_word_count = len(summary.split())

    # Calculate readability scores using readability-lxml
    original_html = f"<html><body>{text}</body></html>"
    summary_html = f"<html><body>{summary}</body></html>"

    original_doc = Document(original_html)
    summary_doc = Document(summary_html)

    try:
        original_readable_text = original_doc.summary()
        original_readability = "Processed" if original_readable_text else "Could not process"
    except Exception as e:
        original_readability = f"Error: {e}"

    try:
        summary_readable_text = summary_doc.summary()
        summary_readability = "Processed" if summary_readable_text else "Could not process"
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

if __name__ == '__main__':
    # This part is for local testing and will be handled differently for deployment
    # app.run(debug=True)
    pass
