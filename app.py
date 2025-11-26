"""
Advanced Streamlit app: PDF -> Summarize -> Generate Questions & Answers
Features included:
- PDF text extraction (pdfplumber)
- Text summarization (transformers)
- Question generation (T5)
- Question answering (RoBERTa SQuAD2)
- Question classification (Who/What/When/Why/How/Others)
- Question quality filtering (sentence-transformers embeddings + cosine similarity)
- Download Q&A as .txt or .csv
- Voice support (gTTS -> audio playback / download)
- Utilities & placeholder for fine-tuning on educational PDFs (instructions + helper functions)

Save this file as `app.py` and run with `streamlit run app.py`.
Note: Some models are large and require time/ram to download. For local development test with small inputs.

"""

import streamlit as st
import pdfplumber
import nltk
from transformers import pipeline
from io import BytesIO
import pandas as pd
import numpy as np
import tempfile
import base64
from gtts import gTTS
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

# download punkt
nltk.download('punkt')

# --------------------- Helper functions ---------------------

def extract_text_from_pdf(file_like):
    text = []
    try:
        with pdfplumber.open(file_like) as pdf:
            for page in pdf.pages:
                content = page.extract_text()
                if content:
                    text.append(content)
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    return "\n".join(text)


def split_into_sentences(text):
    from nltk import sent_tokenize
    return sent_tokenize(text)


# Question classification (simple rule-based)
QUESTION_WORDS = {
    'who': 'Who',
    'what': 'What',
    'when': 'When',
    'where': 'Where',
    'why': 'Why',
    'how': 'How'
}


def classify_question(q):
    q_low = q.strip().lower()
    for w, label in QUESTION_WORDS.items():
        if q_low.startswith(w + ' ') or (' ' + w + ' ') in q_low:
            return label
    return 'Other'


# Filter duplicate / low-quality questions using embeddings
@st.cache_resource
def load_embedding_model():
    # small and fast sentence transformer model
    return SentenceTransformer('all-MiniLM-L6-v2')


def filter_questions(questions, threshold=0.8):
    """Remove duplicate/very-similar questions.
    threshold: cosine similarity above which two questions are considered duplicates.
    """
    if not questions:
        return []
    embed_model = load_embedding_model()
    embeddings = embed_model.encode(questions)
    keep = []
    for i, q in enumerate(questions):
        is_dup = False
        for j in keep:
            sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
            if sim >= threshold:
                is_dup = True
                break
        if not is_dup:
            keep.append(i)
    return [questions[i] for i in keep]


# Text to speech
def text_to_speech(text, lang='en'):
    tts = gTTS(text=text, lang=lang)
    bio = BytesIO()
    tts.write_to_fp(bio)
    bio.seek(0)
    return bio


# Utility to create downloadable text/csv
def make_downloadable_text(qa_pairs):
    buf = ""
    for i, (q, a, qtype) in enumerate(qa_pairs, start=1):
        buf += f"Q{i}: {q}\n"
        buf += f"Type: {qtype}\n"
        buf += f"A{i}: {a}\n\n"
    return buf


def make_downloadable_csv(qa_pairs):
    df = pd.DataFrame([{'question': q, 'answer': a, 'type': t} for q, a, t in qa_pairs])
    return df


# --------------------- Model loaders ---------------------
@st.cache_resource
def load_models():
    summarizer = pipeline('summarization', model='facebook/bart-large-cnn')
    qg = pipeline('text2text-generation', model='valhalla/t5-small-qg-prepend', tokenizer='valhalla/t5-small-qg-prepend')
    qa = pipeline('question-answering', model='deepset/roberta-base-squad2')
    return summarizer, qg, qa


# --------------------- Fine-tuning helper (placeholder) ---------------------
FINE_TUNE_HELP = """
Fine-tuning instructions (summary):
1. Prepare a dataset in SQuAD-style JSON (context, question, answers).
2. Use Hugging Face `transformers` Trainer API and a QA model (e.g., `deepset/roberta-base-squad2` or `distilbert-base-uncased-distilled-squad`).
3. You'll need a GPU and several hours depending on dataset size.

Example (pseudo):
- Load dataset -> tokenization -> DataCollator -> Trainer -> trainer.train()

This app provides `prepare_fine_tune_dataset(uploaded_pdf, generated_qa)` as a helper to produce training samples from your PDFs.
"""


def prepare_fine_tune_dataset(context_texts, qa_pairs):
    """Create SQuAD-like rows from contexts and generated QA pairs.
    context_texts: list of strings (paragraphs / contexts)
    qa_pairs: list of tuples (question, answer)
    Returns pandas DataFrame with columns: context, question, answer, answer_start
    """
    rows = []
    for context in context_texts:
        for q, a in qa_pairs:
            start = context.find(a)
            if start == -1:
                # fallback: choose 0 and leave answer_text as is
                start = 0
            rows.append({'context': context, 'question': q, 'answer': a, 'answer_start': start})
    return pd.DataFrame(rows)


def fast_summarize(text, summarizer, chunk_size=1200):
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    results = summarizer(chunks, max_length=120, min_length=30, do_sample=False)
    return " ".join([res['summary_text'] for res in results])


# --------------------- Streamlit UI ---------------------

def main():
    st.set_page_config(page_title='PDF QA Generator (Advanced)', layout='wide')
    st.title('📘 PDF -> Question & Answer Generator (Advanced)')

    st.sidebar.header('Settings')
    max_sentences = st.sidebar.number_input('Max sentences to process (for speed)', min_value=1, max_value=200, value=30)
    do_summarize = st.sidebar.checkbox('Summarize before QG', value=True)
    summary_ratio = st.sidebar.slider('Summary ratio (smaller => shorter)', 0.05, 0.5, 0.2)
    filter_dup = st.sidebar.checkbox('Filter duplicate/similar questions', value=True)
    similarity_threshold = st.sidebar.slider('Similarity threshold for filtering', 0.7, 0.95, 0.82)
    enable_tts = st.sidebar.checkbox('Enable Voice (TTS)', value=True)

    st.markdown('---')
    uploaded_file = st.file_uploader('Upload a PDF file', type=['pdf'])

    col1, col2 = st.columns([2,1])

    if uploaded_file is not None:
        with st.spinner('Extracting text from PDF...'):
            text = extract_text_from_pdf(uploaded_file)
        if not text.strip():
            st.error('No text found in the PDF.')
            return

        col1.subheader('Extracted Text (preview)')
        col1.text_area('Text Preview', value=text[:2000] + ('...' if len(text) > 2000 else ''), height=250)

        # Load models
        with st.spinner('Loading models (may take time on first run)...'):
            summarizer, qg, qa_model = load_models()

        # Summarize if requested

        if do_summarize:
            with st.spinner("Summarizing text (optimized)..."):
                # limit text for speed
                limited_text = " ".join(split_into_sentences(text)[:80])
                summarized_text = fast_summarize(limited_text, summarizer)
        else:
            summarized_text = text

        col1.subheader('Processed Text (used for Q&A)')
        col1.text_area('Processed Preview', value=summarized_text[:2000] + ('...' if len(summarized_text) > 2000 else ''), height=220)

        # Generate questions
        if st.button('Generate Questions & Answers'):
            with st.spinner('Generating questions...'):
                sentences = split_into_sentences(summarized_text)
                sentences = [s for s in sentences if len(s.split()) > 5]
                sentences = sentences[:max_sentences]

                generated_questions = []
                for s in sentences:
                    prompt = 'generate question: ' + s
                    try:
                        q = qg(prompt, max_length=64, truncation=True)[0]['generated_text']
                    except Exception as e:
                        st.warning(f'QG model error for sentence: {e}')
                        continue
                    generated_questions.append((q, s))

                # extract answers using QA model against the full processed text for better context
                qa_pairs = []  # list of (question, answer, type)
                for q, origin in generated_questions:
                    try:
                        res = qa_model(question=q, context=summarized_text)
                        answer = res.get('answer', '').strip()
                        if not answer:
                            # fallback to searching in origin sentence
                            res2 = qa_model(question=q, context=origin)
                            answer = res2.get('answer', '').strip()
                    except Exception as e:
                        st.warning(f'QA model error: {e}')
                        answer = ''
                    qtype = classify_question(q)
                    qa_pairs.append((q, answer, qtype))

                # Optional filtering of questions
                if filter_dup:
                    questions_only = [q for q, a, t in qa_pairs]
                    filtered = filter_questions(questions_only, threshold=similarity_threshold)
                    # recreate qa_pairs with filtered questions only
                    filtered_pairs = []
                    for q, a, t in qa_pairs:
                        if q in filtered:
                            filtered_pairs.append((q, a, t))
                    qa_pairs = filtered_pairs

                st.success('Generation complete!')

                # Display results
                for idx, (q, a, t) in enumerate(qa_pairs, start=1):
                    st.markdown(f'**Q{idx} ({t}):** {q}')
                    st.markdown(f'**A{idx}:** {a}')
                    if enable_tts and a:
                        try:
                            audio_bio = text_to_speech(a)
                            st.audio(audio_bio.read())
                            audio_bio.seek(0)
                            b64 = base64.b64encode(audio_bio.read()).decode()
                            href = f"data:audio/mp3;base64,{b64}"
                            st.markdown(f"[Download Answer Audio]({href})")
                        except Exception as e:
                            st.warning(f'TTS failed: {e}')
                    st.write('---')

                # Download options
                txt = make_downloadable_text(qa_pairs)
                csv_df = make_downloadable_csv(qa_pairs)

                st.download_button('Download Q&A as TXT', data=txt, file_name='qa_results.txt')
                csv_buf = csv_df.to_csv(index=False).encode('utf-8')
                st.download_button('Download Q&A as CSV', data=csv_buf, file_name='qa_results.csv', mime='text/csv')

                # Prepare fine-tune dataset helper and show option
                st.info('Fine-tuning helper')
                st.write('You can prepare a small SQuAD-style dataset from the processed contexts and generated Q&A pairs.')
                if st.button('Prepare fine-tune dataset (download CSV)'):
                    # create contexts as paragraphs chunk
                    contexts = [p for p in summarized_text.split('\n') if len(p.split()) > 20][:50]
                    # flatten qa_pairs to question-answer tuples for prepare
                    qa_simple = [(q,a) for q,a,t in qa_pairs]
                    df_ft = prepare_fine_tune_dataset(contexts, qa_simple)
                    st.download_button('Download fine-tune CSV', data=df_ft.to_csv(index=False).encode('utf-8'), file_name='fine_tune_dataset.csv')

                # show small analytics
                st.sidebar.subheader('Generation Summary')
                st.sidebar.write(f'Total sentences processed: {len(sentences)}')
                st.sidebar.write(f'Generated Q&A pairs: {len(qa_pairs)}')

    else:
        st.info('Upload a PDF to begin. You can enable/disable summarization or voice support from the sidebar.')

    st.markdown('---')
    st.header('Notes & Tips')
    st.write('- For large PDFs, enable summarization to focus on key points.')
    st.write('- Fine-tuning requires a GPU and knowledge of Hugging Face Trainer. This app only helps prepare training samples.')
    st.write('- Models may take time to download on first run. Use a machine with enough RAM. Consider using smaller models for quick testing.')

    # st.header('Fine-Tuning Guidance')
    # st.code(FINE_TUNE_HELP)


if __name__ == '__main__':
    main()
