# app.py
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
from src.decode import constrained_generate
from utils import extract_entities, apply_kb_on_output
from src.evaluate import factual_consistency_score, hallucination_rate

st.set_page_config(page_title="Reliable MT — Production App", layout="wide")
st.title("Reliable Neural MT — (EACT + RG-CLD + EFC) — Full App")

# Settings sidebar
with st.sidebar:
    st.header("Settings")
    target_choice = st.selectbox("Target language", ["hi", "fr"])
    model_choice = st.text_input("Model (HF repo)", "Helsinki-NLP/opus-mt-en-hi" if target_choice=="hi" else "Helsinki-NLP/opus-mt-en-fr")
    beam_size = st.slider("Beam size", 1, 8, 4)
    use_kb = st.checkbox("Enforce KB entity canonical forms (RG-CLD)", True)
    show_debug = st.checkbox("Show debug info", True)

@st.cache_resource
def load_model(name):
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSeq2SeqLM.from_pretrained(name)
    gen = pipeline("translation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
    return gen, tokenizer, model

gen, tokenizer, model = load_model(model_choice)

src = st.text_area("Source text (English)", height=200)
if st.button("Translate"):
    if not src.strip():
        st.warning("Write a source sentence.")
    else:
        with st.spinner("Translating..."):
            # run constrained_generate wrapper to apply KB substitutions
            device = "cuda" if torch.cuda.is_available() else "cpu"
            best, adjusted, changes = constrained_generate(model, tokenizer, src, target_lang=target_choice, beam_size=beam_size, device=device)
            if not use_kb:
                adjusted = best

            # Entities & metrics
            src_entities = extract_entities(src)
            out_entities = extract_entities(adjusted)
            fcs = factual_consistency_score([src], [adjusted])
            hr = hallucination_rate([src], [adjusted])

        st.subheader("Translation")
        st.write(adjusted)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Factual Consistency Score (heuristic)", f"{fcs:.3f}")
            st.write("Source Entities:")
            st.write(src_entities or "(none)")
        with col2:
            st.metric("Hallucination Rate (heuristic)", f"{hr:.3f}")
            st.write("Output Entities:")
            st.write(out_entities or "(none)")
            if changes:
                st.write("KB canonical substitutions applied:")
                st.json(changes)

        if show_debug:
            st.expander("Debug").write({
                "raw_model_output": best,
                "kb_adjusted_output": adjusted,
                "src_entities": src_entities,
                "out_entities": out_entities
            })
