# Reliable NMT — Full Working Project (EACT / RG-CLD / EFC)

This repo implements a full pipeline for a fact-aware MT system:
- Entity-Aware Contrastive Fine-Tuning (EACT)
- Retrieval-Guided Constrained Lattice Decoding (RG-CLD) (local-KB based)
- Entity Factuality Calibration (EFC) module

Run locally (minimal):
1. Create venv:
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
2. Install:
   pip install -r requirements.txt
3. Download spacy model:
   python -m spacy download en_core_web_sm
4. Train small demo model (synthetic data included):
   python src/train.py --output_dir models/demo_en_hi
5. Run Streamlit UI:
   streamlit run app.py

Notes:
- The repo includes `data/parallel_small.csv` (tiny synthetic parallel pairs). This is to let you run everything end-to-end without external datasets.
- To use your own dataset, modify `src/train.py` to point to your train/val files or use Hugging Face datasets.
- For production deployment, follow Docker instructions in the repo or push to Streamlit Cloud / HF Spaces.
