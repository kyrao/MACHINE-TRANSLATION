Deployment options:

1) Streamlit Cloud:
   - Push the repo to GitHub
   - Create a new app on streamlit cloud and point to the repo and branch
   - Set main file to app.py; streamlit will install requirements

2) Hugging Face Spaces (Streamlit):
   - Create a new space (Streamlit)
   - Push the repo into the HF Space git
   - HF will build using requirements.txt and run streamlit

3) Docker:
   - Build: docker build -t reliable-mt .
   - Run: docker run -p 8501:8501 reliable-mt

Important:
- Large HF models are downloaded on first run; to avoid memory issues, fine-tune small models or use quantization for production.
