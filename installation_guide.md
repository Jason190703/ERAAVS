# ERAAVS Installation Guide

This guide provides detailed instructions for installing and running the End-to-end Requirement Analysis and Verification System (ERAAVS).

## Dependencies

The following packages are required to run ERAAVS:

```
streamlit>=1.25.0
pandas>=2.0.0
matplotlib>=3.7.0
nltk>=3.8.0
numpy>=1.24.0
spacy>=3.6.0
scikit-learn>=1.3.0
pdfplumber>=0.10.0
PyPDF2>=3.0.0
reportlab>=4.0.0
Pillow>=10.0.0
```

## Installation Steps

### Option 1: Using Replit (Recommended for Quick Start)

1. Fork this Replit project
2. Click the "Run" button
3. The application will automatically install dependencies and start

### Option 2: Local Installation

1. Clone the repository
   ```bash
   git clone [repository-url]
   cd eraavs
   ```

2. Create a virtual environment (recommended)
   ```bash
   # Using venv
   python -m venv venv
   
   # Activate on Windows
   venv\Scripts\activate
   
   # Activate on macOS/Linux
   source venv/bin/activate
   ```

3. Install required packages
   ```bash
   pip install streamlit pandas matplotlib nltk numpy spacy scikit-learn pdfplumber PyPDF2 reportlab Pillow
   ```

4. Download required NLTK data
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
   ```

5. Download required spaCy model
   ```bash
   python -m spacy download en_core_web_sm
   ```

6. Run the application
   ```bash
   streamlit run app.py
   ```

7. Open your browser and navigate to `http://localhost:8501`

### Option 3: Using Conda (Recommended for Python 3.13+ Compatibility Issues)

If you're having dependency issues with newer Python versions:

1. Install Miniconda from https://docs.conda.io/projects/miniconda/
2. Create a conda environment with Python 3.10
   ```bash
   conda create -n eraavs_env python=3.10
   conda activate eraavs_env
   ```

3. Install dependencies with conda and pip
   ```bash
   conda install -c conda-forge streamlit pandas matplotlib nltk numpy spacy scikit-learn
   pip install pdfplumber PyPDF2 reportlab
   ```

4. Download required data
   ```bash
   python -m nltk.downloader punkt wordnet stopwords
   python -m spacy download en_core_web_sm
   ```

5. Run the application
   ```bash
   streamlit run app.py
   ```

## Troubleshooting

### Compilation Issues
If you encounter compilation errors with spaCy or NumPy:
1. Try using Python 3.10 instead of newer versions
2. Install Visual C++ Build Tools on Windows
3. Use the conda installation method above

### Missing NLTK Data
If you see NLTK errors:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

### Missing spaCy Model
If you see errors about missing spaCy models:
```bash
python -m spacy download en_core_web_sm
```

## Deployment Options

### Streamlit Cloud
1. Create a free account at https://streamlit.io/cloud
2. Connect your GitHub repository
3. Deploy the application with a few clicks

### Other Cloud Platforms
ERAAVS can be deployed on:
- Heroku
- AWS Elastic Beanstalk
- Google Cloud Run
- Azure App Service

Follow the respective platform's documentation for Python web application deployment.