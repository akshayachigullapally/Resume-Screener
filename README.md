# Resume Screening Web Application

This project is a complete full-stack Resume Screening Web Application.

<img src="Cover.png" alt="resume cover">

## Tech Stack

- **Backend:** Python + Flask
- **Frontend:** React (Vite) + Material UI
- **Machine Learning:** scikit-learn (TF-IDF + cosine similarity) + Sentence Transformers (BERT semantic similarity)
- **NLP:** NLTK (tokenization + stopword removal)
- **File Parsing:** pdfplumber (PDF), python-docx (DOCX)

## Features

- Upload multiple resumes in **PDF/DOCX** format
- Enter a **job description** and run one-click analysis
- Hybrid TF-IDF + BERT semantic similarity for each resume
- Skill extraction, skill match %, missing skills, and matched keywords
- Experience match estimation and overall resume strength scoring
- Candidate explanations ("why selected") and ranked suggestions panel
- Manual improvement advisor (paste resume text to get AI feedback)
- Analytics dashboard: top candidates, average scores, skills distribution
- Download ranked analysis as CSV
- Drag & drop upload, glassmorphism UI, Cover.png background served via backend
- Comprehensive validation for invalid files and empty job descriptions

## Project Structure

```text
Resume-Screening/
├── backend/
│   ├── app.py
│   ├── model/
│   │   ├── results.json
│   │   └── vectorizer.pkl
│   ├── uploads/
│   └── utils/
│       ├── __init__.py
│       ├── model_manager.py
│       ├── parser.py
│       ├── preprocessing.py
│       ├── similarity.py
│       └── suggestions.py
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   └── src/
│       ├── App.jsx
│       ├── main.jsx
│       └── styles.css
├── requirements.txt
└── README.md
```

## API Endpoints

### `POST /upload-resumes`

Uploads multiple resume files using multipart form-data field name `resumes`.

### `POST /analyze`

Analyzes uploaded resumes against a job description.

Returns:
- `overall_match` (hybrid TF-IDF + semantic %)
- `semantic_similarity` and `tfidf_score`
- `skill_match_percentage`
- `matched_skills`, `missing_skills`, `extracted_skills`
- `experience_match_percentage`
- `resume_strength_score`
- `suggestions` & `explanation`
- `analytics` block (top candidates, averages, skill distribution)

Request body (JSON):

```json
{
	"job_description": "We need a Python developer with Flask, SQL, and Docker experience...",
	"uploaded_files": ["<stored_name_from_upload>"]
}
```

### `POST /suggest-improvements`

Generate actionable suggestions by posting raw resume text plus a job description.

Body:

```json
{
	"resume_text": "Paste resume bullet points or summary...",
	"job_description": "JD text goes here"
}
```

Returns matched/missing skills, missing sections, and tailored advice.

### `GET /results`

Returns latest ranked results from `backend/model/results.json`.

### `GET /download-results`

Downloads ranked results as CSV.

### `GET /cover-image`

Serves `Cover.png` for frontend background.

## How It Works

1. Extract text from uploaded PDF/DOCX resumes.
2. Preprocess JD and resumes (lowercase, clean, tokenize, remove stopwords).
3. Train/load TF-IDF vectorizer from `resume_dataset.csv` and persist to `backend/model/vectorizer.pkl` (automatic on first run).
4. Generate TF-IDF vectors **and** Sentence Transformer embeddings.
5. Blend cosine scores into a single overall match signal.
6. Extract skills, compute match %, detect missing skills, parse experience years.
7. Build resume strength score, explanations, suggestions, and analytics.
8. Rank candidates, persist results, and expose CSV export/download endpoints.

## Step-by-Step Run Instructions

### 1) Backend Setup (Flask)

From project root:

```bash
python -m venv .venv
```

Activate virtual environment:

- **Windows (PowerShell):**

```bash
.\.venv\Scripts\Activate.ps1
```

- **Windows (CMD):**

```bash
.venv\Scripts\activate.bat
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run backend:

```bash
cd backend
python ../.venv/Scripts/python.exe app.py
```

Backend starts on: `http://127.0.0.1:5000`

---

### 2) Frontend Setup (React)

Open a new terminal in project root:

```bash
cd frontend
npm install
npm run dev
```

Frontend starts on: `http://127.0.0.1:5173`

---

### 3) Use the App

1. Open `http://127.0.0.1:5173`
2. Upload multiple resumes (`.pdf`, `.docx`)
3. Paste job description
4. Click **Analyze**
5. View ranked candidates, skill match, missing skills, and suggestions
6. Use **CSV** button to download results

## Notes

- Uploaded files are stored in `backend/uploads/`.
- Latest analysis output is stored in `backend/model/results.json`.
- Trained vectorizer is stored in `backend/model/vectorizer.pkl` and reused.
- If invalid files are uploaded, warnings are returned and shown in UI.
- If job description is empty, backend returns validation error.
