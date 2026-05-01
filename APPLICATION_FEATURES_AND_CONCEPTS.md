# Resume Screening Web Application - Features & ML Concepts

## 📋 Project Overview

A full-stack web application that uses machine learning and NLP to automatically screen and rank resumes against job descriptions. The application combines TF-IDF vectorization with BERT-based semantic similarity to provide intelligent resume matching with detailed analytics.

---

## 🎯 Key Features

### 1. **Resume Upload & Management**
- Multi-file upload support (PDF and DOCX formats)
- Drag-and-drop interface
- File validation and secure storage
- Unique file naming with UUID for organization
- Support for multiple resume formats with automatic text extraction

### 2. **Resume Analysis**
- One-click analysis against job descriptions
- Hybrid similarity scoring (TF-IDF + BERT semantic)
- Overall match percentage calculation
- Resume strength scoring based on multiple factors
- Experience matching with year extraction
- Keyword overlap detection

### 3. **Skill Extraction & Matching**
- Curated database of 24+ technical skills (Python, Java, React, AWS, Docker, etc.)
- Automatic skill extraction from resumes
- Job-required vs. candidate-available skill comparison
- Skill match percentage calculation
- Missing skills identification
- Skill suggestions for resume improvement

### 4. **Experience Matching**
- Year of experience extraction using regex patterns
- Required vs. candidate experience comparison
- Experience match percentage calculation
- Support for various year formats ("3 years", "5+ yrs", "2 yr", etc.)

### 5. **Resume Section Analysis**
- Automatic detection and extraction of resume sections:
  - Skills/Competencies
  - Work Experience/Employment History
  - Projects/Portfolio
  - Education/Academic Background
  - Achievements/Accomplishments
- Section presence scoring
- Section-specific metrics and insights
- Resume structure quality assessment

### 6. **Role Fit Prediction**
- 6 predefined role profiles:
  - Backend Developer
  - Frontend Developer
  - Full Stack Developer
  - ML Engineer
  - Data Scientist
  - DevOps Engineer
- Skill-based role matching
- Role-specific keyword analysis
- Best fit role determination

### 7. **Candidate Explanations**
- Automated narrative explanations for why each candidate was selected
- Highlights for matched skills
- Experience alignment details
- Overall fit reasoning

### 8. **Improvement Suggestions**
- AI-generated actionable suggestions for resume improvement
- Missing skills recommendations
- Section gap analysis
- Metrics and quantification recommendations
- Project portfolio suggestions

### 9. **Analytics Dashboard**
- Top candidates ranking
- Average match scores across all candidates
- Skill distribution analysis
- Common missing skills across candidates
- Candidate performance metrics
- Visual analytics data

### 10. **CSV Export**
- Download ranked analysis results as CSV
- Includes all scoring metrics and analysis details
- Ready for further analysis or sharing

### 11. **Manual Improvement Advisor**
- Paste raw resume text to get AI feedback
- Identifies improvement areas
- Suggests specific additions or modifications
- Job-description-aware recommendations

### 12. **User Interface**
- React + Vite frontend
- Material UI components
- Glassmorphism design aesthetic
- Responsive layout
- Intuitive job description input
- Real-time file upload feedback
- Results ranking visualization

---

## 🤖 Machine Learning & NLP Concepts

### **1. TF-IDF (Term Frequency-Inverse Document Frequency)**

**What it does:**
- Converts text into numerical vectors representing term importance
- Higher values = more important terms for distinguishing resumes

**Implementation Details:**
- Uses `scikit-learn`'s `TfidfVectorizer`
- Configuration: 7000 maximum features, unigrams + bigrams (1,2-grams)
- Trained on resume_dataset.csv corpus
- Vectorizer is pickled and cached for reuse

**Formula:**
```
TF-IDF(t, d) = TF(t, d) × IDF(t)
where TF = term frequency, IDF = inverse document frequency
```

**Use Case:**
- Calculates similarity between job description and resumes
- Fast, deterministic, interpretable baseline

---

### **2. BERT Semantic Similarity**

**What it does:**
- Uses pre-trained sentence transformers to understand semantic meaning
- Compares contextual meaning rather than just keyword matching
- Captures synonym and related concept relationships

**Implementation Details:**
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Generates 384-dimensional embeddings for each text
- Normalized embeddings for cosine similarity
- Uses `torch` as backend with GPU optimization support

**Process:**
1. Encodes job description into embedding
2. Encodes all resumes into embeddings
3. Computes cosine similarity between vectors
4. Returns scores normalized to 0-1 range

**Use Case:**
- Captures semantic meaning (e.g., "experience with APIs" matches "REST endpoints")
- More flexible than keyword matching
- Handles synonyms and paraphrasing

---

### **3. Hybrid Similarity Scoring**

**Concept:**
- Combines TF-IDF and BERT scores with weighted averaging
- Default weights: 60% TF-IDF + 40% BERT semantic

**Formula:**
```
Overall Match % = (TF-IDF Score × 0.6) + (Semantic Score × 0.4)
```

**Advantage:**
- Balances keyword-level matching with semantic understanding
- More robust than either method alone

---

### **4. Natural Language Processing (NLP)**

#### **Text Preprocessing Pipeline**

**Steps:**
1. **Lowercasing** - Normalize case for consistent matching
2. **Special Character Removal** - Remove non-alphabetic characters
3. **Tokenization** - Break text into words using NLTK punkt tokenizer
4. **Stopword Removal** - Filter out common words (the, is, a, etc.)
5. **Alphabetic Filtering** - Keep only alphabetic tokens

**Tools:**
- NLTK (Natural Language Toolkit)
- Punkt tokenizer for word segmentation
- English stopwords corpus

**Purpose:**
- Reduces noise in text analysis
- Focuses on meaningful terms
- Improves similarity calculations

---

### **5. Feature Extraction**

#### **A. Skill Extraction**
- **Curated Database**: 24+ technical keywords
  - Languages: Python, Java, C++, JavaScript
  - Frameworks: React, Node, Flask, Django
  - Cloud: AWS, Azure, GCP
  - DevOps: Docker, Kubernetes, Git, Linux
  - AI/ML: Machine Learning, Deep Learning, NLP, TensorFlow, PyTorch
  - Databases: SQL

- **Matching Algorithm**: Word boundary regex matching
- **Output**: Sorted, deduplicated set of extracted skills
- **Deterministic**: Same text always produces same skills

#### **B. Experience Extraction**
- **Regex Pattern**: `(\d+)\s*\+?\s*(?:years|year|yrs|yr)`
- **Variations Supported**:
  - "5 years"
  - "3+ years"
  - "2 yrs"
  - "10 yr"
- **Output**: Maximum years found (handles multiple mentions)

#### **C. Keyword Overlap**
- **Process**:
  1. Tokenize job description and resume
  2. Find common tokens (2+ characters minimum)
  3. Count occurrences in resume
  4. Return top 12 most frequent overlaps
- **Purpose**: Shows explicit keyword matches between job and resume

#### **D. Section Detection**
- **Regex-based Detection**:
  - Normalizes heading text
  - Matches against predefined aliases
  - Sections detected: Skills, Experience, Projects, Education

- **Section Examples**:
  - Skills: "Technical Skills", "Core Skills", "Competencies"
  - Experience: "Work Experience", "Employment History"
  - Projects: "Projects", "Case Studies", "Portfolio"
  - Education: "Education", "Qualifications"

#### **E. Quantified Achievements Detection**
- **Pattern Recognition**: `\b\d+%|\$\d+|\b\d+\b`
- **Identifies**:
  - Percentage improvements ("improved by 20%")
  - Dollar amounts ("saved $500K")
  - Raw numbers ("trained 100 models")

---

### **6. Similarity Metrics**

#### **Cosine Similarity**
- **Formula**: `cos(θ) = (A · B) / (||A|| × ||B||)`
- **Range**: 0 to 1 (0 = completely different, 1 = identical)
- **Used For**:
  - TF-IDF vector comparison
  - BERT embedding comparison
  - Fast, scalable similarity calculation

---

### **7. Experience Matching**

**Algorithm:**
```
Experience Match % = (Candidate Years / Required Years) × 100
Capped at 100% maximum
```

**Edge Cases:**
- Required years = 0: Returns 100%
- Candidate has more than required: Returns 100%
- Used in resume strength scoring

---

### **8. Resume Strength Scoring**

**Components:**
1. **Similarity Score** - Overall TF-IDF + Semantic match (45% weight)
2. **Skill Match** - Percentage of required skills present (30% weight)
3. **Experience Match** - Years of experience alignment (15% weight)
4. **Structure Score** - Presence of key resume sections (5% weight)

**Section Scoring:**
- Checks for: Experience, Education, Projects, Skills, Certification
- `Structure Score = (Sections Found / 5) × 100`

**Quantified Achievement Bonus:**
- 100 points if metrics/achievements detected
- 40 points if none detected
- Rewards resume detail and specificity

**Formula:**
```
Strength Score = (Similarity × 0.45) + (Skills × 0.3) + 
                 (Experience × 0.15) + (Structure × 0.05) +
                 (Achievement × Weight)
```

---

### **9. Weighted Score Aggregation**

**Scoring Breakdown:**
- **Skills Score** - 40% weight
- **Experience Score** - 30% weight
- **Projects Score** - 20% weight
- **Semantic Similarity** - 10% weight

**Overall Score Calculation:**
```
Overall = (Skills × 0.4) + (Experience × 0.3) + 
          (Projects × 0.2) + (Semantic × 0.1)
```

Returns detailed breakdown with individual component scores.

---

### **10. Section-Based Scoring**

**For Each Section (Skills, Experience, Projects):**
1. **Content Length** - Word count in section (normalized)
2. **Skill Match** - Percentage of required skills in section
3. **Achievement Density** - Frequency of action verbs and metrics
4. **Relevance** - Match between section content and job description

**Action Verbs Tracked:**
- built, designed, implemented, deployed, optimized, trained, architected, launched

---

### **11. Role Fit Prediction**

**Role Profiles Defined:**
1. **Backend Developer** - Python, Java, C++, Node, SQL, Docker, AWS, Git
2. **Frontend Developer** - JavaScript, React, CSS, HTML
3. **Full Stack Developer** - JavaScript, React, Node, Python, SQL
4. **ML Engineer** - Python, Machine Learning, Deep Learning, TensorFlow, PyTorch, AWS
5. **Data Scientist** - Python, Data Science, NLP, SQL, Machine Learning
6. **DevOps Engineer** - Linux, Docker, Kubernetes, AWS, Azure, Git

**Role Keywords Tracked:**
- Backend: API, Microservice, Database
- Frontend: UI, UX, Responsive, Web App
- Full Stack: End-to-end, Full Stack
- ML Engineer: Model, Training, Inference, ML
- Data Scientist: Analysis, Insights, Statistics
- DevOps: Pipeline, CI/CD, Infrastructure, Automation

**Matching Algorithm:**
1. Count skill matches against profile requirements
2. Count keyword matches in experience sections
3. Weighted combination determines best fit role

---

### **12. Natural Language Generation**

#### **Candidate Explanation**
Automatically generates narrative text explaining:
- Why candidate matches the role
- Key strengths relative to job requirements
- Experience alignment
- Unique selling points

#### **Resume Suggestions**
AI-generated recommendations including:
- Missing skill additions
- Section improvements
- Quantified metric suggestions
- Portfolio/project hints

#### **Improvement Plan**
Structured feedback including:
- Section gap analysis
- Actionable suggestions (max 6)
- Priority ordering

---

## 📊 Data Processing Pipeline

```
Resume (PDF/DOCX)
    ↓
Text Extraction (pdfplumber / python-docx)
    ↓
Preprocessing (NLTK tokenization + stopword removal)
    ↓
Feature Extraction
├── Skills extraction
├── Experience extraction
├── Section detection
└── Keyword overlap
    ↓
Similarity Computation
├── TF-IDF vectorization & cosine similarity
└── BERT semantic similarity
    ↓
Score Aggregation
├── Weighted overall score
├── Resume strength score
└── Role fit prediction
    ↓
Analysis Output
├── Ranked results
├── Suggestions
├── Explanations
└── Analytics
```

---

## 🧠 ML Technologies Used

| Technology | Purpose | Implementation |
|---|---|---|
| **scikit-learn** | TF-IDF vectorization + cosine similarity | TfidfVectorizer, cosine_similarity |
| **Sentence Transformers** | BERT semantic embeddings | all-MiniLM-L6-v2 model |
| **PyTorch** | Neural network backend for embeddings | Automatic GPU acceleration |
| **NLTK** | Tokenization + stopword removal | Punkt tokenizer, English stopwords |
| **pdfplumber** | PDF text extraction | Page-by-page extraction |
| **python-docx** | DOCX file parsing | Paragraph extraction |
| **Regex** | Pattern matching for experience/metrics | Python `re` module |
| **Pickle** | Model serialization | TF-IDF vectorizer caching |

---

## 🔄 API Endpoints

### `POST /upload-resumes`
- **Purpose**: Store resumes for later analysis
- **Input**: Multipart form-data with files
- **Output**: Stored file names and metadata
- **Validation**: File type checking, size limits

### `POST /analyze`
- **Purpose**: Analyze uploaded resumes against job description
- **Input**: Job description + file references
- **Processing**: 
  - Extract text from files
  - Run ML pipeline (TF-IDF + BERT)
  - Calculate all metrics
  - Generate suggestions
- **Output**: Ranked results, analytics, detailed scores
- **Caching**: TF-IDF vectorizer cached globally

### `POST /suggest-improvements`
- **Purpose**: Get improvement advice for a resume
- **Input**: Raw resume text + job description
- **Processing**: Skill extraction, gap analysis, suggestion generation
- **Output**: Improvement plan, actionable suggestions

### `GET /download-csv`
- **Purpose**: Export analysis results
- **Output**: CSV file with all metrics

### `GET /`
- **Purpose**: Health check
- **Output**: Service status

---

## 📈 Analytics Provided

1. **Top Candidates** - Ranked by overall match percentage
2. **Average Scores** - Mean TF-IDF, semantic, skill, experience metrics
3. **Skill Distribution** - Frequency of each skill across candidates
4. **Missing Skills** - Skills required but rarely found
5. **Candidate Count Metrics** - Total analyzed, parsing errors
6. **Score Distribution** - Quartiles and percentiles

---

## 🛡️ Validation & Error Handling

- File type validation (PDF/DOCX only)
- Empty file handling
- Preprocessing failure detection
- Missing job description validation
- Resume parsing error tracking
- Invalid UTF-8 character handling
- Secure filename sanitization
- Duplicate file detection

---

## 🚀 Performance Features

- **Vectorizer Caching**: TF-IDF model loaded once, reused
- **Model Caching**: BERT model cached after first use
- **Global Caches**: Result and analytics caching
- **Efficient Filtering**: Stopword + character removal
- **Batch Processing**: Multiple resumes processed together
- **Regex Compilation**: Patterns compiled for reuse

---

## 💾 Data Storage

- **Uploaded Resumes**: Stored in `backend/uploads/`
- **Models**: TF-IDF vectorizer pickled in `backend/model/`
- **Results**: JSON persistence in `backend/model/results.json`
- **Training Corpus**: CSV dataset in `resume_dataset.csv`

---

## 🎨 Frontend Features

- React + Vite for fast development and bundling
- Material UI components for professional interface
- CSS glassmorphism for modern aesthetic
- Drag-and-drop file upload
- Real-time validation feedback
- Results visualization with ranking
- CSV download button
- Responsive design for mobile support

---

## 📚 Training Data

- Resume dataset CSV with real resumes
- 7000 TF-IDF features trained from corpus
- Bigrams (2-grams) included for better phrase matching
- Training happens automatically on first analysis

---

## 🔗 Dependencies Summary

```
Backend:
- Flask (REST API)
- Flask-CORS (Cross-origin requests)
- scikit-learn (ML/TF-IDF)
- NLTK (NLP)
- pdfplumber (PDF parsing)
- python-docx (DOCX parsing)
- sentence-transformers (BERT embeddings)
- torch (Neural networks)

Frontend:
- React (UI framework)
- Vite (Build tool)
- Material UI (Components)
```

---

## 🎯 Use Cases

1. **HR Screening** - Automatically rank candidates for a position
2. **Applicant Tracking** - Filter incoming resumes by job fit
3. **Resume Optimization** - Get suggestions to improve match for target roles
4. **Skills Gap Analysis** - Identify missing skills in candidate pool
5. **Role Fit Assessment** - Determine which roles candidates best fit
6. **Candidate Comparison** - Rank candidates objectively with metrics

---

## 🔍 Key Innovations

1. **Hybrid Matching** - Combines TF-IDF (fast, keyword-based) with BERT (semantic, meaning-based)
2. **Section-Based Analysis** - Breaks down resume by logical sections for detailed scoring
3. **Role Prediction** - Suggests best-fit roles beyond the job description
4. **Explanations** - Provides human-readable "why selected" narratives
5. **Improvement Advisor** - Interactive suggestions for resume enhancement
6. **Analytics** - Aggregated insights across candidate pool

---

## 📝 Created: March 31, 2026

This document provides a comprehensive overview of all features and machine learning concepts implemented in the Resume Screening Web Application.
