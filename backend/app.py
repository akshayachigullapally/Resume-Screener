import csv
import json
import uuid
from collections import Counter
from io import BytesIO, StringIO
from pathlib import Path

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

from utils.model_manager import load_or_train_vectorizer
from utils.parser import extract_text_from_file
from utils.preprocessing import preprocess_text
from utils.similarity import (
    compute_resume_strength_score,
    compute_similarity_scores,
    compute_semantic_similarity_scores,
    extract_keyword_overlap,
    extract_years_of_experience,
    get_experience_match_percentage,
)
from utils.skills import extract_skills, get_skill_match_details
from utils.sections import extract_section_texts, predict_role_fit, score_resume_sections
from utils.scoring import compute_weighted_overall_score
from utils.suggestions import (
    build_candidate_explanation,
    generate_improvement_plan,
    generate_resume_suggestions,
)


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
MODEL_DIR = BASE_DIR / "model"
RESULTS_FILE = MODEL_DIR / "results.json"
VECTORIZER_FILE = MODEL_DIR / "vectorizer.pkl"
DATASET_FILE = BASE_DIR.parent / "resume_dataset.csv"
ALLOWED_EXTENSIONS = {".pdf", ".docx"}

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def load_previous_results():
    if RESULTS_FILE.exists():
        try:
            with open(RESULTS_FILE, "r", encoding="utf-8") as file:
                payload = json.load(file)
                if isinstance(payload, dict):
                    return payload.get("results", []), payload.get("analytics", {})
                if isinstance(payload, list):
                    return payload, {}
        except (json.JSONDecodeError, OSError):
            return [], {}
    return [], {}


def save_results(results, analytics):
    with open(RESULTS_FILE, "w", encoding="utf-8") as file:
        json.dump(
            {
                "results": results,
                "analytics": analytics,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )


RESULTS_CACHE, ANALYTICS_CACHE = load_previous_results()
ADVISOR_TEXT_CACHE = {}
VECTORIZER = None


def get_vectorizer():
    global VECTORIZER
    if VECTORIZER is None:
        VECTORIZER = load_or_train_vectorizer(DATASET_FILE, VECTORIZER_FILE)
    return VECTORIZER


def clean_candidate_name(file_name: str) -> str:
    stem = Path(file_name).stem
    if "_" in stem and len(stem.split("_", 1)[0]) == 32:
        return stem.split("_", 1)[1]
    return stem

app = Flask(__name__)
CORS(
    app,
    resources={
        r"/*": {
            "origins": [
                "http://localhost:5173",
                "http://127.0.0.1:5173",
            ]
        }
    },
)


@app.get("/")
def health_check():
    return jsonify({"message": "Resume Screening API is running."})


@app.post("/upload-resumes")
def upload_resumes():
    if "resumes" not in request.files:
        return jsonify({"error": "No files provided. Use field name 'resumes'."}), 400

    files = request.files.getlist("resumes")
    if not files:
        return jsonify({"error": "Empty file list."}), 400

    uploaded = []
    errors = []

    for file in files:
        filename = file.filename or ""
        if not filename:
            errors.append("Skipped an unnamed file.")
            continue

        if not allowed_file(filename):
            errors.append(f"Invalid file type: {filename}. Allowed: PDF, DOCX")
            continue

        safe_name = secure_filename(filename)
        unique_name = f"{uuid.uuid4().hex}_{safe_name}"
        save_path = UPLOAD_DIR / unique_name
        file.save(save_path)

        uploaded.append(
            {
                "original_name": filename,
                "stored_name": unique_name,
                "path": str(save_path),
            }
        )

    if not uploaded:
        return jsonify({"error": "No valid files uploaded.", "details": errors}), 400

    return jsonify({"message": "Upload complete.", "uploaded": uploaded, "errors": errors}), 200


@app.post("/analyze")
def analyze_resumes():
    global RESULTS_CACHE, ANALYTICS_CACHE, ADVISOR_TEXT_CACHE

    payload = request.get_json(silent=True) if request.is_json else {}
    payload = payload or {}

    job_description = payload.get("job_description") or request.form.get("job_description", "")
    job_description = job_description.strip()

    if not job_description:
        return jsonify({"error": "Job description is required."}), 400

    incoming_files = request.files.getlist("resumes") if "resumes" in request.files else []
    new_file_names: list[str] = []
    for incoming in incoming_files:
        incoming_name = incoming.filename or ""
        if not incoming_name or not allowed_file(incoming_name):
            continue
        unique_name = f"{uuid.uuid4().hex}_{secure_filename(incoming_name)}"
        incoming.save(UPLOAD_DIR / unique_name)
        new_file_names.append(unique_name)

    uploaded_files = payload.get("uploaded_files") if isinstance(payload.get("uploaded_files"), list) else []
    requested_files = [name for name in uploaded_files if isinstance(name, str)] + new_file_names

    stored_files: list[Path] = []
    seen: set[str] = set()
    for file_name in requested_files:
        safe_name = Path(file_name).name
        if safe_name in seen:
            continue
        seen.add(safe_name)
        file_path = UPLOAD_DIR / safe_name
        if file_path.exists() and file_path.suffix.lower() in ALLOWED_EXTENSIONS:
            stored_files.append(file_path)

    if not stored_files:
        return jsonify({"error": "No valid resumes selected. Provide 'uploaded_files' from /upload-resumes."}), 400

    processed_jd = preprocess_text(job_description)
    if not processed_jd:
        return jsonify({"error": "Job description has no meaningful content after preprocessing."}), 400

    resumes_data = []
    parsing_errors = []
    job_skills = extract_skills(job_description)
    required_years = extract_years_of_experience(job_description)

    try:
        vectorizer = get_vectorizer()
    except Exception as exc:
        return jsonify({"error": f"Failed to initialize vectorizer: {exc}"}), 500

    ADVISOR_TEXT_CACHE = {}

    for file_path in stored_files:
        if file_path.suffix.lower() not in ALLOWED_EXTENSIONS:
            continue

        try:
            raw_text = extract_text_from_file(str(file_path))
            cleaned_resume_text = preprocess_text(raw_text)

            if not cleaned_resume_text:
                parsing_errors.append(f"No meaningful text found in {file_path.name}.")
                continue

            resumes_data.append(
                {
                    "filename": file_path.name,
                    "raw_text": raw_text,
                    "resume_text": cleaned_resume_text,
                }
            )
            ADVISOR_TEXT_CACHE[file_path.name] = raw_text
        except Exception as exc:
            parsing_errors.append(f"Failed to parse {file_path.name}: {exc}")

    if not resumes_data:
        return jsonify({"error": "Unable to parse any uploaded resumes.", "details": parsing_errors}), 400

    resume_texts = [item["resume_text"] for item in resumes_data]
    raw_texts = [item["raw_text"] for item in resumes_data]
    tfidf_scores = compute_similarity_scores(vectorizer, processed_jd, resume_texts)
    semantic_scores = compute_semantic_similarity_scores(job_description, raw_texts)

    ranked_results = []
    for item, tfidf_score, semantic_score in zip(
        resumes_data, tfidf_scores, semantic_scores
    ):
        candidate_name = clean_candidate_name(item["filename"])
        resume_skills = extract_skills(item["raw_text"])
        skill_details = get_skill_match_details(job_skills, resume_skills)

        candidate_years = extract_years_of_experience(item["raw_text"])
        experience_match = get_experience_match_percentage(required_years, candidate_years)

        similarity_percentage = round(float(tfidf_score) * 100, 2)
        semantic_percentage = round(float(semantic_score) * 100, 2)
        suggestions = generate_resume_suggestions(
            job_description,
            item["raw_text"],
            skill_details["missing_skills"],
        )
        resume_sections = extract_section_texts(item["raw_text"])
        section_scores = score_resume_sections(
            resume_sections,
            skill_details["skill_match_percentage"],
            experience_match,
            resume_skills,
        )
        role_fit = predict_role_fit(resume_skills, resume_sections, item["raw_text"])
        score_breakdown = compute_weighted_overall_score(section_scores, semantic_percentage)
        overall_match_percentage = score_breakdown["overall_score"]
        strength_score = compute_resume_strength_score(
            overall_match_percentage,
            skill_details["skill_match_percentage"],
            experience_match,
            item["raw_text"],
        )
        explanation = build_candidate_explanation(
            candidate_name,
            overall_match_percentage,
            semantic_score,
            skill_details["matched_skills"],
            skill_details["missing_skills"],
            experience_match,
        )
        matched_keywords = extract_keyword_overlap(processed_jd, item["resume_text"])

        ranked_results.append(
            {
                "candidate_name": candidate_name,
                "file_name": item["filename"],
                "tfidf_score": similarity_percentage,
                "semantic_similarity": semantic_percentage,
                "overall_match": overall_match_percentage,
                "resume_strength_score": strength_score,
                "job_required_experience_years": required_years,
                "candidate_experience_years": candidate_years,
                "experience_match_percentage": experience_match,
                "extracted_skills": resume_skills,
                "matched_skills": skill_details["matched_skills"],
                "missing_skills": skill_details["missing_skills"],
                "skill_match_percentage": skill_details["skill_match_percentage"],
                "matched_keywords": matched_keywords,
                "suggestions": suggestions,
                "explanation": explanation,
                "section_scores": section_scores,
                "recommended_role": role_fit,
                "score_breakdown": score_breakdown,
            }
        )

    ranked_results.sort(
        key=lambda record: (
            record["overall_match"],
            record["skill_match_percentage"],
            record["resume_strength_score"],
        ),
        reverse=True,
    )

    for index, record in enumerate(ranked_results, start=1):
        record["rank"] = index

    skill_counter = Counter()
    overall_total = 0.0
    semantic_total = 0.0
    for record in ranked_results:
        skill_counter.update(record["matched_skills"])
        overall_total += record["overall_match"]
        semantic_total += record["semantic_similarity"]

    analytics = {
        "top_candidates": [
            {
                "candidate_name": result["candidate_name"],
                "overall_match": result["overall_match"],
                "file_name": result["file_name"],
            }
            for result in ranked_results[:3]
        ],
        "average_overall_score": round(overall_total / len(ranked_results), 2),
        "average_semantic_score": round(semantic_total / len(ranked_results), 2),
        "skill_distribution": [
            {"skill": skill, "count": count}
            for skill, count in skill_counter.most_common()
        ],
        "total_candidates": len(ranked_results),
    }

    RESULTS_CACHE = ranked_results
    ANALYTICS_CACHE = analytics
    save_results(ranked_results, analytics)

    return (
        jsonify(
            {
                "message": "Analysis completed.",
                "total_candidates": len(ranked_results),
                "job_description_skills": job_skills,
                "results": ranked_results,
                "analytics": analytics,
                "warnings": parsing_errors,
            }
        ),
        200,
    )


@app.get("/results")
def get_results():
    if not RESULTS_CACHE:
        return jsonify({"message": "No analysis results found.", "results": [], "analytics": {}}), 200
    return jsonify({"results": RESULTS_CACHE, "analytics": ANALYTICS_CACHE}), 200


@app.post("/suggest-improvements")
def suggest_improvements():
    payload = request.get_json(silent=True) or {}
    resume_text = (payload.get("resume_text") or "").strip()
    candidate_file = (payload.get("candidate_file") or "").strip()
    job_description = (payload.get("job_description") or "").strip()

    if not job_description:
        return jsonify({"error": "Job description is required."}), 400

    if not resume_text and candidate_file:
        resume_text = (ADVISOR_TEXT_CACHE.get(candidate_file) or "").strip()

    if not resume_text:
        return jsonify(
            {
                "error": "Provide resume_text or candidate_file referencing an analyzed resume."
            }
        ), 400

    jd_skills = extract_skills(job_description)
    resume_skills = extract_skills(resume_text)
    skill_details = get_skill_match_details(jd_skills, resume_skills)
    plan = generate_improvement_plan(resume_text, job_description, skill_details["missing_skills"])

    return (
        jsonify(
            {
                "matched_skills": skill_details["matched_skills"],
                "missing_skills": skill_details["missing_skills"],
                "suggestions": plan["suggestions"],
                "missing_sections": plan["missing_sections"],
            }
        ),
        200,
    )


@app.get("/download-results")
def download_results():
    if not RESULTS_CACHE:
        return jsonify({"error": "No results available. Run analysis first."}), 400

    csv_buffer = StringIO()
    field_names = [
        "rank",
        "candidate_name",
        "file_name",
        "overall_match",
        "semantic_similarity",
        "tfidf_score",
        "skill_match_percentage",
        "resume_strength_score",
        "experience_match_percentage",
        "matched_skills",
        "missing_skills",
        "matched_keywords",
    ]
    writer = csv.DictWriter(csv_buffer, fieldnames=field_names)
    writer.writeheader()

    for index, row in enumerate(RESULTS_CACHE, start=1):
        writer.writerow(
            {
                "rank": index,
                "candidate_name": row.get("candidate_name", ""),
                "file_name": row.get("file_name", ""),
                "overall_match": row.get("overall_match", ""),
                "semantic_similarity": row.get("semantic_similarity", ""),
                "tfidf_score": row.get("tfidf_score", ""),
                "skill_match_percentage": row.get("skill_match_percentage", ""),
                "resume_strength_score": row.get("resume_strength_score", ""),
                "experience_match_percentage": row.get("experience_match_percentage", ""),
                "matched_skills": ", ".join(row.get("matched_skills", [])),
                "missing_skills": ", ".join(row.get("missing_skills", [])),
                "matched_keywords": ", ".join(row.get("matched_keywords", [])),
            }
        )

    output_file = BytesIO(csv_buffer.getvalue().encode("utf-8"))
    csv_buffer.close()
    return send_file(
        output_file,
        mimetype="text/csv",
        as_attachment=True,
        download_name="resume_screening_results.csv",
    )


@app.get("/cover-image")
def get_cover_image():
    cover_path = BASE_DIR.parent / "Cover.png"
    if not cover_path.exists():
        return jsonify({"error": "Cover image not found."}), 404
    return send_file(cover_path, mimetype="image/png")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
