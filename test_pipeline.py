"""
Comprehensive End-to-End Pipeline Test Script.
Tests:
1. PyMuPDF text and contact extraction on sample PDFs.
2. Skill dictionary matching.
3. TF-IDF + Logistic Regression classification inference.
4. Top 5 job recommendations with cosine similarity, match scores, and missing skills.
5. SQLite database logging and retrieval.
"""

import os
import sys

# Ensure root in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from resume_parser import ResumeParser
from modules.classifier import ResumeClassifier
from job_recommender import JobRecommender
from modules.database import log_resume_scan, fetch_scan_history, fetch_all_jobs


def run_tests():
    print("=" * 70)
    print("STARTING COMPLETE END-TO-END SYSTEM TEST")
    print("=" * 70)

    # 1. Initialize components
    parser = ResumeParser()
    classifier = ResumeClassifier()
    recommender = JobRecommender()

    assert classifier.is_loaded, "Classifier model failed to load!"
    jobs = fetch_all_jobs()
    assert len(jobs) >= 30, f"Expected >= 30 jobs in SQLite, found {len(jobs)}"
    print(f"[OK] Classifier loaded successfully with {len(classifier.metadata.get('categories', []))} categories.")
    print(f"[OK] SQLite database verified with {len(jobs)} active job postings.")

    # 2. Test sample PDFs across different domains
    sample_files = [
        ("sample_data_scientist.pdf", "Data Science & Machine Learning"),
        ("sample_fullstack_dev.pdf", "Full Stack Web Development"),
        ("sample_devops_engineer.pdf", "DevOps & Cloud Engineering"),
        ("sample_cybersecurity_analyst.pdf", "Cybersecurity & Network Engineering"),
        ("sample_mobile_developer.pdf", "Mobile Application Development"),
        ("sample_qa_engineer.pdf", "Quality Assurance & Testing"),
    ]

    for filename, expected_category in sample_files:
        pdf_path = os.path.join(BASE_DIR, "data", "sample_resumes", filename)
        assert os.path.exists(pdf_path), f"Sample file not found: {pdf_path}"

        print(f"\n--- Testing Resume: {filename} ---")
        parsed = parser.parse(pdf_path, filename)

        assert parsed["is_valid"], f"Parsing failed: {parsed.get('error')}"
        assert len(parsed["raw_text"]) > 100, "Extracted text too short!"
        assert len(parsed["skills"]["all"]) > 0, "No skills extracted!"

        print(f"Extracted Text: {parsed['stats']['word_count']} words, {parsed['stats']['page_count']} page(s)")
        print(f"Identified Contacts: Email={parsed['contacts']['emails']}, Phone={parsed['contacts']['phones']}")
        print(f"Identified Skills ({len(parsed['skills']['all'])}): {', '.join(parsed['skills']['all'][:7])}...")

        # Classify
        prediction = classifier.predict(parsed["clean_text"])
        pred_cat = prediction["category"]
        conf = prediction["confidence"]
        print(f"Predicted Category: {pred_cat} (Confidence: {conf*100:.1f}%)")
        print(f"Expected Category : {expected_category}")
        assert pred_cat == expected_category, f"Mismatch: expected {expected_category}, got {pred_cat}"

        # Recommend
        recommendations = recommender.recommend(
            parsed["clean_text"],
            parsed["skills"]["all"],
            top_k=5
        )
        assert len(recommendations) == 5, f"Expected 5 recommendations, got {len(recommendations)}"
        top_rec = recommendations[0]
        print(f"Top Recommended Job: '{top_rec['title']}' at {top_rec['company']}")
        print(f"  Match Score   : {top_rec['match_score']}% (Similarity: {top_rec['similarity_score']})")
        print(f"  Matched Skills: {top_rec['matched_skills']}")
        print(f"  Missing Skills: {top_rec['missing_skills']}")

        # Log to SQLite
        log_id = log_resume_scan(
            filename=filename,
            predicted_category=pred_cat,
            confidence=conf,
            skills=parsed["skills"]["all"],
            top_job_match=top_rec["title"],
            match_score=top_rec["match_score"]
        )
        assert log_id > 0, "Failed to log scan into SQLite"

    print("\n" + "=" * 70)
    print("Verifying SQLite Scan History:")
    history = fetch_scan_history(limit=5)
    for entry in history:
        print(f" - [{entry['scanned_at']}] {entry['filename']} -> {entry['predicted_category']} ({entry['confidence']*100:.1f}%) | Top Match: {entry['top_job_match']} ({entry['match_score']}%)")

    print("\n[ALL TESTS PASSED SUCCESSFULLY!]")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()

