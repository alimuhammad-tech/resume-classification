"""
Job Recommendation Module.
Calculates TF-IDF vector representations and Cosine Similarity scores
between parsed resume profiles and available job postings.
Computes matched skills and missing skill gaps for career growth.
"""

from typing import List, Dict, Any, Optional
import os
import sys
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure project root in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from modules.database import fetch_all_jobs, DB_PATH
from modules.preprocessor import clean_text


class JobRecommender:
    """Recommends top job matches using TF-IDF and Cosine Similarity."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._jobs_cache: Optional[List[Dict[str, Any]]] = None

    def load_jobs(self, force_reload: bool = False) -> List[Dict[str, Any]]:
        """Fetch all available job listings from the SQLite database."""
        if self._jobs_cache is None or force_reload:
            self._jobs_cache = fetch_all_jobs(self.db_path)
        return self._jobs_cache

    @staticmethod
    def _parse_skills_list(skills_field: Any) -> List[str]:
        """Convert comma-separated or list skills into a clean list of strings."""
        if isinstance(skills_field, list):
            return [s.strip() for s in skills_field if s.strip()]
        if isinstance(skills_field, str):
            return [s.strip() for s in skills_field.split(",") if s.strip()]
        return []

    def compute_skill_gap(self, candidate_skills: List[str], job_skills: List[str]) -> Dict[str, Any]:
        """
        Identify matched skills and missing skill gaps between candidate and job.
        Case-insensitive with alias normalization.
        """
        cand_normalized = {s.strip().lower(): s.strip() for s in candidate_skills}
        
        matched = []
        missing = []

        for req in job_skills:
            req_clean = req.strip()
            req_lower = req_clean.lower()
            req_stem = req_lower.rstrip("s")
            
            # Direct or alias matching (e.g., 'react' matches 'react.js', plurals 'api'/'apis')
            found = False
            for c_lower, c_orig in cand_normalized.items():
                c_stem = c_lower.rstrip("s")
                if (
                    req_lower == c_lower
                    or req_stem == c_stem
                    or (len(req_lower) >= 3 and req_lower in c_lower)
                    or (len(c_lower) >= 3 and c_lower in req_lower)
                ):
                    matched.append(req_clean)
                    found = True
                    break
            if not found:
                missing.append(req_clean)


        total_req = len(job_skills)
        match_ratio = len(matched) / total_req if total_req > 0 else 1.0

        return {
            "matched_skills": matched,
            "missing_skills": missing,
            "match_ratio": round(match_ratio, 4)
        }

    def recommend(
        self,
        resume_clean_text: str,
        resume_skills: List[str],
        top_k: int = 5,
        category_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Recommend top K matching jobs using TF-IDF and Cosine Similarity.
        
        Args:
            resume_clean_text: Cleaned text of the candidate's resume.
            resume_skills: Flat list of extracted candidate skills.
            top_k: Number of recommendations to return (default: 5).
            category_filter: Optional filter by job category.

        Returns:
            List of job recommendations with match scores and skill gap analysis.
        """
        jobs = self.load_jobs()
        if not jobs:
            return []

        # Filter by category if requested
        if category_filter and category_filter != "All":
            filtered_jobs = [j for j in jobs if j.get("category") == category_filter]
            if filtered_jobs:
                jobs = filtered_jobs

        # Prepare corpus for TF-IDF: Job descriptions + required skills
        job_texts = []
        for j in jobs:
            req_skills = j.get("required_skills", "")
            title = j.get("title", "")
            category = j.get("category", "")
            desc = j.get("description", "")
            # Combine components with emphasis on title, category, and skills
            corpus_entry = f"{title} {category} {req_skills} {desc} {req_skills}"
            job_texts.append(clean_text(corpus_entry))

        # Prepare candidate representation (clean text + skill reinforcement)
        candidate_corpus = resume_clean_text + " " + " ".join(resume_skills) + " " + " ".join(resume_skills)

        # Fit TF-IDF on jobs corpus + candidate document
        all_docs = job_texts + [candidate_corpus]
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            max_features=2500,
            sublinear_tf=True
        )
        tfidf_matrix = vectorizer.fit_transform(all_docs)

        # Candidate vector is the last row; jobs vectors are all preceding rows
        job_vectors = tfidf_matrix[:-1]
        candidate_vector = tfidf_matrix[-1:]

        # Compute cosine similarity
        cosine_scores = cosine_similarity(candidate_vector, job_vectors).flatten()

        recommendations = []
        for idx, job in enumerate(jobs):
            similarity = float(cosine_scores[idx])
            job_skills = self._parse_skills_list(job.get("required_skills", ""))
            gap_analysis = self.compute_skill_gap(resume_skills, job_skills)
            
            # Blended final score: 70% TF-IDF Cosine Similarity + 30% direct skill overlap ratio
            blended_score = (similarity * 0.70) + (gap_analysis["match_ratio"] * 0.30)
            
            recommendations.append({
                "job_id": job.get("id"),
                "title": job.get("title"),
                "company": job.get("company"),
                "category": job.get("category"),
                "location": job.get("location"),
                "experience_level": job.get("experience_level"),
                "required_skills": job_skills,
                "description": job.get("description"),
                "similarity_score": round(similarity, 4),
                "skill_match_ratio": gap_analysis["match_ratio"],
                "match_score": round(blended_score * 100, 1), # Percentage e.g. 87.5%
                "matched_skills": gap_analysis["matched_skills"],
                "missing_skills": gap_analysis["missing_skills"]
            })

        # Rank descending by final match score
        recommendations.sort(key=lambda x: x["match_score"], reverse=True)

        return recommendations[:top_k]
