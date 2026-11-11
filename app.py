"""
Resume Classification and Job Recommendation System.
Senior-engineered, modern Streamlit dashboard for internship presentation.
"""

from typing import Dict, Any, List, Optional
import os
import sys
import io
import json
import pandas as pd
import numpy as np
import altair as alt
import streamlit as st

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from resume_parser import ResumeParser
from modules.classifier import ResumeClassifier
from job_recommender import JobRecommender
from modules.database import (
    fetch_all_jobs,
    log_resume_scan,
    fetch_scan_history,
    init_database,
    get_db_connection,
    DB_PATH
)
from modules.skills_dictionary import SKILLS_TAXONOMY

# Page configuration
st.set_page_config(
    page_title="ResumeAI - Classifier & Job Recommender",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    /* Global style tweaks */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
    }
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    /* Metric Card */
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-title {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 4px;
    }
    /* Skill Badges */
    .skill-badge-matched {
        display: inline-block;
        background-color: #DEF7EC;
        color: #03543F;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.82rem;
        font-weight: 600;
        margin: 2px 4px 2px 0;
        border: 1px solid #BCF0DA;
    }
    .skill-badge-missing {
        display: inline-block;
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.82rem;
        font-weight: 600;
        margin: 2px 4px 2px 0;
        border: 1px solid #FBD5D5;
    }
    .skill-badge-general {
        display: inline-block;
        background-color: #E1EFFE;
        color: #1E429F;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.82rem;
        font-weight: 500;
        margin: 2px 4px 2px 0;
        border: 1px solid #C3DDFD;
    }
    /* Job Card */
    .job-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .job-card:hover {
        border-color: #CBD5E1;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.08);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_components():
    """Cache models, vectorizer, and database initialization."""
    init_database()
    parser = ResumeParser()
    classifier = ResumeClassifier()
    recommender = JobRecommender()
    return parser, classifier, recommender


parser, classifier, recommender = get_components()

# Sidebar: Project Info & Quick Sample Selector
with st.sidebar:
    st.markdown("### 💼 ResumeAI Assistant")
    st.markdown(
        "**Resume Classification & Job Recommendation System**  \n"
        "*Machine Learning Internship Project*"
    )
    st.divider()

    st.markdown("#### ⚙️ System Status")
    if classifier.is_loaded:
        st.success("✅ ML Classifier: **Ready**")
        st.caption(
            f"**Model**: Logistic Regression  \n"
            f"**Features**: TF-IDF (1,500 n-grams)  \n"
            f"**Categories**: {len(classifier.metadata.get('categories', []))}  \n"
            f"**Validation Accuracy**: {classifier.metadata.get('accuracy', 0.0)*100:.1f}%"
        )
    else:
        st.error("❌ Classifier model not loaded. Please run `train_model.py`.")

    st.divider()

    st.markdown("#### 🧪 Test with Sample Resume")
    sample_dir = os.path.join(BASE_DIR, "data", "sample_resumes")
    sample_files = [
        "Select a sample...",
        "sample_data_scientist.pdf",
        "sample_fullstack_dev.pdf",
        "sample_devops_engineer.pdf",
        "sample_cybersecurity_analyst.pdf",
        "sample_mobile_developer.pdf",
        "sample_qa_engineer.pdf"
    ]
    selected_sample = st.selectbox(
        "Quickly test without uploading your own PDF:",
        options=sample_files,
        index=0,
        help="Loads realistic pre-generated candidate resumes directly."
    )

    st.divider()
    st.markdown("#### 🛠️ Tech Stack")
    st.markdown("""
    - **Language**: Python 3.12
    - **UI**: Streamlit
    - **Parsing**: PyMuPDF (`pymupdf`)
    - **NLP/ML**: Scikit-Learn, TF-IDF
    - **Classifier**: Logistic Regression
    - **Recommendation**: Cosine Similarity
    - **Storage**: SQLite 3 & Pandas
    """)

    st.caption("Developed with clean, production-standard ML principles.")


# Header
st.markdown("<div class='main-header'>Resume Classification & Job Recommendation System</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-header'>Upload a candidate resume in PDF format to automatically extract technical skills, "
    "predict the professional domain using Machine Learning, and receive top-matching job opportunities with missing skill gap insights.</div>",
    unsafe_allow_html=True
)

# Tabs
tab_analysis, tab_jobs, tab_model = st.tabs([
    "📄 Resume Analysis & Recommendation",
    "💼 Browse Available Jobs (SQLite)",
    "📊 Model Performance & Insights"
])


# -------------------------------------------------------------
# TAB 1: RESUME ANALYSIS & RECOMMENDATION
# -------------------------------------------------------------
with tab_analysis:
    st.subheader("1. Resume Upload & Document Validation")

    col_upload, col_preview = st.columns([1.5, 1], gap="medium")

    uploaded_pdf = None
    target_filename = "resume.pdf"
    file_bytes = None

    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload Candidate Resume (PDF format only):",
            type=["pdf"],
            help="PyMuPDF extracts text directly from standard PDF resumes."
        )

        if uploaded_file is not None:
            file_bytes = uploaded_file.read()
            target_filename = uploaded_file.name
        elif selected_sample != "Select a sample...":
            sample_path = os.path.join(sample_dir, selected_sample)
            if os.path.exists(sample_path):
                with open(sample_path, "rb") as f:
                    file_bytes = f.read()
                target_filename = selected_sample
                st.info(f"Loaded sample resume: **{selected_sample}**")

    # Perform Parsing and Analysis if file is available
    if file_bytes is not None:
        with st.spinner("Analyzing resume text and running ML inference..."):
            parsed_result = parser.parse(file_bytes, filename=target_filename)

        if not parsed_result["is_valid"]:
            st.error(f"❌ **Validation Error**: {parsed_result.get('error')}")
        else:
            if parsed_result.get("warning"):
                st.warning(f"⚠️ {parsed_result['warning']}")

            # Predict category
            prediction = classifier.predict(parsed_result["clean_text"])
            pred_category = prediction["category"]
            pred_conf = prediction["confidence"]

            # Recommend jobs
            recommendations = recommender.recommend(
                parsed_result["clean_text"],
                parsed_result["skills"]["all"],
                top_k=5
            )

            # Log scan in database
            top_job_title = recommendations[0]["title"] if recommendations else None
            top_job_score = recommendations[0]["match_score"] if recommendations else None
            log_resume_scan(
                filename=target_filename,
                predicted_category=pred_category,
                confidence=pred_conf,
                skills=parsed_result["skills"]["all"],
                top_job_match=top_job_title,
                match_score=top_job_score
            )

            # Preview Metrics in right column
            with col_preview:
                stats = parsed_result["stats"]
                contacts = parsed_result["contacts"]
                
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.markdown("<div class='metric-title'>Document Summary</div>", unsafe_allow_html=True)
                st.write(f"📁 **File**: `{target_filename}`")
                st.write(f"📄 **Pages**: {stats.get('page_count', 1)} | **Words**: {stats.get('word_count', 0)}")
                
                if contacts.get("emails"):
                    st.write(f"✉️ **Email**: `{contacts['emails'][0]}`")
                if contacts.get("phones"):
                    st.write(f"📞 **Phone**: `{contacts['phones'][0]}`")
                if contacts.get("github"):
                    st.write(f"💻 **GitHub**: [{contacts['github']}]({contacts['github']})")
                if contacts.get("linkedin"):
                    st.write(f"🔗 **LinkedIn**: [{contacts['linkedin']}]({contacts['linkedin']})")
                st.markdown("</div>", unsafe_allow_html=True)

            st.divider()

            # SECTION 2: Classification Results
            st.subheader("2. Machine Learning Classification")
            
            c_pred1, c_pred2 = st.columns([1.2, 1.8], gap="large")

            with c_pred1:
                st.markdown("<div class='metric-card' style='border-left: 5px solid #2563EB;'>", unsafe_allow_html=True)
                st.markdown("<div class='metric-title'>Predicted Job Category</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-value' style='color:#1D4ED8;'>{pred_category}</div>", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"**Model Confidence**: `{pred_conf * 100:.1f}%`")
                st.progress(float(pred_conf))
                st.caption("Classified using TF-IDF n-gram vectorization + Multi-class Logistic Regression.")
                st.markdown("</div>", unsafe_allow_html=True)

            with c_pred2:
                st.markdown("**Confidence Distribution Across Categories:**")
                chart_data = pd.DataFrame({
                    "Category": [item[0] for item in prediction["sorted_rankings"]],
                    "Probability (%)": [round(item[1] * 100, 1) for item in prediction["sorted_rankings"]]
                })

                bar_chart = alt.Chart(chart_data).mark_bar(cornerRadius=4).encode(
                    x=alt.X("Probability (%):Q", scale=alt.Scale(domain=[0, 100]), title="Probability (%)"),
                    y=alt.Y("Category:N", sort="-x", title=None),
                    color=alt.condition(
                        alt.datum.Category == pred_category,
                        alt.value("#2563EB"),
                        alt.value("#CBD5E1")
                    ),
                    tooltip=["Category", "Probability (%)"]
                ).properties(height=260)
                st.altair_chart(bar_chart, use_container_width=True)

            st.divider()

            # SECTION 3: Extracted Skills
            st.subheader("3. Extracted Skills Inventory")
            total_skills = len(parsed_result["skills"]["all"])
            st.markdown(f"Identified **{total_skills} technical skills** mapped against domain taxonomy:")

            if total_skills == 0:
                st.warning("No technical skills from the dictionary were identified in this resume.")
            else:
                col_sk1, col_sk2 = st.columns([1.6, 1.4], gap="medium")
                with col_sk1:
                    for category, skills in parsed_result["skills"]["by_category"].items():
                        with st.expander(f"📌 {category} ({len(skills)})", expanded=True):
                            badges_html = " ".join([
                                f"<span class='skill-badge-general'>{s}</span>"
                                for s in skills
                            ])
                            st.markdown(badges_html, unsafe_allow_html=True)
                with col_sk2:
                    st.markdown("**Skill Domain Distribution:**")
                    cat_counts = [
                        {"Domain": cat, "Count": len(s_list)}
                        for cat, s_list in parsed_result["skills"]["by_category"].items()
                    ]
                    if cat_counts:
                        sk_chart_df = pd.DataFrame(cat_counts)
                        domain_chart = alt.Chart(sk_chart_df).mark_bar(cornerRadius=4, color="#10B981").encode(
                            x=alt.X("Count:Q", title="Number of Skills"),
                            y=alt.Y("Domain:N", sort="-x", title=None),
                            tooltip=["Domain", "Count"]
                        ).properties(height=220)
                        st.altair_chart(domain_chart, use_container_width=True)

            with st.expander("🔍 View Extracted Cleaned Text (TF-IDF Input)"):
                st.text_area("Preprocessed Resume Text", parsed_result["clean_text"], height=160, disabled=True)

            st.divider()

            # SECTION 4: Top Matching Job Recommendations
            st.subheader("4. Top 5 Recommended Job Matches")
            st.markdown(
                "Ranked using **TF-IDF representation and Cosine Similarity** against live SQLite job listings. "
                "Missing skills highlight actionable career growth areas."
            )

            if not recommendations:
                st.info("No matching jobs found in the database.")
            else:
                for idx, job in enumerate(recommendations, 1):
                    match_pct = job["match_score"]
                    card_border_color = "#10B981" if match_pct >= 50 else "#3B82F6" if match_pct >= 35 else "#64748B"

                    with st.container():
                        st.markdown(f"""
                        <div class='job-card' style='border-left: 6px solid {card_border_color};'>
                            <div style='display:flex; justify-content:space-between; align-items:center;'>
                                <div>
                                    <span style='font-size:1.25rem; font-weight:700; color:#1E293B;'>#{idx}. {job['title']}</span>
                                    <span style='color:#64748B; font-size:0.95rem; margin-left:8px;'>at <strong>{job['company']}</strong></span>
                                </div>
                                <div style='text-align:right;'>
                                    <span style='font-size:1.4rem; font-weight:800; color:{card_border_color};'>{match_pct}% Match</span>
                                </div>
                            </div>
                            <div style='margin-top: 6px; font-size:0.85rem; color:#64748B;'>
                                📍 {job['location']} &nbsp;|&nbsp; 🏷️ {job['category']} &nbsp;|&nbsp; 💼 {job['experience_level']}
                            </div>
                            <p style='margin-top:10px; font-size:0.92rem; color:#334155;'>
                                {job['description']}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                        col_m1, col_m2 = st.columns(2)
                        with col_m1:
                            st.markdown("**✅ Matched Skills:**")
                            if job["matched_skills"]:
                                matched_html = " ".join([
                                    f"<span class='skill-badge-matched'>{s}</span>"
                                    for s in job["matched_skills"]
                                ])
                                st.markdown(matched_html, unsafe_allow_html=True)
                            else:
                                st.caption("No direct skill overlap found.")

                        with col_m2:
                            st.markdown("**⚠️ Missing Skills for this Job:**")
                            if job["missing_skills"]:
                                missing_html = " ".join([
                                    f"<span class='skill-badge-missing'>{s}</span>"
                                    for s in job["missing_skills"]
                                ])
                                st.markdown(missing_html, unsafe_allow_html=True)
                            else:
                                st.markdown("<span style='color:#059669; font-weight:600;'>🎉 Perfect match! You possess all required skills.</span>", unsafe_allow_html=True)

                        st.write("")

                # Generate Downloadable Career Report
                report_lines = [
                    f"# Candidate Resume Analysis & Job Recommendation Report",
                    f"- **Analyzed File**: {target_filename}",
                    f"- **Predicted Domain**: {pred_category} (Confidence: {pred_conf * 100:.1f}%)",
                    f"- **Total Identified Skills**: {len(parsed_result['skills']['all'])}",
                    f"- **Identified Skills**: {', '.join(parsed_result['skills']['all'])}",
                    f"\n## Top Recommended Job Listings\n"
                ]
                for i, r in enumerate(recommendations, 1):
                    report_lines.append(f"### {i}. {r['title']} - {r['company']}")
                    report_lines.append(f"- **Match Score**: {r['match_score']}% (Similarity: {r['similarity_score']})")
                    report_lines.append(f"- **Location**: {r['location']} | **Experience**: {r['experience_level']}")
                    report_lines.append(f"- **Matched Skills**: {', '.join(r['matched_skills']) if r['matched_skills'] else 'None'}")
                    report_lines.append(f"- **Missing Skills to Learn**: {', '.join(r['missing_skills']) if r['missing_skills'] else 'None (Complete Match)'}\n")

                report_text = "\n".join(report_lines)
                st.download_button(
                    label="📥 Download Career & Job Match Report (.md)",
                    data=report_text,
                    file_name=f"career_report_{target_filename.replace('.pdf', '')}.md",
                    mime="text/markdown"
                )

    else:
        st.info("👆 Please upload a PDF resume above or pick a sample resume from the left sidebar to start analysis.")


# -------------------------------------------------------------
# TAB 2: EXPLORE JOB LISTINGS (SQLite)
# -------------------------------------------------------------
with tab_jobs:
    st.subheader("Available Job Postings (SQLite Database)")
    st.caption("Active job listings stored in `data/jobs.db`. Used by TF-IDF & Cosine Similarity recommender.")

    all_jobs = fetch_all_jobs()
    jobs_df = pd.DataFrame(all_jobs)

    if not jobs_df.empty:
        col_f1, col_f2, col_f3 = st.columns([1.5, 1, 1.5])
        with col_f1:
            category_filter = st.selectbox(
                "Filter by Domain Category:",
                options=["All"] + sorted(jobs_df["category"].unique().tolist())
            )
        with col_f2:
            exp_filter = st.selectbox(
                "Filter by Experience:",
                options=["All"] + sorted(jobs_df["experience_level"].unique().tolist())
            )
        with col_f3:
            search_query = st.text_input("Search Title or Required Skill:", "")

        # Apply filters
        filtered_df = jobs_df.copy()
        if category_filter != "All":
            filtered_df = filtered_df[filtered_df["category"] == category_filter]
        if exp_filter != "All":
            filtered_df = filtered_df[filtered_df["experience_level"] == exp_filter]
        if search_query:
            q = search_query.lower()
            filtered_df = filtered_df[
                filtered_df["title"].str.lower().str.contains(q, na=False) |
                filtered_df["required_skills"].str.lower().str.contains(q, na=False) |
                filtered_df["company"].str.lower().str.contains(q, na=False)
            ]

        st.markdown(f"Showing **{len(filtered_df)}** of {len(jobs_df)} total job postings:")

        display_df = filtered_df[["id", "title", "company", "category", "location", "experience_level", "required_skills"]].copy()
        display_df.columns = ["ID", "Job Title", "Company", "Category", "Location", "Level", "Required Skills"]
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        with st.expander("➕ Add a New Job Listing to SQLite Database"):
            with st.form("add_job_form", clear_on_submit=True):
                col_n1, col_n2 = st.columns(2)
                with col_n1:
                    new_title = st.text_input("Job Title *", placeholder="e.g. Lead Machine Learning Engineer")
                    new_company = st.text_input("Company *", placeholder="e.g. Innovate AI")
                    new_category = st.selectbox("Job Category *", options=list(SKILLS_TAXONOMY.keys())[:8])
                with col_n2:
                    new_location = st.text_input("Location", value="Remote")
                    new_exp = st.selectbox("Experience Level", ["Entry-Level", "Mid-Level", "Senior"])
                    new_skills = st.text_input("Required Skills (comma-separated) *", placeholder="Python, PyTorch, Docker, Git")
                new_desc = st.text_area("Job Description *", placeholder="Describe the day-to-day role and responsibilities...")

                submit_job = st.form_submit_button("Save Job to Database")

                if submit_job:
                    if new_title and new_company and new_skills and new_desc:
                        with get_db_connection(DB_PATH) as conn:
                            cursor = conn.cursor()
                            cursor.execute("""
                                INSERT INTO jobs (title, company, category, location, experience_level, required_skills, description)
                                VALUES (?, ?, ?, ?, ?, ?, ?)
                            """, (new_title, new_company, new_category, new_location, new_exp, new_skills, new_desc))
                            conn.commit()
                        st.success(f"✅ Added job '{new_title}' at '{new_company}' to SQLite database!")
                        recommender.load_jobs(force_reload=True)
                        st.rerun()
                    else:
                        st.warning("Please fill in all required fields.")
    else:
        st.warning("No jobs found in database. Run `python modules/data_generator.py` to populate.")


# -------------------------------------------------------------
# TAB 3: MODEL PERFORMANCE & INSIGHTS
# -------------------------------------------------------------
with tab_model:
    st.subheader("Model Performance & Architecture Insights")
    st.markdown(
        "Detailed evaluation metrics, technical architecture, and SQLite audit history for this internship project."
    )

    col_info1, col_info2, col_info3, col_info4 = st.columns(4)
    meta = classifier.metadata

    with col_info1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("<div class='metric-title'>Algorithm</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-value' style='font-size:1.25rem;'>Logistic Regression</div>", unsafe_allow_html=True)
        st.caption("Multi-class One-vs-Rest / Multithreaded L-BFGS")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_info2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("<div class='metric-title'>Feature Extraction</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value' style='font-size:1.25rem;'>{meta.get('vocabulary_size', 1500):,} n-grams</div>", unsafe_allow_html=True)
        st.caption("TF-IDF Vectorizer (unigrams + bigrams)")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_info3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("<div class='metric-title'>Validation Accuracy</div>", unsafe_allow_html=True)
        acc = meta.get("accuracy", 0.0)
        st.markdown(f"<div class='metric-value' style='color:#059669;'>{acc * 100:.1f}%</div>", unsafe_allow_html=True)
        st.caption("Stratified 25% test split")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_info4:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("<div class='metric-title'>Training Corpus</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{meta.get('total_samples', 120)}</div>", unsafe_allow_html=True)
        st.caption(f"{len(meta.get('categories', []))} Distinct Tech Categories")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Classification report table
    report = meta.get("classification_report", {})
    if report:
        st.markdown("#### 📋 Test Split Evaluation Metrics (Per-Category Breakdown)")
        cat_rows = []
        for cat in meta.get("categories", []):
            if cat in report:
                cat_rows.append({
                    "Category": cat,
                    "Precision": f"{report[cat]['precision']:.2f}",
                    "Recall": f"{report[cat]['recall']:.2f}",
                    "F1-Score": f"{report[cat]['f1-score']:.2f}",
                    "Support": report[cat]["support"]
                })
        st.dataframe(pd.DataFrame(cat_rows), use_container_width=True, hide_index=True)

    # Important Academic / Ethical Disclaimer
    st.info(
        "**Academic & Engineering Integrity Note** :  \n"
        "The model metrics above are derived from a curated benchmark validation split. "
        "In production environments with unstructured, messy, or image-heavy PDFs, performance will depend "
        "on OCR accuracy and larger domain-specific datasets."
    )

    st.divider()

    # Scan history from SQLite
    st.markdown("#### 🕒 Recent Resume Scan Audit Log (SQLite)")
    history_records = fetch_scan_history(limit=10)
    if history_records:
        hist_df = pd.DataFrame(history_records)
        hist_df.columns = ["ID", "Filename", "Predicted Category", "Confidence", "Skills Count", "Top Match", "Match %", "Scanned At"]
        hist_df["Confidence"] = hist_df["Confidence"].apply(lambda x: f"{x*100:.1f}%" if pd.notnull(x) else "-")
        hist_df["Match %"] = hist_df["Match %"].apply(lambda x: f"{x:.1f}%" if pd.notnull(x) else "-")
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
    else:
        st.caption("No recent scans logged yet.")

