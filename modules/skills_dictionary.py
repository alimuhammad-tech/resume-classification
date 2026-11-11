"""
Skills Dictionary and Extraction Helpers.
Defines domain-specific skill catalogs and regex-based matching logic.
"""

from typing import Dict, List, Set
import re

# Comprehensive dictionary of technical and analytical skills mapped by domain
SKILLS_TAXONOMY: Dict[str, List[str]] = {
    "Programming Languages": [
        "Python", "Java", "C++", "C#", "C", "JavaScript", "TypeScript", "Go", "Golang",
        "Rust", "Ruby", "PHP", "Kotlin", "Swift", "Scala", "R", "Dart", "SQL", "HTML5", "CSS3", "Bash", "Shell"
    ],
    "Data Science & Machine Learning": [
        "Machine Learning", "Deep Learning", "Natural Language Processing", "NLP",
        "Computer Vision", "TensorFlow", "PyTorch", "Keras", "Scikit-Learn",
        "Pandas", "NumPy", "Matplotlib", "Seaborn", "OpenCV", "NLTK", "Spacy",
        "Hugging Face", "LLMs", "Generative AI", "Data Analysis", "Feature Engineering",
        "Predictive Modeling", "Statistics", "Data Visualization", "XGBoost", "LightGBM"
    ],
    "Data Engineering & Big Data": [
        "Apache Spark", "PySpark", "Hadoop", "Kafka", "Apache Airflow", "ETL",
        "Data Pipelines", "Snowflake", "Databricks", "dbt", "Hive", "Presto",
        "PostgreSQL", "MySQL", "MongoDB", "Cassandra", "Redis", "Elasticsearch",
        "BigQuery", "Redshift", "Data Warehousing"
    ],
    "Full Stack & Web Development": [
        "React", "React.js", "Angular", "Vue.js", "Next.js", "Node.js", "Express.js",
        "Django", "Flask", "FastAPI", "Spring Boot", "ASP.NET", "Ruby on Rails",
        "GraphQL", "REST API", "Tailwind CSS", "Bootstrap", "Redux", "Webpack",
        "Microservices", "HTML", "CSS", "JSON"
    ],
    "DevOps & Cloud Engineering": [
        "AWS", "Amazon Web Services", "Microsoft Azure", "Azure", "Google Cloud", "GCP",
        "Docker", "Kubernetes", "K8s", "Terraform", "Ansible", "Jenkins", "GitHub Actions",
        "GitLab CI", "CI/CD", "Linux", "Prometheus", "Grafana", "Helm", "Nginx",
        "Infrastructure as Code", "IaC", "CloudFormation"
    ],
    "Cybersecurity & Networks": [
        "Ethical Hacking", "Penetration Testing", "Network Security", "Information Security",
        "SIEM", "Splunk", "Wireshark", "Firewalls", "Cryptography", "Vulnerability Assessment",
        "OWASP", "SOC", "Incident Response", "Identity and Access Management", "IAM",
        "TCP/IP", "DNS", "VPN", "Metasploit", "Burp Suite"
    ],
    "Mobile App Development": [
        "Android", "iOS", "Flutter", "React Native", "SwiftUI", "Jetpack Compose",
        "Xcode", "Android Studio", "Kotlin", "Swift", "Dart", "Mobile UI", "SQLite"
    ],
    "UI/UX Design & Frontend": [
        "Figma", "Adobe XD", "Wireframing", "Prototyping", "User Research",
        "Usability Testing", "Design Systems", "UI Design", "UX Design", "Responsive Design",
        "CSS", "Sass", "SCSS", "InVision", "Interaction Design"
    ],
    "Quality Assurance & Testing": [
        "Selenium", "Cypress", "Playwright", "JUnit", "PyTest", "TestNG",
        "Manual Testing", "Automated Testing", "API Testing", "Postman",
        "Performance Testing", "JMeter", "Regression Testing", "Jira", "Bug Tracking",
        "QA Automation", "Test Planning"
    ],
    "Methodologies & Tools": [
        "Git", "GitHub", "GitLab", "Bitbucket", "Agile", "Scrum", "Kanban",
        "Jira", "Confluence", "Trello", "CI/CD", "Unit Testing", "Code Review"
    ]
}

def get_all_skills_flat() -> List[str]:
    """Return a flat, deduplicated list of all skills."""
    all_skills = set()
    for category, skills in SKILLS_TAXONOMY.items():
        for skill in skills:
            all_skills.add(skill)
    return sorted(list(all_skills), key=lambda x: (-len(x), x))


def extract_skills_from_text(text: str) -> Dict[str, List[str]]:
    """
    Extract skills present in the given text categorized by domain.
    Returns:
        dict: {
            "all": sorted list of all unique matched skills,
            "by_category": {category: [matched skills]}
        }
    """
    if not text:
        return {"all": [], "by_category": {}}

    matched_all: Set[str] = set()
    by_category: Dict[str, List[str]] = {}

    # Pre-clean text for matching: replace multiple whitespace/newlines with single spaces
    prepared_text = " " + text.lower() + " "

    for category, skills in SKILLS_TAXONOMY.items():
        matched_in_cat = []
        for skill in skills:
            # Handle special symbols like C++, C#, .js, etc.
            escaped = re.escape(skill.lower())
            
            # Boundary checks: allow optional plural suffix (s/es) for alphabetic skills
            if skill[-1].isalpha():
                pattern = rf"(?<![a-zA-Z0-9_#+]){escaped}(?:s|es)?(?![a-zA-Z0-9_#+])"
            else:
                pattern = rf"(?<![a-zA-Z0-9_#+]){escaped}(?![a-zA-Z0-9_#+])"
            
            if re.search(pattern, prepared_text, re.IGNORECASE):
                matched_all.add(skill)
                matched_in_cat.append(skill)

        
        if matched_in_cat:
            by_category[category] = sorted(matched_in_cat)

    return {
        "all": sorted(list(matched_all)),
        "by_category": by_category
    }

