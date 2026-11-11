"""
Dataset Generator.
Generates comprehensive synthetic datasets for:
1. Job listings (>= 35 listings across 8 tech domains) -> data/jobs_dataset.csv
2. Labeled resume dataset (120+ resumes across 8 categories) -> data/resumes_dataset.csv
3. Sample PDF resumes for live testing in Streamlit -> data/sample_resumes/
"""

import os
import sys
import csv

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import pandas as pd
import pymupdf as fitz
from modules.database import populate_jobs_from_dataframe, DB_PATH


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
SAMPLE_RESUMES_DIR = os.path.join(DATA_DIR, "sample_resumes")

CATEGORIES = [
    "Data Science & Machine Learning",
    "Full Stack Web Development",
    "DevOps & Cloud Engineering",
    "Cybersecurity & Network Engineering",
    "Mobile Application Development",
    "UI/UX Design & Frontend",
    "Quality Assurance & Testing",
    "Data Engineering & Big Data"
]

JOBS_DATA = [
    # 1. Data Science & Machine Learning
    {
        "title": "Machine Learning Engineer",
        "company": "Apex AI Labs",
        "category": "Data Science & Machine Learning",
        "location": "San Francisco, CA (Hybrid)",
        "experience_level": "Mid-Level",
        "required_skills": "Python, Machine Learning, TensorFlow, PyTorch, Scikit-Learn, Docker, Git",
        "description": "Design and deploy scalable deep learning and machine learning models for production computer vision and natural language processing pipelines. Collaborate with backend engineers to integrate PyTorch and TensorFlow inference microservices with Docker and REST APIs."
    },
    {
        "title": "Data Scientist",
        "company": "Quantix Analytics",
        "category": "Data Science & Machine Learning",
        "location": "New York, NY (Remote)",
        "experience_level": "Mid-Level",
        "required_skills": "Python, Pandas, NumPy, Scikit-Learn, Statistics, Data Analysis, Matplotlib, SQL",
        "description": "Formulate predictive statistical models and machine learning algorithms on large-scale tabular customer data. Translate business questions into data-driven insights, perform rigorous feature engineering, and communicate findings using interactive dashboards."
    },
    {
        "title": "NLP & Generative AI Specialist",
        "company": "Cognitive Cloud",
        "category": "Data Science & Machine Learning",
        "location": "Boston, MA (Hybrid)",
        "experience_level": "Senior",
        "required_skills": "Python, NLP, Hugging Face, LLMs, PyTorch, Deep Learning, Generative AI",
        "description": "Architect state-of-the-art transformer pipelines, fine-tune open-source Large Language Models, and construct retrieval-augmented generation (RAG) applications using Hugging Face, PyTorch, and vector databases."
    },
    {
        "title": "Junior Computer Vision Researcher",
        "company": "Visionary Robotics",
        "category": "Data Science & Machine Learning",
        "location": "Austin, TX (On-site)",
        "experience_level": "Entry-Level",
        "required_skills": "Python, OpenCV, Computer Vision, PyTorch, NumPy, Git",
        "description": "Assist in researching and training convolutional neural networks and object detection pipelines for autonomous robotics. Benchmark inference speeds and curate high-resolution image datasets."
    },
    {
        "title": "AI Research Scientist",
        "company": "DeepMind Frontiers",
        "category": "Data Science & Machine Learning",
        "location": "Seattle, WA (Remote)",
        "experience_level": "Senior",
        "required_skills": "Python, PyTorch, Deep Learning, Statistics, Scikit-Learn, TensorFlow, Git",
        "description": "Lead foundational research in generative modeling and reinforcement learning. Publish academic contributions and pioneer novel deep learning architectures for enterprise intelligence platforms."
    },

    # 2. Full Stack Web Development
    {
        "title": "Full Stack Software Engineer",
        "company": "Nexus Technologies",
        "category": "Full Stack Web Development",
        "location": "Austin, TX (Remote)",
        "experience_level": "Mid-Level",
        "required_skills": "JavaScript, TypeScript, React, Node.js, Express.js, PostgreSQL, REST API, Git",
        "description": "Build high-throughput web applications utilizing modern TypeScript, React, and Node.js microservices. Architect relational schemas in PostgreSQL and engineer resilient, well-documented RESTful APIs."
    },
    {
        "title": "Python Web Developer",
        "company": "Streamline Digital",
        "category": "Full Stack Web Development",
        "location": "Chicago, IL (Hybrid)",
        "experience_level": "Entry-Level",
        "required_skills": "Python, Django, FastAPI, PostgreSQL, HTML, CSS, JavaScript, Git",
        "description": "Develop scalable web backends with Django and FastAPI. Integrate frontend user interfaces, optimize database queries with Django ORM, and author automated unit and integration tests."
    },
    {
        "title": "Senior MERN Stack Architect",
        "company": "Elevate Platforms",
        "category": "Full Stack Web Development",
        "location": "Seattle, WA (Remote)",
        "experience_level": "Senior",
        "required_skills": "React, Node.js, MongoDB, Express.js, TypeScript, GraphQL, Docker, Microservices",
        "description": "Direct architecture for high-volume consumer platforms using React, Node.js, and MongoDB. Pioneer GraphQL schemas, modular microservices decomposition, and real-time state synchronization."
    },
    {
        "title": "Java Spring Boot Developer",
        "company": "Enterprise Horizon",
        "category": "Full Stack Web Development",
        "location": "Dallas, TX (Hybrid)",
        "experience_level": "Mid-Level",
        "required_skills": "Java, Spring Boot, MySQL, REST API, Microservices, Git, JUnit",
        "description": "Engineer enterprise financial service backends with Java and Spring Boot. Implement event-driven microservices, configure secure REST endpoints, and ensure high test coverage."
    },
    {
        "title": "Frontend / React Engineer",
        "company": "HyperPixel Inc",
        "category": "Full Stack Web Development",
        "location": "San Jose, CA (Remote)",
        "experience_level": "Mid-Level",
        "required_skills": "React, JavaScript, TypeScript, Redux, Tailwind CSS, HTML5, CSS3, Webpack",
        "description": "Craft responsive, accessible frontend single-page web applications using React, Next.js, and Tailwind CSS. Optimize web vitals and frontend bundle sizes for millions of monthly active users."
    },

    # 3. DevOps & Cloud Engineering
    {
        "title": "Cloud Infrastructure Engineer",
        "company": "CloudScale Systems",
        "category": "DevOps & Cloud Engineering",
        "location": "Denver, CO (Remote)",
        "experience_level": "Senior",
        "required_skills": "AWS, Terraform, Kubernetes, Docker, Linux, CI/CD, Python, Prometheus",
        "description": "Architect multi-region fault-tolerant infrastructure on Amazon Web Services using Terraform and Kubernetes. Automate containerized deployments with GitLab CI and configure Prometheus and Grafana telemetry."
    },
    {
        "title": "Site Reliability Engineer (SRE)",
        "company": "Reliant Cloud",
        "category": "DevOps & Cloud Engineering",
        "location": "New York, NY (Hybrid)",
        "experience_level": "Mid-Level",
        "required_skills": "Linux, Kubernetes, Docker, Python, Bash, CI/CD, Ansible, Grafana",
        "description": "Drive 99.99% system availability across globally distributed microservices. Write automation runbooks in Python and Bash, conduct incident post-mortems, and eliminate operational toil."
    },
    {
        "title": "DevOps Automation Engineer",
        "company": "Pipeline Velocity",
        "category": "DevOps & Cloud Engineering",
        "location": "Atlanta, GA (Remote)",
        "experience_level": "Mid-Level",
        "required_skills": "Jenkins, GitHub Actions, Docker, AWS, Linux, Terraform, Shell",
        "description": "Standardize continuous integration and continuous deployment pipelines across 30+ engineering squads using GitHub Actions and Docker. Ensure rapid, zero-downtime blue-green deployments."
    },
    {
        "title": "Azure Cloud Solutions Architect",
        "company": "Global Enterprise Cloud",
        "category": "DevOps & Cloud Engineering",
        "location": "Raleigh, NC (Hybrid)",
        "experience_level": "Senior",
        "required_skills": "Azure, Terraform, Docker, Kubernetes, CI/CD, Linux, CloudFormation",
        "description": "Lead enterprise cloud migration to Microsoft Azure. Implement identity governance, zero-trust network boundaries, and automated infrastructure as code with Terraform."
    },
    {
        "title": "Junior DevOps Engineer",
        "company": "StartUp Ops",
        "category": "DevOps & Cloud Engineering",
        "location": "Austin, TX (Remote)",
        "experience_level": "Entry-Level",
        "required_skills": "Linux, Docker, Git, CI/CD, Bash, Python, AWS",
        "description": "Support engineering workflows by containerizing microservices with Docker, assisting with cloud provisioning on AWS, and maintaining CI/CD pipeline health."
    },

    # 4. Cybersecurity & Network Engineering
    {
        "title": "Cybersecurity Analyst",
        "company": "FinSecure Corp",
        "category": "Cybersecurity & Network Engineering",
        "location": "Washington, DC (Hybrid)",
        "experience_level": "Mid-Level",
        "required_skills": "SIEM, Splunk, Incident Response, Network Security, Firewalls, Wireshark, Linux",
        "description": "Monitor and defend enterprise infrastructure from malicious intrusions using Splunk SIEM and packet capture tools. Conduct real-time threat triage and incident response investigations."
    },
    {
        "title": "Penetration Tester / Ethical Hacker",
        "company": "RedTeam Offense",
        "category": "Cybersecurity & Network Engineering",
        "location": "San Francisco, CA (Remote)",
        "experience_level": "Senior",
        "required_skills": "Penetration Testing, Ethical Hacking, Metasploit, Burp Suite, OWASP, Python, Linux",
        "description": "Perform authorized red team assessments, web application penetration tests, and network vulnerability exploits. Author actionable remediation reports for C-suite and engineering leaders."
    },
    {
        "title": "Network Security Engineer",
        "company": "CyberShield Networks",
        "category": "Cybersecurity & Network Engineering",
        "location": "Dallas, TX (On-site)",
        "experience_level": "Mid-Level",
        "required_skills": "Network Security, Firewalls, VPN, TCP/IP, DNS, Wireshark, Linux",
        "description": "Design, configure, and maintain secure enterprise routing, virtual private networks (VPN), next-gen Palo Alto firewalls, and cryptographic tunnels for distributed corporate offices."
    },
    {
        "title": "Information Security Officer",
        "company": "TrustVault Financial",
        "category": "Cybersecurity & Network Engineering",
        "location": "Charlotte, NC (Hybrid)",
        "experience_level": "Senior",
        "required_skills": "Information Security, Vulnerability Assessment, IAM, OWASP, Cryptography, Firewalls",
        "description": "Oversee enterprise security compliance, vulnerability management, identity and access management (IAM) lifecycle, and cryptographic key governance across core banking systems."
    },
    {
        "title": "Junior SOC Analyst",
        "company": "DefendPoint Security",
        "category": "Cybersecurity & Network Engineering",
        "location": "Tampa, FL (Remote)",
        "experience_level": "Entry-Level",
        "required_skills": "Network Security, Wireshark, Linux, TCP/IP, Incident Response",
        "description": "Analyze security alerts in a 24/7 Security Operations Center (SOC). Escalate suspicious activities, examine endpoint logs, and assist with containment procedures."
    },

    # 5. Mobile Application Development
    {
        "title": "Android Software Engineer",
        "company": "AppNova Studios",
        "category": "Mobile Application Development",
        "location": "Los Angeles, CA (Remote)",
        "experience_level": "Mid-Level",
        "required_skills": "Android, Kotlin, Jetpack Compose, Android Studio, SQLite, REST API, Git",
        "description": "Engineer native Android applications with modern Kotlin and declarative Jetpack Compose UI. Implement offline-first local persistence with Room/SQLite and asynchronous coroutines."
    },
    {
        "title": "iOS Application Developer",
        "company": "Cupertino Apps",
        "category": "Mobile Application Development",
        "location": "San Diego, CA (Hybrid)",
        "experience_level": "Mid-Level",
        "required_skills": "iOS, Swift, SwiftUI, Xcode, Mobile UI, REST API, Git",
        "description": "Create engaging, performant consumer iOS applications in Swift and SwiftUI. Integrate Apple ecosystem APIs, push notifications, and background tasks while adhering to Human Interface Guidelines."
    },
    {
        "title": "Cross-Platform Flutter Developer",
        "company": "OmniMobile Labs",
        "category": "Mobile Application Development",
        "location": "Miami, FL (Remote)",
        "experience_level": "Mid-Level",
        "required_skills": "Flutter, Dart, Mobile UI, REST API, SQLite, Git, Firebase",
        "description": "Build high-performance cross-platform mobile apps for iOS and Android using Flutter and Dart. Manage complex application state with Bloc/Riverpod and optimize native platform bridges."
    },
    {
        "title": "React Native Mobile Engineer",
        "company": "Verve Digital",
        "category": "Mobile Application Development",
        "location": "Salt Lake City, UT (Remote)",
        "experience_level": "Senior",
        "required_skills": "React Native, JavaScript, TypeScript, Redux, Mobile UI, REST API, Git",
        "description": "Lead the development of flagship hybrid mobile applications in React Native and TypeScript. Integrate native device modules, in-app purchases, and real-time WebSockets."
    },

    # 6. UI/UX Design & Frontend
    {
        "title": "Lead UI/UX Product Designer",
        "company": "PixelCraft Studios",
        "category": "UI/UX Design & Frontend",
        "location": "San Francisco, CA (Remote)",
        "experience_level": "Senior",
        "required_skills": "Figma, UI Design, UX Design, Prototyping, Wireframing, User Research, Design Systems",
        "description": "Direct end-to-end user experience design for enterprise SaaS products. Conduct user research interviews, establish multi-brand Figma design systems, and author interactive high-fidelity prototypes."
    },
    {
        "title": "UI Designer & Design Technologist",
        "company": "Canvas Design Co",
        "category": "UI/UX Design & Frontend",
        "location": "Portland, OR (Hybrid)",
        "experience_level": "Mid-Level",
        "required_skills": "Figma, CSS, Responsive Design, Prototyping, HTML, Wireframing, Adobe XD",
        "description": "Bridge the gap between product design and engineering by translating Figma wireframes into responsive CSS and interactive prototypes. Conduct usability testing sessions."
    },
    {
        "title": "Frontend UI Developer",
        "company": "DesignFirst Interactive",
        "category": "UI/UX Design & Frontend",
        "location": "Austin, TX (Remote)",
        "experience_level": "Entry-Level",
        "required_skills": "HTML5, CSS3, JavaScript, Tailwind CSS, Responsive Design, Figma, Git",
        "description": "Implement pixel-perfect, accessible web interfaces from Figma designs using semantic HTML5, modern CSS3, and Tailwind CSS. Ensure cross-browser visual fidelity and smooth animations."
    },
    {
        "title": "UX Researcher & Strategist",
        "company": "HumanCentric Lab",
        "category": "UI/UX Design & Frontend",
        "location": "Seattle, WA (Remote)",
        "experience_level": "Mid-Level",
        "required_skills": "User Research, Usability Testing, Wireframing, UX Design, Figma, Interaction Design",
        "description": "Plan and execute qualitative and quantitative user experience studies. Map user journey flows, identify pain points, and collaborate with product managers to validate hypotheses."
    },

    # 7. Quality Assurance & Testing
    {
        "title": "Senior QA Automation Engineer",
        "company": "QualityEdge Labs",
        "category": "Quality Assurance & Testing",
        "location": "Minneapolis, MN (Remote)",
        "experience_level": "Senior",
        "required_skills": "Selenium, Cypress, Python, Automated Testing, API Testing, Postman, CI/CD, Jira",
        "description": "Build automated end-to-end testing frameworks using Cypress, Selenium, and Python. Integrate regression suites into continuous deployment pipelines and mentor junior QA analysts."
    },
    {
        "title": "Software Test Engineer (SDET)",
        "company": "Apex Fintech",
        "category": "Quality Assurance & Testing",
        "location": "New York, NY (Hybrid)",
        "experience_level": "Mid-Level",
        "required_skills": "Selenium, Java, JUnit, API Testing, Postman, Automated Testing, Git",
        "description": "Architect test automation harnesses in Java and JUnit for financial microservices. Execute rigorous REST API contract testing and generate detailed defect reports in Jira."
    },
    {
        "title": "Performance Test Engineer",
        "company": "ScaleVerify Inc",
        "category": "Quality Assurance & Testing",
        "location": "Phoenix, AZ (Remote)",
        "experience_level": "Mid-Level",
        "required_skills": "JMeter, Performance Testing, Automated Testing, Postman, Linux, Python",
        "description": "Simulate high concurrency loads on distributed platforms using Apache JMeter. Analyze server bottlenecks, response time degradation, and database connection pool contention."
    },
    {
        "title": "Junior QA Analyst",
        "company": "BetaTest Hub",
        "category": "Quality Assurance & Testing",
        "location": "Nashville, TN (On-site)",
        "experience_level": "Entry-Level",
        "required_skills": "Manual Testing, Bug Tracking, Jira, Postman, Regression Testing, Test Planning",
        "description": "Author structured test plans and test cases. Execute manual functional and regression testing across staging builds, verify bug fixes, and collaborate with developers in Jira."
    },

    # 8. Data Engineering & Big Data
    {
        "title": "Lead Data Platform Engineer",
        "company": "DataStream Matrix",
        "category": "Data Engineering & Big Data",
        "location": "Boston, MA (Remote)",
        "experience_level": "Senior",
        "required_skills": "Apache Spark, PySpark, Kafka, Python, Snowflake, Apache Airflow, ETL, SQL",
        "description": "Construct petabyte-scale real-time data pipelines using Kafka and Apache Spark. Orchestrate DAG workflows in Apache Airflow and optimize dimensional models in Snowflake."
    },
    {
        "title": "Big Data Pipeline Developer",
        "company": "InsightFlow Analytics",
        "category": "Data Engineering & Big Data",
        "location": "Denver, CO (Hybrid)",
        "experience_level": "Mid-Level",
        "required_skills": "PySpark, Hadoop, PostgreSQL, Apache Airflow, ETL, Data Pipelines, Python, SQL",
        "description": "Develop and maintain robust ETL data ingestion routines. Cleanse unstructured logs into normalized analytics warehouses and collaborate with BI analysts."
    },
    {
        "title": "Cloud Data Warehouse Architect",
        "company": "Nexus Cloud Data",
        "category": "Data Engineering & Big Data",
        "location": "San Jose, CA (Remote)",
        "experience_level": "Senior",
        "required_skills": "Snowflake, Databricks, dbt, SQL, Python, BigQuery, Data Warehousing, Data Pipelines",
        "description": "Design modern data stack architectures combining Databricks Lakehouse, dbt transformations, and Snowflake data warehouses. Enforce data quality metrics and data governance."
    },
    {
        "title": "Junior Data Engineer",
        "company": "DataPulse Corp",
        "category": "Data Engineering & Big Data",
        "location": "Philadelphia, PA (Remote)",
        "experience_level": "Entry-Level",
        "required_skills": "Python, SQL, PostgreSQL, ETL, Data Pipelines, Git",
        "description": "Write automated Python scripts to pull data from third-party REST APIs and flat files into PostgreSQL databases. Write SQL views and assist in data validation."
    }
]


# Training dataset generator: 16-20 rich resume samples per category
RESUME_SAMPLES_BY_CATEGORY = {
    "Data Science & Machine Learning": [
        "Data Scientist with 4 years experience developing machine learning models using Python, Scikit-Learn, Pandas, and NumPy. Built churn prediction models with XGBoost achieving 88% precision. Trained deep neural networks in PyTorch for tabular and time-series forecasting. Solid statistics, data visualization with Matplotlib, and SQL query optimization. Deployed inference APIs via Flask and Docker.",
        "Machine Learning Engineer proficient in TensorFlow, Keras, and PyTorch. Implemented computer vision pipelines using OpenCV and YOLO for automated defect detection on factory lines. Strong background in linear algebra, predictive modeling, feature engineering, and hyperparameter tuning with Optuna. Git and MLflow for experiment tracking.",
        "NLP Researcher and Data Scientist specializing in Natural Language Processing, Hugging Face transformers, BERT, and LLMs. Developed generative AI document summarizers and sentiment classification pipelines in Python. Proficient in PyTorch, NLTK, Spacy, vector embeddings, and LangChain. Master's in Computer Science.",
        "Junior Data Analyst transitioning to Machine Learning. Strong foundation in Python, Pandas, NumPy, Scikit-Learn, and statistical inference. Conducted exploratory data analysis on customer transaction datasets, built linear regression and decision tree models. Visualized insights with Seaborn and Tableau.",
        "Senior AI Scientist with expertise in deep learning, reinforcement learning, and neural network architectures. Published research on generative models. Proficient in PyTorch, Python, high-performance computing, distributed training, Scikit-Learn, and Docker. Led a team of 4 data scientists.",
        "Data Science practitioner with proven success in predictive modeling and customer segmentation. Utilized Python, Scikit-Learn, K-Means clustering, Random Forest, and LightGBM. Extensive experience cleaning noisy datasets, handling missing values, and engineering predictive features.",
        "Computer Vision Engineer specializing in deep learning, image segmentation, object tracking, and OpenCV. Built real-time video analytics models using PyTorch, TensorFlow, and CUDA. Deployed models to edge devices using ONNX and TensorRT.",
        "Applied Machine Learning Specialist experienced in fraud detection and risk assessment. Trained supervised classification algorithms using Scikit-Learn, XGBoost, and Logistic Regression. Automated model evaluation and cross-validation pipelines in Python.",
        "AI / Deep Learning Engineer focused on multi-modal models, transformer architectures, and PyTorch. Fine-tuned vision-language models and scaled training across GPU clusters. Proficient in Python, Linux, Git, and Docker.",
        "Data Scientist with strong mathematics and probability background. Formulated A/B testing methodologies and customer lifetime value models. Proficient in Python, SQL, Pandas, Scikit-Learn, Statsmodels, and Jupyter Notebooks.",
        "Machine Learning Intern with academic projects in classification and NLP. Implemented sentiment analysis using Scikit-Learn TF-IDF and Logistic Regression. Handled data preprocessing, tokenization, stopword removal, and accuracy metric evaluation in Python.",
        "Senior Machine Learning Engineer with 6 years experience architecting ML platforms. Deployed deep learning models to production using FastAPI, Triton Inference Server, and Docker. Extensive expertise in PyTorch, Scikit-Learn, and continuous model monitoring.",
        "Data Science graduate student with hands-on projects in biomedical image classification. Applied convolutional neural networks with PyTorch and data augmentation. Skilled in Python, Scikit-Learn, NumPy, Matplotlib, and statistical hypothesis testing.",
        "Predictive Modeling Analyst utilizing Python, Scikit-Learn, and SQL. Built sales forecasting models with ARIMA and gradient boosting regressors. Skilled in data visualization, stakeholder presentations, and pipeline reproducibility.",
        "AI Solutions Developer experienced in Natural Language Processing, text summarization, entity extraction, and conversational agents using Hugging Face, Spacy, and Python. Engineered prompt templates and evaluated model hallucinations."
    ],
    "Full Stack Web Development": [
        "Full Stack Developer with 4 years experience in JavaScript, TypeScript, React, and Node.js. Built scalable RESTful microservices with Express.js and PostgreSQL. Implemented Redux for client state management, responsive UI with Tailwind CSS, and unit testing with Jest.",
        "Web Developer experienced in Python, Django, and modern frontend technologies. Developed end-to-end e-commerce applications with relational database modeling in PostgreSQL. Implemented JWT authentication, payment gateway integration, and Docker containerization.",
        "MERN Stack Developer proficient in MongoDB, Express.js, React, and Node.js. Architected high-traffic REST APIs, implemented real-time notifications with Socket.io, and built reusable component libraries. Strong grasp of HTML5, CSS3, and modern ES6+ JavaScript.",
        "Senior Full Stack Engineer with 7 years building SaaS applications. Skilled in React, Next.js, TypeScript, GraphQL, Node.js, and Redis caching. Led architectural migration from monolith to microservices on AWS. Strong advocate for clean code and CI/CD.",
        "Java Full Stack Developer with expertise in Java, Spring Boot, Hibernate, and Angular. Engineered enterprise banking portals with secure REST APIs, role-based access control, and MySQL database persistence. Managed builds using Maven and Git.",
        "Frontend-heavy Full Stack Developer proficient in React, Vue.js, JavaScript, HTML, CSS, and Node.js. Crafted responsive web applications with mobile-first methodologies. Integrated third-party APIs and authored modular client-side logic.",
        "Python Full Stack Engineer specializing in FastAPI, SQLAlchemy, PostgreSQL, and React. Created async backend microservices handling 5,000 requests per second. Built clean administrative dashboards with Tailwind CSS.",
        "Junior Web Developer with strong foundation in JavaScript, HTML5, CSS3, React, and Git. Built personal portfolio projects, weather dashboard apps, and task management tools using Express.js and MongoDB.",
        "Full Stack Web Engineer experienced in PHP, Laravel, MySQL, and Vue.js. Maintained legacy enterprise web apps, refactored monolithic controllers into service layers, and integrated Stripe payment processing.",
        "Backend & Web Systems Developer with expertise in Node.js, TypeScript, PostgreSQL, and Docker. Implemented OAuth2 social authentication, database migrations, and OpenAPI documentation.",
        "Full Stack Application Engineer specializing in React, Next.js, Node.js, and Prisma ORM. Optimized page speed scores and server-side rendering for marketing websites.",
        "Software Engineer with 3 years developing responsive web applications using Angular, TypeScript, Node.js, and MongoDB. Practiced agile development, code reviews, and automated integration testing.",
        "Full Stack Developer proficient in Ruby on Rails, PostgreSQL, JavaScript, and Bootstrap. Built rapid prototypes for seed-stage startups, managed database schema migrations, and authored RSpec tests.",
        "ASP.NET Core Full Stack Developer skilled in C#, .NET Core, SQL Server, and React. Built enterprise management systems with Entity Framework Core, LINQ queries, and secure Web API endpoints.",
        "Web Applications Engineer experienced in building collaborative whiteboard tools using WebSockets, React, Canvas API, and Node.js backends. Proficient in Git, Docker, and RESTful architectures."
    ],
    "DevOps & Cloud Engineering": [
        "DevOps Engineer with 5 years experience managing cloud infrastructure on AWS. Proficient in Terraform, Docker, Kubernetes, and Linux systems administration. Automated continuous integration and deployment pipelines using GitLab CI and Jenkins. Monitored clusters with Prometheus and Grafana.",
        "Cloud Solutions Architect specializing in Microsoft Azure and multi-cloud environments. Implemented infrastructure as code (IaC) with Terraform and Azure Resource Manager. Configured virtual networks, application gateways, and container registries.",
        "Site Reliability Engineer (SRE) focused on system resilience and high availability. Wrote automation scripts in Python, Bash, and Go. Maintained Kubernetes clusters, resolved production alerts, and conducted root cause analysis post-mortems.",
        "Infrastructure Automation Engineer experienced in AWS, Docker, Kubernetes, Helm, Ansible, and Terraform. Migrated on-premise monolithic applications to containerized cloud architectures, reducing server costs by 35%.",
        "CI/CD Pipeline Specialist with expertise in Jenkins, GitHub Actions, Docker, and Linux. Created automated build, test, security scanning, and deployment stages across 50+ repositories.",
        "Junior Cloud Engineer with certifications in AWS Solutions Architect Associate. Hands-on experience provisioning EC2 instances, S3 buckets, RDS databases, and IAM policies. Proficient in Linux command line and Bash scripting.",
        "Platform Engineer experienced in building developer platforms on top of Kubernetes and AWS. Configured service meshes with Istio, ingress controllers with Nginx, and centralized logging with ELK Stack.",
        "Cloud Security and DevOps Specialist with expertise in Terraform, AWS, Docker, and Linux. Implemented CIS benchmark security hardening, secret management with HashiCorp Vault, and vulnerability scanning in CI/CD.",
        "Systems Administrator transitioning to DevOps. 6 years managing Linux servers, network configurations, DNS, and Apache/Nginx web servers. Automated server configuration using Ansible and Bash.",
        "Kubernetes Administrator proficient in Docker, K8s, Helm, Prometheus, and Linux. Managed production Kubernetes clusters across multiple availability zones, implemented horizontal pod autoscaling.",
        "DevOps Consultant specializing in GCP (Google Cloud Platform), Terraform, and Cloud Build. Architected serverless deployments on Google Cloud Run and Cloud Functions.",
        "Release Engineer with extensive experience in automated deployment strategies, blue-green deployments, canary releases, and Git branching workflows. Proficient in Docker, Linux, and Jenkins.",
        "Cloud Infrastructure Engineer experienced with OpenShift, Kubernetes, Linux, and Ansible. Orchestrated enterprise hybrid cloud infrastructure and automated backup disaster recovery routines.",
        "DevOps Engineer focusing on observability and telemetry. Configured Grafana dashboards, Prometheus metrics, and distributed tracing with OpenTelemetry for distributed microservices on AWS.",
        "Site Reliability Engineer with deep experience in Linux kernel tuning, TCP/IP networking, shell scripting, Python automation, and automated recovery mechanisms."
    ],
    "Cybersecurity & Network Engineering": [
        "Cybersecurity Analyst with 4 years experience in Security Operations Center (SOC). Monitored enterprise traffic using Splunk SIEM, investigated alert anomalies, and conducted incident response. Skilled in Wireshark, network protocols, firewalls, and malware triage.",
        "Penetration Tester and Ethical Hacker with CEH and OSCP certifications. Conducted web application penetration testing using Burp Suite and OWASP Top 10 methodology. Performed network vulnerability assessments with Metasploit, Nmap, and Nessus.",
        "Network Security Engineer experienced in configuring Cisco and Palo Alto firewalls, VPN tunnels, and intrusion detection systems (IDS/IPS). Deep understanding of TCP/IP, DNS, subnetting, and routing protocols (BGP, OSPF).",
        "Information Security Specialist focused on governance, risk, and compliance (GRC), ISO 27001, and NIST frameworks. Performed enterprise risk assessments, identity and access management (IAM) audits, and security policy authoring.",
        "Junior SOC Analyst skilled in log analysis, threat intelligence, and endpoint detection and response (EDR). Proficient in Wireshark packet capture, Linux commands, and security incident escalation procedures.",
        "Application Security Engineer with expertise in secure code review, SAST, DAST, and OWASP mitigation. Collaborated with developers to remediate SQL injection, XSS, and authentication bypass vulnerabilities in Python and JavaScript apps.",
        "Security Architect with 8 years experience designing zero-trust enterprise networks. Implemented public key infrastructure (PKI), TLS cryptography, multi-factor authentication (MFA), and secure network enclaves.",
        "Incident Responder and Threat Hunter utilizing SIEM, EDR, and memory forensics tools (Volatility). Investigated phishing campaigns, ransomware infections, and advanced persistent threat (APT) indicators of compromise.",
        "Network Infrastructure Engineer with CCNA certification. Configured enterprise switches, routers, VLANs, and firewall access control lists (ACLs). Troubleshot complex network outages with Wireshark and traceroute.",
        "Vulnerability Management Analyst experienced in Qualys, Nessus, and Rapid7 scanners. Prioritized vulnerability remediation based on CVSS scoring and threat intelligence feeds. Reported risk metrics to executive leadership.",
        "Cloud Security Engineer specializing in AWS IAM policies, GuardDuty, Security Hub, and CloudTrail audit logging. Ensured automated compliance and encryption of data at rest and in transit.",
        "Digital Forensics and Incident Response (DFIR) Specialist. Extracted digital evidence from Windows and Linux artifacts, preserved chain of custody, and authored court-admissible forensic reports.",
        "Cybersecurity Researcher with hands-on experience in cryptography, reverse engineering, and assembly language. Analyzed malware samples in sandbox environments to identify command-and-control communication.",
        "Security Operations Analyst proficient in Splunk, QRadar, Wazuh, and Python scripting for security automation. Built custom detection rules and automated threat alert triage.",
        "Network Administrator and Security Specialist managing enterprise VPNs, firewalls, radius servers, and wireless security. Monitored network bandwidth and thwarted DDoS attempts."
    ],
    "Mobile Application Development": [
        "Android Developer with 4 years experience writing native applications in Kotlin and Java. Built responsive user interfaces with Jetpack Compose and XML layouts. Implemented MVVM architecture, Room database for local caching, Coroutines, and Retrofit for REST APIs.",
        "iOS Software Engineer specializing in Swift and SwiftUI. Developed consumer mobile apps with seamless animations, Apple Pay integration, and CoreData persistence. Proficient in Xcode, TestFlight, and App Store submission processes.",
        "Cross-Platform Mobile Engineer with expertise in Flutter and Dart. Engineered offline-first mobile applications for iOS and Android. Implemented state management using Bloc and Provider, integrated Firebase push notifications and SQLite.",
        "React Native Developer with 3 years experience building high-performance hybrid mobile apps using TypeScript, React Native, and Redux. Optimized native bridge communication, integrated camera and geolocation sensors.",
        "Junior Android Developer with solid understanding of Kotlin, Android Studio, and Android SDK. Created personal projects including a habit tracker and recipe app with SQLite storage and Material Design components.",
        "Senior Mobile Architect with 7 years leading iOS and Android development teams. Guided architectural patterns (Clean Architecture, MVVM), CI/CD mobile pipelines with Fastlane, and automated UI testing.",
        "Mobile UI/UX Developer focusing on Flutter, custom animations, and responsive mobile layouts across varying screen densities. Integrated GraphQL queries and local biometric authentication.",
        "iOS Developer experienced in Objective-C to Swift codebase migration. Implemented AVFoundation for audio playback, CoreLocation for mapping, and unit tests with XCTest.",
        "Android Engineer proficient in Kotlin, Dagger-Hilt dependency injection, Flow, and Jetpack Navigation component. Refactored legacy codebases into modern modular multi-module architectures.",
        "Flutter App Developer with extensive experience in building e-commerce and fintech mobile apps. Integrated payment SDKs, deep linking, internationalization, and SQLite local storage in Dart.",
        "Mobile Games and Interactive App Developer using Unity, C#, and native Android/iOS integration. Optimized rendering performance, frame rates, and mobile memory footprints.",
        "Cross-Platform Developer experienced in React Native and Expo. Delivered 5 mobile apps to Apple App Store and Google Play Store with automated over-the-air (OTA) updates.",
        "Android Software Engineer with background in hardware peripheral integration via Bluetooth Low Energy (BLE) and USB on Android devices with Kotlin and C++ NDK.",
        "Mobile Application QA & Developer testing mobile apps on real devices and emulators. Automated UI tests with Appium and Espresso in Kotlin and Python.",
        "Junior iOS Developer skilled in Swift, SwiftUI, Xcode, and Git. Built weather and fitness tracking apps consuming public REST APIs and utilizing UserDefaults and SwiftData."
    ],
    "UI/UX Design & Frontend": [
        "Senior UI/UX Designer with 6 years experience crafting user-centered web and mobile experiences. Expert in Figma, wireframing, interactive prototyping, and comprehensive design systems. Led user research, usability testing, and persona creation.",
        "UI Designer and Design Technologist adept in Figma, Adobe XD, HTML5, CSS3, and responsive design. Created atomic design component libraries in Figma and translated them into clean Tailwind CSS code.",
        "UX Researcher specializing in qualitative interviews, usability testing, heuristic evaluations, and journey mapping. Collaborated with product managers to transform user feedback into actionable wireframes in Figma.",
        "Frontend Designer proficient in CSS3, Sass, Tailwind CSS, JavaScript, and HTML. Developed responsive landing pages with high conversion rates, smooth micro-interactions, and accessibility (WCAG) compliance.",
        "Product Designer with expertise in Figma, design tokens, responsive typography, and mobile app prototyping. Partnered with engineering teams to conduct design QA and ensure pixel-perfect implementation.",
        "Junior UI/UX Designer with a strong portfolio of web and mobile app redesigns. Proficient in Figma, user flows, wireframes, moodboards, and interactive prototypes.",
        "Interaction Designer focused on micro-animations, transitions, and user delight using Figma, Principle, and CSS animations. Designed intuitive interfaces for complex SaaS workflow tools.",
        "Visual & Graphic UI Designer with expertise in branding, typography, color theory, iconography, and illustration. Designed cohesive visual design systems for fintech applications in Figma and Adobe Illustrator.",
        "UX Architect skilled in information architecture, card sorting, user personas, sitemaps, and low-to-high fidelity wireframing in Figma and Miro. Championed accessibility standards across products.",
        "Web Designer & Frontend Integrator with strong command of HTML, CSS, Bootstrap, Figma, and basic JavaScript. Built modern landing pages, email templates, and corporate website redesigns.",
        "Product Designer experienced in e-commerce checkout flow optimization, A/B testing user interface variants, and improving user retention metrics through iterative UX enhancements.",
        "Design Systems Specialist who created and maintained multi-brand Figma design systems with automated documentation for 50+ engineers and designers across global teams.",
        "Junior UX Designer skilled in usability testing, user journey mapping, wireframing in Figma, and conducting user interviews to identify user friction points.",
        "Creative UI/UX Designer combining visual aesthetics with analytical thinking. Skilled in Figma, Adobe XD, user research, rapid prototyping, and stakeholder presentations.",
        "Frontend & UI Stylist experienced in modern CSS Grid, Flexbox, CSS animations, Figma, and responsive web design. Ensured cross-browser visual consistency and high performance."
    ],
    "Quality Assurance & Testing": [
        "Senior QA Automation Engineer with 5 years experience designing test frameworks using Selenium WebDriver, Python, and PyTest. Automated REST API testing with Postman and Requests. Integrated regression suites into Jenkins CI/CD pipelines.",
        "Software Development Engineer in Test (SDET) proficient in Java, Selenium, TestNG, and Cucumber BDD. Built scalable test automation architectures for microservices, reducing manual regression testing by 70%.",
        "QA Engineer specializing in modern frontend testing with Cypress and Playwright in TypeScript. Experienced in automated end-to-end tests, visual regression testing, and CI execution on GitHub Actions.",
        "Manual and Automated Test Analyst with strong expertise in Jira, test case design, defect life cycle, and exploratory testing. Wrote API test scripts in Postman and performed SQL database verification.",
        "Performance and Load Test Engineer using Apache JMeter to simulate thousands of concurrent user journeys. Analyzed response latencies, server resource bottlenecks, and throughput limits.",
        "Junior QA Tester with experience executing functional, smoke, and regression test cases. Documented reproducible bug reports in Jira with clear steps, logs, and screenshots.",
        "Mobile QA Automation Engineer testing Android and iOS applications using Appium and Python. Executed cross-device automated tests on cloud device farms like BrowserStack.",
        "Quality Assurance Lead managing a team of 5 test engineers. Defined overall test strategy, authored comprehensive test plans, and reported quality metrics to engineering directors.",
        "API Test Automation Engineer experienced with Postman Newman CLI, REST Assured, and Python. Validated JSON schema contracts, status codes, response payloads, and edge test cases.",
        "Software QA Analyst with experience in financial systems. Performed rigorous end-to-end testing, user acceptance testing (UAT), regression testing, and data validation using SQL queries.",
        "QA Engineer skilled in Selenium, Python, Git, and Linux. Created modular page object model (POM) automated suites and generated HTML test execution reports.",
        "Continuous Testing and SDET specialist who embedded automated security, performance, and unit testing into GitLab CI pipelines using Docker containers and PyTest.",
        "Junior Software Tester with passion for bug hunting, test case writing, exploratory testing, and Jira issue tracking. Assisted developers with defect verification and reproduction.",
        "Automation Engineer focusing on backend microservices verification. Used Python, PyTest, Docker, and PostgreSQL mocks to run isolated integration tests.",
        "QA Automation Specialist with expertise in Cypress, JavaScript, Postman, and Jira. Conducted automated UI and API regressions for e-commerce platforms."
    ],
    "Data Engineering & Big Data": [
        "Data Engineer with 5 years experience building distributed data pipelines using Apache Spark, PySpark, and Python. Orchestrated daily ETL workflows with Apache Airflow. Built high-scale data warehouses in Snowflake and Amazon Redshift.",
        "Big Data Architect specializing in real-time streaming architectures with Apache Kafka, Spark Streaming, and Cassandra. Designed fault-tolerant message queues handling 50,000 events/sec.",
        "Cloud Data Warehouse Engineer proficient in Snowflake, dbt, SQL, and Python. Created normalized dimensional models (Star/Snowflake schema), automated data transformations, and enforced data quality tests.",
        "ETL Developer with extensive experience in Python, SQL, PostgreSQL, and Apache Airflow. Extracted data from REST APIs, SFTP servers, and relational databases into central data lakes on AWS S3.",
        "Junior Data Engineer with strong Python and SQL skills. Assisted in building data ingestion pipelines, writing database views, and cleaning messy tabular data for business intelligence reporting.",
        "Data Platform Engineer experienced with Databricks, Apache Spark, Lakehouse architecture, and Delta Lake. Optimized Spark jobs, resolved memory skew, and partitioned parquet datasets.",
        "Senior Data Engineer with expertise in Hadoop, Hive, Presto, PySpark, and Linux. Migrated legacy on-premise Hadoop clusters to AWS cloud-native analytics services.",
        "Analytics Engineer bridging the gap between raw data and business dashboards. Expert in dbt, BigQuery, SQL, and data modeling. Implemented automated data testing and documentation.",
        "Database Administrator & Data Engineer skilled in PostgreSQL, MySQL, database indexing, query execution plan tuning, partitioning, and automated backup routines.",
        "Real-Time Stream Processing Engineer utilizing Kafka, Python, Docker, and Redis. Built event-driven ingestion layers for IoT sensor telemetry and fraud detection streams.",
        "Data Pipeline Specialist experienced in building serverless ETL workflows with AWS Glue, Athena, S3, and Lambda in Python. Monitored pipeline runs using CloudWatch.",
        "Big Data Engineer with 4 years experience in PySpark, Scala, and Cassandra. Ingested large volumes of clickstream data into analytical repositories for machine learning teams.",
        "Data Integration Developer proficient in SQL, Python, REST APIs, and Apache NiFi. Automated data synchronizations between enterprise CRMs and relational data warehouses.",
        "Junior ETL Developer skilled in writing complex SQL queries, stored procedures, Python scripts, and building automated data cleansing jobs in PostgreSQL.",
        "Data Infrastructure Engineer experienced in managing Apache Airflow clusters, configuring Celery workers, and managing distributed data storage on S3 and HDFS."
    ]
}


def generate_jobs_csv(filepath: str = os.path.join(DATA_DIR, "jobs_dataset.csv")) -> pd.DataFrame:
    """Generate jobs_dataset.csv with >= 30 jobs and populate SQLite database."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df = pd.DataFrame(JOBS_DATA)
    df.to_csv(filepath, index=False)
    populate_jobs_from_dataframe(df)
    return df


def generate_resumes_csv(filepath: str = os.path.join(DATA_DIR, "resumes_dataset.csv")) -> pd.DataFrame:
    """Generate labeled resumes dataset with 120+ samples across 8 categories."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    rows = []
    for category, resumes in RESUME_SAMPLES_BY_CATEGORY.items():
        for idx, resume_text in enumerate(resumes):
            rows.append({
                "id": f"{category[:3].upper()}_{idx+1:03d}",
                "category": category,
                "resume_text": resume_text
            })
    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)
    return df


def generate_sample_pdf_resumes(output_dir: str = SAMPLE_RESUMES_DIR) -> list:
    """
    Generate clean, realistic PDF resumes for live testing in Streamlit.
    PyMuPDF creates valid, professional PDF documents.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    test_resumes = [
        {
            "filename": "sample_data_scientist.pdf",
            "name": "Alex Chen",
            "title": "Machine Learning Engineer & Data Scientist",
            "email": "alex.chen.ml@example.com",
            "phone": "+1-555-019-2834",
            "github": "https://github.com/alexchen-ml",
            "linkedin": "https://linkedin.com/in/alexchen-data",
            "category": "Data Science & Machine Learning",
            "summary": "Experienced Data Scientist with 4+ years of expertise in designing predictive machine learning algorithms, deep learning models, and NLP pipelines. Proven record of developing end-to-end computer vision and LLM applications using PyTorch and Scikit-Learn.",
            "skills": "Python, Machine Learning, Deep Learning, Scikit-Learn, PyTorch, TensorFlow, Pandas, NumPy, OpenCV, NLP, SQL, Docker, Git, Matplotlib",
            "experience": [
                "Senior AI Engineer - Apex Innovations (2022 - Present): Designed and deployed transformer-based NLP models with PyTorch and Hugging Face. Improved inference speed by 40% with Docker and FastAPI.",
                "Data Scientist - Quantix Data Corp (2020 - 2022): Developed tabular predictive models with Scikit-Learn and XGBoost for customer retention analysis. Conducted A/B testing and statistical feature selection in Python."
            ],
            "education": "M.S. in Computer Science - Stanford University | B.S. in Applied Mathematics"
        },
        {
            "filename": "sample_fullstack_dev.pdf",
            "name": "Sarah Miller",
            "title": "Full Stack Web Developer",
            "email": "sarah.miller.dev@example.com",
            "phone": "+1-555-014-8831",
            "github": "https://github.com/sarahmiller-dev",
            "linkedin": "https://linkedin.com/in/sarahmiller-web",
            "category": "Full Stack Web Development",
            "summary": "Dynamic Full Stack Developer proficient in building responsive, scalable web applications with React, Node.js, Express.js, and PostgreSQL. Experienced in RESTful microservices, state management, and modern cloud deployment.",
            "skills": "JavaScript, TypeScript, React, Node.js, Express.js, PostgreSQL, MongoDB, HTML5, CSS3, Tailwind CSS, REST API, Git, Docker, Jest",
            "experience": [
                "Full Stack Developer - CloudNexus Labs (2021 - Present): Developed e-commerce portal with React, TypeScript, and Node.js microservices. Engineered relational database schemas in PostgreSQL with automated migrations.",
                "Junior Web Developer - PixelWorks Studio (2019 - 2021): Built interactive single-page applications with React and Tailwind CSS. Implemented user authentication with JWT and secure REST APIs."
            ],
            "education": "B.S. in Software Engineering - University of Washington"
        },
        {
            "filename": "sample_devops_engineer.pdf",
            "name": "Marcus Vance",
            "title": "DevOps & Cloud Infrastructure Engineer",
            "email": "marcus.vance.ops@example.com",
            "phone": "+1-555-017-4920",
            "github": "https://github.com/marcusvance-devops",
            "linkedin": "https://linkedin.com/in/marcusvance-cloud",
            "category": "DevOps & Cloud Engineering",
            "summary": "DevOps Engineer with 5+ years of experience automating CI/CD pipelines and provisioning highly available cloud infrastructure on AWS. Strong expertise in Kubernetes, Docker, Terraform, and Linux administration.",
            "skills": "AWS, Docker, Kubernetes, Terraform, Linux, CI/CD, Jenkins, GitHub Actions, Prometheus, Grafana, Ansible, Bash, Python, Git",
            "experience": [
                "Lead DevOps Engineer - ScaleVelocity Networks (2021 - Present): Architected multi-region Kubernetes clusters on AWS using Terraform. Automated deployments with GitHub Actions and monitored infrastructure with Prometheus & Grafana.",
                "Cloud Systems Administrator - Enterprise Cloud Inc (2018 - 2021): Configured Linux production servers, implemented automated backup policies, and built Docker container images for microservices."
            ],
            "education": "B.S. in Computer Information Systems - Georgia Tech"
        },
        {
            "filename": "sample_cybersecurity_analyst.pdf",
            "name": "Elena Rostova",
            "title": "Cybersecurity Analyst & Penetration Tester",
            "email": "elena.rostova.sec@example.com",
            "phone": "+1-555-012-7749",
            "github": "https://github.com/elenarostova-sec",
            "linkedin": "https://linkedin.com/in/elenarostova-cyber",
            "category": "Cybersecurity & Network Engineering",
            "summary": "Certified Cybersecurity Analyst (CEH, CompTIA Security+) with 4 years experience in Security Operations Center monitoring, SIEM log analysis, network security, and web penetration testing with OWASP standards.",
            "skills": "Network Security, SIEM, Splunk, Wireshark, Penetration Testing, OWASP, Metasploit, Burp Suite, Firewalls, Linux, Incident Response, Python, TCP/IP",
            "experience": [
                "SOC Analyst Tier 2 - FinSecure Defense (2022 - Present): Investigated security incidents and anomalous network traffic using Splunk SIEM and Wireshark. Automated threat intelligence feeds with Python.",
                "Junior Security Analyst - CyberGuard Corp (2020 - 2022): Conducted vulnerability assessments using Nessus and Burp Suite. Assessed firewall rules and prepared incident mitigation reports."
            ],
            "education": "B.S. in Cybersecurity & Network Forensics - Penn State"
        },
        {
            "filename": "sample_mobile_developer.pdf",
            "name": "David Kim",
            "title": "Mobile App Developer (Android & Flutter)",
            "email": "david.kim.mobile@example.com",
            "phone": "+1-555-018-3619",
            "github": "https://github.com/davidkim-apps",
            "linkedin": "https://linkedin.com/in/davidkim-mobile",
            "category": "Mobile Application Development",
            "summary": "Mobile Developer with 4 years experience building native Android applications in Kotlin and cross-platform mobile apps in Flutter. Passionate about declarative UI, clean MVVM architecture, and smooth user experiences.",
            "skills": "Android, Kotlin, Flutter, Dart, Android Studio, SQLite, Mobile UI, Jetpack Compose, REST API, Git, Firebase",
            "experience": [
                "Mobile Software Engineer - OmniApp Studios (2021 - Present): Developed consumer fintech application with Flutter and Dart. Engineered offline data persistence with SQLite and state management with Bloc.",
                "Android Developer - AppPulse Technologies (2019 - 2021): Crafted native Android apps using Kotlin and Jetpack Compose. Integrated REST APIs and implemented biometric authentication."
            ],
            "education": "B.S. in Computer Science - UCLA"
        },
        {
            "filename": "sample_qa_engineer.pdf",
            "name": "Priya Sharma",
            "title": "QA Automation & Test Engineer",
            "email": "priya.sharma.qa@example.com",
            "phone": "+1-555-016-5542",
            "github": "https://github.com/priyasharma-qa",
            "linkedin": "https://linkedin.com/in/priyasharma-test",
            "category": "Quality Assurance & Testing",
            "summary": "Detail-oriented QA Automation Engineer with 4 years of experience building scalable test automation frameworks for web applications and REST APIs using Selenium WebDriver, Cypress, and Python.",
            "skills": "Selenium, Cypress, Python, Automated Testing, API Testing, Postman, Jira, Manual Testing, Performance Testing, JMeter, Git, CI/CD",
            "experience": [
                "Senior QA Engineer - QualityFirst Software (2022 - Present): Architected end-to-end automated testing suites using Cypress and TypeScript. Integrated test runs into GitLab CI/CD pipelines.",
                "QA Analyst - AgileTech Solutions (2020 - 2022): Executed automated and manual testing for e-commerce platforms. Wrote comprehensive API test collections in Postman and tracked defects in Jira."
            ],
            "education": "B.Tech in Information Technology - Delhi Technological University"
        }
    ]

    generated_files = []

    for item in test_resumes:
        pdf_path = os.path.join(output_dir, item["filename"])
        doc = fitz.open()
        page = doc.new_page(width=595, height=842) # A4 size

        # Insert clean text layout
        y = 50
        # Header: Name
        page.insert_text(fitz.Point(50, y), item["name"], fontsize=20, fontname="helv", color=(0.1, 0.2, 0.5))
        y += 22
        # Title
        page.insert_text(fitz.Point(50, y), item["title"], fontsize=12, fontname="helv", color=(0.3, 0.3, 0.3))
        y += 18
        # Contact details
        contact_line = f"Email: {item['email']}  |  Phone: {item['phone']}"
        page.insert_text(fitz.Point(50, y), contact_line, fontsize=9, fontname="helv", color=(0.4, 0.4, 0.4))
        y += 14
        social_line = f"GitHub: {item['github']}  |  LinkedIn: {item['linkedin']}"
        page.insert_text(fitz.Point(50, y), social_line, fontsize=9, fontname="helv", color=(0.4, 0.4, 0.4))
        y += 25

        # Horizontal rule
        page.draw_line(fitz.Point(50, y), fitz.Point(545, y), color=(0.8, 0.8, 0.8), width=1)
        y += 20

        # Section: Professional Summary
        page.insert_text(fitz.Point(50, y), "PROFESSIONAL SUMMARY", fontsize=11, fontname="helv", color=(0.1, 0.2, 0.5))
        y += 16
        # Multi-line summary
        rect_summary = fitz.Rect(50, y, 545, y + 60)
        page.insert_textbox(rect_summary, item["summary"], fontsize=9.5, fontname="helv", color=(0.2, 0.2, 0.2))
        y += 65

        # Section: Technical Skills
        page.insert_text(fitz.Point(50, y), "TECHNICAL SKILLS", fontsize=11, fontname="helv", color=(0.1, 0.2, 0.5))
        y += 16
        rect_skills = fitz.Rect(50, y, 545, y + 45)
        page.insert_textbox(rect_skills, item["skills"], fontsize=9.5, fontname="helv", color=(0.2, 0.2, 0.2))
        y += 50

        # Section: Work Experience
        page.insert_text(fitz.Point(50, y), "WORK EXPERIENCE", fontsize=11, fontname="helv", color=(0.1, 0.2, 0.5))
        y += 18
        for exp in item["experience"]:
            rect_exp = fitz.Rect(50, y, 545, y + 55)
            page.insert_textbox(rect_exp, f"• {exp}", fontsize=9.5, fontname="helv", color=(0.2, 0.2, 0.2))
            y += 58

        # Section: Education
        y += 5
        page.insert_text(fitz.Point(50, y), "EDUCATION", fontsize=11, fontname="helv", color=(0.1, 0.2, 0.5))
        y += 16
        page.insert_text(fitz.Point(50, y), item["education"], fontsize=9.5, fontname="helv", color=(0.2, 0.2, 0.2))

        doc.save(pdf_path)
        doc.close()
        generated_files.append(pdf_path)

    return generated_files


if __name__ == "__main__":
    print("Generating jobs dataset and SQLite database...")
    jobs_df = generate_jobs_csv()
    print(f"-> Generated {len(jobs_df)} jobs in data/jobs_dataset.csv and {DB_PATH}")

    print("Generating training resumes dataset...")
    resumes_df = generate_resumes_csv()
    print(f"-> Generated {len(resumes_df)} resumes in data/resumes_dataset.csv")

    print("Generating sample PDF resumes for live testing...")
    pdfs = generate_sample_pdf_resumes()
    print(f"-> Generated {len(pdfs)} sample PDF files in {SAMPLE_RESUMES_DIR}")
