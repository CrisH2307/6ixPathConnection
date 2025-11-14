import re
SKILL_TO_BUCKET = {
    # project management
    "agile": "project_management",
    "scrum": "project_management",
    "kanban": "project_management",
    "jira": "project_management",
    "confluence": "project_management",
    "stakeholder management": "project_management",
    "project management": "project_management",

    # product / design
    "figma": "product_design",
    "sketch": "product_design",
    "adobe xd": "product_design",
    "wireframing": "product_design",
    "prototyping": "product_design",
    "ux": "product_design",
    "ui": "product_design",
    "usability testing": "product_design",

    # frontend
    "html": "frontend",
    "css": "frontend",
    "sass": "frontend",
    "tailwind": "frontend",
    "javascript": "frontend",
    "typescript": "frontend",
    "react": "frontend",
    "next": "frontend",
    "vue": "frontend",
    "nuxt": "frontend",
    "angular": "frontend",
    "redux": "frontend",

    # backend
    "node": "backend",
    "express": "backend",
    "java": "backend",
    "spring": "backend",
    "python": "backend",
    "django": "backend",
    "flask": "backend",
    "fastapi": "backend",
    ".net": "backend",
    "c#": "backend",
    "go": "backend",
    "rust": "backend",
    "graphql": "backend",
    "rest apis": "backend",
    "grpc": "backend",

    # mobile
    "swift": "mobile",
    "ios": "mobile",
    "objective c": "mobile",
    "kotlin": "mobile",
    "android": "mobile",
    "react native": "mobile",
    "flutter": "mobile",

    # qa / test automation
    "selenium": "qa_automation",
    "cypress": "qa_automation",
    "playwright": "qa_automation",
    "jest": "qa_automation",
    "mocha": "qa_automation",
    "junit": "qa_automation",
    "test automation": "qa_automation",
    "api testing": "qa_automation",
    "web testing": "qa_automation",
    "database testing": "qa_automation",

    # data science
    "pandas": "data_science",
    "numpy": "data_science",
    "scipy": "data_science",
    "matplotlib": "data_science",
    "seaborn": "data_science",
    "scikit learn": "data_science",
    "statistics": "data_science",
    "notebooks": "data_science",
    "feature engineering": "data_science",

    # ml engineering
    "machine learning": "ml_engineering",
    "pytorch": "ml_engineering",
    "tensorflow": "ml_engineering",
    "keras": "ml_engineering",
    "onnx": "ml_engineering",
    "mlflow": "ml_engineering",
    "kubeflow": "ml_engineering",
    "model serving": "ml_engineering",
    "inference": "ml_engineering",
    "nlp": "ml_engineering",
    "computer vision": "ml_engineering",

    # data engineering
    "spark": "data_engineering",
    "hadoop": "data_engineering",
    "airflow": "data_engineering",
    "kafka": "data_engineering",
    "dbt": "data_engineering",
    "etl": "data_engineering",
    "elt": "data_engineering",

    # devops
    "ci_cd": "devops",
    "github actions": "devops",
    "gitlab ci": "devops",
    "jenkins": "devops",
    "docker": "devops",
    "kubernetes": "devops",
    "helm": "devops",
    "terraform": "devops",
    "ansible": "devops",
    "openshift": "devops",

    # cloud
    "aws": "cloud",
    "ec2": "cloud",
    "s3": "cloud",
    "lambda": "cloud",
    "rds": "cloud",
    "gcp": "cloud",
    "bigquery": "cloud",
    "pubsub": "cloud",
    "azure": "cloud",
    "aks": "cloud",
    "cosmos db": "cloud",

    # databases
    "postgres": "databases",
    "mysql": "databases",
    "sql server": "databases",
    "sqlite": "databases",
    "mongodb": "databases",
    "redis": "databases",
    "cassandra": "databases",
    "snowflake": "databases",
    "elasticsearch": "databases",

    # security
    "oauth": "security",
    "jwt": "security",
    "oidc": "security",
    "sso": "security",
    "owasp": "security",
    "sast": "security",
    "dast": "security",
    "secrets management": "security",

    # observability
    "elastic stack": "observability",
    "elk": "observability",   # if you don’t normalize to “elastic stack”
    "logstash": "observability",
    "kibana": "observability",
    "grafana": "observability",
    "prometheus": "observability",
    "datadog": "observability",
    "dynatrace": "observability",

    # analytics / bi
    "sql": "analytics_bi",
    "tableau": "analytics_bi",
    "power bi": "analytics_bi",
    "looker": "analytics_bi",
    "mode": "analytics_bi",
}

COMPANY_AFFINITY = {
    ("kpmg canada", "microsoft"): 0.7,
    ("pwc canada", "aws"): 0.6,
    ("pwc canada", "formula 1"): 0.5,
    ("kpmg canada", "pwc canada"): 0.4,
}

SENIORITY_LEVEL = {
    "Student/Intern": 0,
    "Entry": 1,
    "Mid": 2,
    "Senior": 3,
    "Lead/Staff": 4,
    "Manager+": 5,
    "Founder": 6,
    "Other": 2,
}

'''Seniority gap (simple numeric score)'''
def seniority_gap_score(a, b) -> float:
    sa = SENIORITY_LEVEL.get(a.get("seniority"), 2)
    sb = SENIORITY_LEVEL.get(b.get("seniority"), 2)
    gap = abs(sa - sb)  # 0..maybe 6
    # we want: small gap -> positive, huge gap -> negative or small
    if gap == 0:
        return 1.0
    elif gap == 1:
        return 0.7
    elif gap == 2:
        return 0.4
    elif gap == 3:
        return 0.2
    else:
        return 0.1

"""lowercase, remove punctuation/variants (e.g., react.js -> react)"""
def normalize_skill(s):
    s = s.strip().lower()
    s = s.strip().lower()
    s = s.replace("&", " and ")
    s = s.replace("/", " ")
    s = s.replace("-", " ")
    s = s.replace(".", "")
    s = re.sub(r"\s+", " ", s)

    ALIASES = {
        "reactjs": "react",
        "nodejs": "node",
        "nextjs": "next",
        "vuejs": "vue",
        "ci cd": "ci_cd",
        "cicd": "ci_cd",
        "ml": "machine learning",
        "ai": "artificial intelligence",
        "elk": "elastic stack",
        "ms sql": "sql server",
        "postgresql": "postgres",
        "js": "javascript",
        "ts": "typescript",
        "tf": "tensorflow",
        "pt": "pytorch",
    }
    s = ALIASES.get(s, s)
    return s

def categorize_skills(skill: str) -> str:
    bucket = set()
    normalized = []

    for s in skill:
        ns = normalize_skill(s)
        normalized.append(ns)
        if ns in SKILL_TO_BUCKET:
            bucket.add(SKILL_TO_BUCKET[ns])
    return list(bucket), normalized

'''
skills = [
    "Selenium", "Test Automation", "API Testing", "Jest",
    "Kubernetes", "OpenShift", "Docker", "Elastic Stack (ELK)",
    "Python", "React.js", "Next.js", "SQL", "OAuth"
]

buckets, norm = categorize_skills(skills)
print("normalized:", norm)
print("buckets:", buckets)
'''
def enrich_person(p: dict) -> dict:
    p = {**p}
    p["id"] = str(p.get("id", p["name"]))
    p["company"] = (p.get("company") or "").strip()
    p["school"]  = (p.get("school")  or "").strip()
    p["skills"]  = p.get("skills", [])
    p["keywords"]= p.get("keywords", [])
    buckets, norm = categorize_skills(p["skills"])
    p["skills_norm"] = norm
    p["categories"]  = buckets
    return p

def rule_skill_sim(a: dict, b: dict) -> float:
    A, B = set(a["skills_norm"]), set(b["skills_norm"])
    shared = len(A & B)
    return min(shared / 10.0, 1.0)  # 0..1