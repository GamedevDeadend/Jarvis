from pathlib import Path

BASE = Path(__file__).parent.parent
RES_DIR = BASE / "resources"
PROFILE_MD = RES_DIR / "jv_resume_profile.md"
RESUME_PDF = RES_DIR / "resume.pdf"
RULES_JSON = RES_DIR / "jv_rules.json"
HASH_FILE = RES_DIR / ".resume_hash"
REQUIREMENTS_TXT = RES_DIR / "additional_requirements.txt"
DB_PATH = BASE / "jobs.db"