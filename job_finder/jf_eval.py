import logging

from pathlib import Path
from job_finder.jf_nodes import SOURCES
from job_finder.core.jf_job_structure import Job

logger = logging.getLogger(__name__)


EVAL_PATH = Path(__file__).parent / "resources" / "eval_jobs.jsonl"
MAX_JOBS = 400

def is_unique(job: Job) -> bool:
    """Check if the job is unique in the eval set."""
    if not EVAL_PATH.exists():
        return True

    with open(EVAL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            existing_job = Job.model_validate_json(line)
            if existing_job.id == job.id:
                return False
            
    return True


def unlabelled_dataset_builder(logger, EVAL_PATH, MAX_JOBS, is_unique):
    EVAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    jobs: list[Job] = []
    
    for source in SOURCES:
        logger.info(f"Testing {source.__name__}...")
        try:
            fetched_jobs = source()
            unique_jobs = [job for job in fetched_jobs if is_unique(job)]
            jobs.extend(unique_jobs)
            
            logger.info(f"Fetched {len(fetched_jobs)} jobs, {len(unique_jobs)} unique.")
            
        except Exception as exc:
            logger.error(f"Source {source.__name__} failed: {exc}")
            
        logger.info(f"Fetched {len(jobs)} jobs")
        
    with open(EVAL_PATH, "a+", encoding="utf-8") as f:
        f.seek(0)
        count = sum(1 for line in f if line.strip())
        
        for job in jobs:
            if(count >= MAX_JOBS) : 
                logger.info(f"Reached {MAX_JOBS} lines in eval file, stopping write.")
                break
            
            f.write(job.model_dump_json() + "\n")
            count += 1
            
def label_jobs_for_evaluation():
    """Label jobs for evaluation purposes."""
    
    with open(EVAL_PATH, "r+", encoding="utf-8") as f:
        jobs = [Job.model_validate_json(line) for line in f if line.strip()]
        
        for job in jobs:
            
            print(f"Job ID: {job.id}")
            print(f"Title: {job.title}")
            print(f"Company: {job.company}")
            print(f"Salary: {job.salary_text}")
            print(f"Location: {job.location}")
            print(f"Description: {job.description}")
            
            print("Label this job for evaluation purposes:")
            input_label = input("Enter 'accepted' or 'skip': ").strip().lower()
            job.label = input_label
        
        # Move the file pointer to the beginning and truncate the file
        f.seek(0)
        f.truncate()
        
        for job in jobs:
            f.write(job.model_dump_json() + "\n")

            


if __name__ == "__main__":
    
    logging.basicConfig(level=logging.INFO)
    
    unlabelled_dataset_builder(logger, EVAL_PATH, MAX_JOBS, is_unique)
    # label_jobs_for_evaluation()
