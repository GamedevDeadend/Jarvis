"""
jv_jf_try_discord.py

Throwaway test: fetch from RemoteOK, run rules(), then post a few
survivors to #primary/#secondary and a few rejects to #rejected.

Does NOT use jobs.db, so nothing is saved and nothing is deduplicated.
Survivors have NOT been judged yet, so expect some junk in the alerts.

Run from the repo root:
    python -m job_finder.jv_jf_try_discord        (3 of each)
    python -m job_finder.jv_jf_try_discord 5      (5 of each)
"""

import logging
import sys
import time

from job_finder import jv_jf_tools as tools

logging.basicConfig(level=logging.INFO)

limit = int(sys.argv[1]) if len(sys.argv) > 1 else 3

jobs = tools.fetch_remoteok()
survivors, rejected = tools.rules(jobs)
print(f"fetched {len(jobs)} | passed rules {len(survivors)} | rejected {len(rejected)}")

print(f"\nposting {min(limit, len(survivors))} survivors...")
for job in survivors[:limit]:
    print("  alert sent:", tools.notify_job(job), "|", job["title"])
    time.sleep(0.5)   # stay under Discord's webhook rate limit

print(f"\nposting {min(limit, len(rejected))} rejects...")
for job in rejected[:limit]:
    print("  reject sent:", tools.notify_rejected(job), "|", job["title"], "|", job["reason"])
    time.sleep(0.5)

tools.notify_heartbeat(
    f"rules() test: {len(jobs)} fetched, {len(survivors)} passed, {len(rejected)} rejected"
)
print("\nheartbeat sent")