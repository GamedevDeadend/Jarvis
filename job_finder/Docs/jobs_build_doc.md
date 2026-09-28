# Jobs Agent: Build Doc

Working document for the Job Fetcher agent inside the Jarvis repo (`jobs/`). This is a build plan, not the README. The README gets written once something is built.

Last updated: September 28, 2026

---

## 1. Goal

An agent that runs on a schedule, finds remote job openings matching my resume, filters them by salary and fit, and sends a Discord alert with the career link so I can apply early.

Hard requirements:
- Remote: worldwide or India
- Pay: at least 2 lakh INR per month (about $2K)
- Role matches resume (agentic AI, LangChain/LangGraph/MCP/RAG, Python/FastAPI, junior-to-mid)
- Link to the career/job page
- Small startups (0-50 people) preferred, not required

---

## 2. Locked decisions

| Area | Decision |
|---|---|
| Salary, stated | Hard filter at the 2L/month floor |
| Salary, unstated | LLM + search estimate, tagged `[est]`, dropped if the estimate is below the floor |
| Career link | Whatever link the source gives, as-is |
| Company size | Soft ranking signal, looked up by search only for candidates that survive the other filters |
| Career page type | Self-hosted career page = primary. ATS-hosted (Greenhouse/Lever/Ashby/Workable) = secondary. Both still alerted |
| LLM | Hosted (Groq, `openai/gpt-oss-20b`). Local 3B is not reliable enough for judgment |
| State | SQLite, one `jobs.db` file, behind a small storage module so it can be swapped later |
| Alerts | Discord webhooks: `#primary`, `#secondary`, `#heartbeat` |
| Location in repo | `jobs/` folder in the Jarvis repo, `jv_` file prefix |
| Deployment | Deferred, decide later |

Out of v1: Crunchbase (paid), LinkedIn and Naukri (scraping risk), paid search APIs.

---

## 3. Pipeline

There are two discovery phases that feed one shared filter-and-alert path.

**Phase A: Job aggregator discovery (fast cadence, every 2-4 hours)**
Pull listings from job boards that offer feeds or APIs.

**Phase B: Company discovery (slow cadence, daily or every few days)**
1. Filter companies from directories (YC directory, Wellfound company search, Product Hunt) by tags, recency and description.
2. LLM judges each company's description against my profile (no fetching yet).
3. For survivors only: resolve the career page, then fetch and parse it for live openings.
   - Search for the company's careers page and detect whether it is on a known ATS.
   - ATS found: use the platform's structured job feed (Greenhouse and Lever have public JSON).
   - No ATS: fetch the company's own careers page and let the LLM read it.
   - Nothing resolves: skip, do not over-invest per company.
4. Any live opening found joins the shared path below.

**Shared path (both phases)**
1. Dedup against SQLite (skip anything already seen).
2. Hard filters: remote/location, obvious mismatches from my JD filter rules (4+ years required, visa sponsorship, onsite/hybrid, EU-only, and so on).
3. LLM role-fit judgment against the resume.
4. Salary: stated goes through the hard filter, unstated goes through estimate and `[est]` rule.
5. Company size lookup for survivors (soft boost).
6. Assign priority: self-hosted career page = primary, ATS = secondary.
7. Send the Discord alert, record the job in SQLite.

---

## 4. Sources

**Job aggregators:** RemoteOK, Remotive, Wellfound, WorkingNomads, We Work Remotely, Himalayas, YC Work at a Startup.

**Company discovery:** YC company directory, Wellfound company search, Product Hunt.

**Enrichment:** web search tool, used sparingly for salary estimates, employee count and career-page resolution.

Optional, not decided: GitHub search for small AI-stack companies.

---

## 5. Data model (SQLite, first draft)

Table `jobs`:
- `id` (primary key, hash of the canonical URL)
- `url`
- `company`
- `title`
- `source`
- `first_seen` (timestamp)
- `salary_text` (raw, if stated)
- `salary_est_inr` (if estimated) and `salary_is_est` (bool)
- `company_size` (nullable)
- `priority` (`primary` or `secondary`)
- `fit_score` and `fit_reason` (LLM output)
- `alerted` (bool) and `alerted_at`

Table `companies` (for Phase B):
- `id`, `name`, `domain`, `career_url`, `ats_type` (nullable), `last_checked`

---

## 6. Planned files (`jobs/`)

- `jv_jobs_store.py`: SQLite access, dedup checks
- `jv_jobs_notify.py`: Discord webhook sender (embed formatting, channel routing)
- `jv_jobs_sources.py`: one fetch function per job board, all returning a common job shape
- `jv_jobs_match.py`: LLM fit judgment, salary estimate, size lookup
- `jv_jobs_companies.py`: Phase B company discovery and career-page resolution
- `jv_jobs_main.py`: `run_cycle()` and the scheduler entry point
- `resources/jv_resume_profile.md`: resume context used for matching
- `.env`: Groq key and Discord webhook URLs (never committed)

`run_cycle()` stays a plain function so the future Orchestrator can call it as a worker.

---

## 7. Build order

Each step ends with something that visibly works.

1. **Store + notify.** SQLite store and Discord webhook sender. Done when a hardcoded fake job gets stored, alerted once, and not alerted again on a second run.
2. **One source.** Fetch and parse RemoteOK into the common job shape. Done when real listings land in SQLite.
3. **Matching.** LLM fit judgment plus the salary rule (stated filter, `[est]` estimate). Done when real listings get accepted or rejected with a readable reason, checked by hand against my own judgment.
4. **First real alerts.** Wire steps 1-3 end to end with priority tags and channel routing. Run for a few days on the laptop and calibrate (too loose, too strict, wrong drops).
5. **More sources.** Remotive, WorkingNomads, We Work Remotely, Himalayas, Wellfound, YC.
6. **Company size lookup** and ranking boost.
7. **Phase B.** Company discovery, career-page resolution, ATS detection.
8. **Scheduling, heartbeat, low-battery warning.**
9. **Deployment** (phone or other host, decide then).
10. **Langfuse tracing** on all LLM calls. Could move earlier, since it is cheap and already familiar.

---

## 8. Open items

- Whether to include GitHub search as a source
- Exact Phase A and Phase B cadence
- How the LLM gets the resume (static profile file vs embedding); a static profile file is enough for v1
- Whether to add a small eval set (jobs I would and would not apply to) to measure match quality; strong candidate, given the eval gap on my resume
- Deployment target and its constraints (deferred)
- Redmi battery replacement before it hosts anything (safety, not optional)

---

## 9. Known risks

- Most listings do not state salary, so estimates carry real error. The `[est]` tag and the drop rule are the guardrails; recheck after the first week.
- The agent is only as early as its sources. Aggregators lag the original post.
- JS-rendered career pages may return an empty shell to a plain fetch. Those get skipped in v1.
- Source APIs and feeds change without warning. Each source lives in its own function so one breaking does not stop the cycle.
- Coverage of small remote startups is patchy by nature. Expect to add sources over time.
