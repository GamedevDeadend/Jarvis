# Multi-Agent Orchestration Workflow(Old version)(Ongoing Project : Currently Browser Agent is almost done)

A flowchart describing how an agentic system routes, plans, executes, and validates tasks using multiple specialized agents.

---

## Flow Overview

```
Start
  │
  ▼
Route the query (Small or large LLM?)
  │
  ▼
Planner agent — picks agents, builds plan
  │
  ▼
◇ Plan solid? (Critique agent)
  │ No ──────────────────────────────► back to Planner agent
  │ Yes
  ▼
┌─────────────────────────────────────────────┐
│  Dispatch to agents (in parallel)            │
│                                             │
│  Agent 1          Agent 2       Agent 3     │
│  Browser agent    File ops      Knowledge   │
│  (web search)     (files &      agent       │
│                   storage)      (agentic    │
│                                  RAG)       │
└─────────────────────────────────────────────┘
  │
  ▼
Result evaluation
  │
  ▼
◇ Meets requirement?  ◄──── Reflector (critique pass)
  │ No ──────────────────────────────► Replan (back to Planner agent)
  │ Yes
  ▼
End
```

---

## Stages

### 1. Start
Entry point of the workflow.

---

### 2. Route the query
**Decision:** Small or large LLM?

Determines which model tier should handle the query — a smaller, faster model (e.g. 3/7B) or a larger, more capable one (e.g. 70B+) — based on task complexity.

---

### 3. Planner agent
Picks the agents required for the task and builds a structured plan of action.

---

### 4. Plan validation (Critique agent)
**Decision:** Is the plan solid?

- **Yes** → proceed to agent dispatch
- **No** → return to the Planner agent to revise

---

### 5. Agent dispatch

Tasks are routed to one or more specialized agents in parallel:

| Agent | Role |
|---|---|
| **Browser agent** (Agent 1) | Web search and browsing |
| **File operations agent** (Agent 2) | File reads, writes, and storage |
| **Knowledge agent** (Agent 3) | Retrieval-augmented generation (agentic RAG) |

---

### 6. Result evaluation
Aggregates and assesses the outputs returned from all dispatched agents.

---

### 7. Reflector (Critique agent)
Reviews the evaluated result and flags gaps or quality issues before the final check.

---

### 8. Meets requirement?
**Decision:** Does the result satisfy the original requirement?

- **Yes** → End
- **No** → Replan (loop back to the Planner agent)

---

### 9. End
Workflow terminates successfully once the result meets the initial requirement.

---

## Key Design Principles

- **Critique at two points** — once before execution (plan validation) and once after (reflector), catching errors early and late.
- **Parallel agent dispatch** — Browser, File operations, and Knowledge agents can run concurrently for efficiency.
- **Self-correcting loop** — a failed requirement check feeds back into the planner, not just the agents, so the whole plan can be revised rather than just re-executed.
