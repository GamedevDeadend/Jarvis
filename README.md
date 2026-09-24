# Jarvis — Multi-Agent Personal AI Assistant

A personal AI assistant built as a set of independently working agents, designed
to eventually run under a single orchestrator. This repo tracks both what's
actually built and tested, and what's still on the drawing board — see status
below.

## Status

| Agent | Status | Details |
|---|---|---|
| Browser Automation | 🟡 Partially working, actively debugging | [browser/README.md](browser/README.md) |
| Voice (STT/TTS) | 🟡 Built, not yet wired to orchestrator | [voice/README.md](voice/README.md) |
| Orchestrator | ⚪ Designed, not yet implemented | see Planned Architecture below |
| File Operations Agent | ⚪ Not started | — |
| Knowledge Agent (RAG) | ⚪ Not started | — |

---

## Planned Architecture

A flowchart describing how the agentic system will route, plan, execute, and
validate tasks using multiple specialized agents, once the orchestrator is built.

## Flow Overview

```text
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
│ Dispatch to agents (in parallel) │
│ │
│ Agent 1 Agent 2 Agent 3 │
│ Browser agent File ops Knowledge │
│ (web search) (files & agent │
│ storage) (agentic │
│ RAG) │
└─────────────────────────────────────────────┘
│
▼
Result evaluation
│
▼
◇ Meets requirement? ◄──── Reflector (critique pass)
│ No ──────────────────────────────► Replan (back to Planner agent)
│ Yes
▼
End
```