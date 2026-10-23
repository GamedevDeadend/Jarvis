import logging

from langgraph.graph import END, START, StateGraph

from job_finder.core.jf_database import init_db
from job_finder.jf_nodes import (
    dedup_node,
    ensure_profile_node,
    fetch_jobs_node,
    gate_node,
    heartbeat_node,
    judge_node,
    keyword_filter_node,
    persist_alert_node,
    route_after_keyword_filter,
    route_after_profile,
)
from job_finder.jf_state import AgentState

logger = logging.getLogger(__name__)


def build_graph():
    g = StateGraph(AgentState)

    g.add_node("ensure_profile", ensure_profile_node)
    g.add_node("fetch_jobs", fetch_jobs_node)
    g.add_node("dedup", dedup_node)
    g.add_node("keyword_filter", keyword_filter_node)
    g.add_node("judge", judge_node)
    g.add_node("gate", gate_node)
    g.add_node("persist_alert", persist_alert_node)
    g.add_node("heartbeat", heartbeat_node)

    g.add_edge(START, "ensure_profile")
    g.add_conditional_edges(
        "ensure_profile", route_after_profile,
        {"fetch_jobs": "fetch_jobs", END: END},
    )
    g.add_edge("fetch_jobs", "dedup")
    g.add_edge("dedup", "keyword_filter")
    g.add_conditional_edges(
        "keyword_filter", route_after_keyword_filter,
        {"judge": "judge", "persist_alert": "persist_alert"},
    )
    g.add_edge("judge", "gate")
    g.add_edge("gate", "persist_alert")
    g.add_edge("persist_alert", "heartbeat")
    g.add_edge("heartbeat", END)

    return g.compile()


def run_cycle(dry_run: bool = False) -> dict:
    """One full cycle. Plain function so a future orchestrator can call it
    as a worker. Returns the final state."""
    init_db()
    graph = build_graph()
    return graph.invoke({"rejected": [], "stats": {}, "dry_run": dry_run})


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dry", action="store_true",
                        help="print verdicts only: no DB writes, no Discord posts")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    run_cycle(dry_run=args.dry)