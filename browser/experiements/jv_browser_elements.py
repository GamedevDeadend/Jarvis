"""
jv_browser_elements.py  (v2 — no page.accessibility, it was removed from Playwright)

THROWAWAY EXPERIMENT SCRIPT — Step 2 of the browser layer build.

Goal: prove we can turn a live webpage into a numbered, LLM-readable list
of interactive elements, and reliably re-locate any one of them afterward.

v2 change: instead of walking a full accessibility tree (page.accessibility
was removed in recent Playwright versions), we directly query each role we
care about using page.get_by_role(role). Simpler, and it uses the exact
same method we'll use later to re-locate/click elements — no mismatch
between "how we found it" and "how we click it."

Nothing about agents, tools, or LangGraph belongs here. Read-only,
extraction only — no clicking, no typing, no LLM calls.

Run:
    python jv_browser_elements_v2.py
"""

from playwright.sync_api import sync_playwright

# Roles we consider "interactive" for now. Keep this list short and add to
# it only when a real page shows you're missing something.
INTERACTIVE_ROLES = [
    "link",
    "button",
    "textbox",
    "searchbox",
    "checkbox",
    "radio",
    "combobox",
    "menuitem",
    "tab",
]

TEST_URL = "https://news.ycombinator.com"

# Safety cap so one noisy page doesn't dump 500 elements onto an LLM later.
MAX_ELEMENTS = 60


def build_element_list(page):
    """
    Query the page directly, role by role, using get_by_role — instead of
    walking a tree. Returns a list of dicts: {"id": int, "role": str, "name": str}
    """
    collected = []

    for role in INTERACTIVE_ROLES:
        locator = page.get_by_role(role)
        count = locator.count()

        for i in range(count):
            el = locator.nth(i)
            try:
                # accessible name: prefer visible text, fall back to
                # aria-label/placeholder for things like empty icon buttons
                # or inputs with no visible text.
                name = el.inner_text(timeout=500).strip()
            except Exception:
                name = ""

            if not name:
                try:
                    name = (el.get_attribute("aria-label") or "").strip()
                except Exception:
                    name = ""

            if not name:
                try:
                    name = (el.get_attribute("placeholder") or "").strip()
                except Exception:
                    name = ""

            if name:
                collected.append({"role": role, "name": name})

    # De-duplicate identical (role, name) pairs — pages often repeat labels
    # (e.g. multiple "more" links) which is fine to collapse for this test.
    seen = set()
    deduped = []
    for el in collected:
        key = (el["role"], el["name"])
        if key not in seen:
            seen.add(key)
            deduped.append(el)

    deduped = deduped[:MAX_ELEMENTS]

    # Assign stable numeric IDs.
    numbered = []
    for i, el in enumerate(deduped, start=1):
        numbered.append({"id": i, "role": el["role"], "name": el["name"]})

    return numbered


def print_element_list(elements):
    print(f"\nFound {len(elements)} interactive elements:\n")
    for el in elements:
        label = el["name"] if len(el["name"]) <= 70 else el["name"][:67] + "..."
        print(f"  [{el['id']}] {el['role']}: \"{label}\"")
    print()


def try_relocate(page, element):
    """
    Test whether we can reliably re-find this exact element later using
    page.get_by_role(role, name=...) — same method used to build the list,
    so this is really testing "is the name unique enough," not the API.
    """
    role = element["role"]
    name = element["name"]

    locator = page.get_by_role(role, name=name, exact=True)
    count = locator.count()

    print(f"Re-locating [{element['id']}] {role}: \"{name[:60]}\"")
    print(f"  -> get_by_role match count: {count}")

    if count == 0:
        print("  -> FAILED to relocate (exact match not found)")
    elif count == 1:
        print("  -> OK, unique match")
    else:
        print(f"  -> WARNING: {count} elements share this role+name (ambiguous)")

    return count


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()

        print(f"Navigating to {TEST_URL} ...")
        page.goto(TEST_URL)
        page.wait_for_load_state("networkidle")

        elements = build_element_list(page)
        print_element_list(elements)

        if elements:
            test_index = len(elements) // 2
            test_element = elements[test_index]
            try_relocate(page, test_element)

        input("\nPress Enter to close the browser...")
        browser.close()


if __name__ == "__main__":
    main()