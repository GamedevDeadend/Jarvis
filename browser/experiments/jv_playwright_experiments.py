from playwright.async_api import async_playwright, Playwright, Locator

import os
import asyncio
import re

from dotenv import load_dotenv

from langgraph.types import interrupt, Command

from langchain.messages import SystemMessage, HumanMessage

from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq

from pydantic import BaseModel, Field

MODEL_ID = "qwen2.5:3b"

load_dotenv()

browser_profile_path = os.getenv("BROWSER_PROFILE_PATH")
browser_executable_path = os.getenv("BROWSER_EXECUTABLE_PATH")


ref_map = {}

VALID_ROLES = {
    "alert",
    "alertdialog",
    "application",
    "article",
    "banner",
    "blockquote",
    "button",
    "caption",
    "cell",
    "checkbox",
    "code",
    "columnheader",
    "combobox",
    "complementary",
    "contentinfo",
    "definition",
    "deletion",
    "dialog",
    "directory",
    "document",
    "emphasis",
    "feed",
    "figure",
    "form",
    "generic",
    "grid",
    "gridcell",
    "group",
    "heading",
    "img",
    "insertion",
    "link",
    "list",
    "listbox",
    "listitem",
    "log",
    "main",
    "marquee",
    "math",
    "meter",
    "menu",
    "menubar",
    "menuitem",
    "menuitemcheckbox",
    "menuitemradio",
    "navigation",
    "none",
    "note",
    "option",
    "paragraph",
    "presentation",
    "progressbar",
    "radio",
    "radiogroup",
    "region",
    "row",
    "rowgroup",
    "rowheader",
    "scrollbar",
    "search",
    "searchbox",
    "separator",
    "slider",
    "spinbutton",
    "status",
    "strong",
    "subscript",
    "superscript",
    "switch",
    "tab",
    "table",
    "tablist",
    "tabpanel",
    "term",
    "textbox",
    "time",
    "timer",
    "toolbar",
    "tooltip",
    "tree",
    "treegrid",
    "treeitem",
}


def ref_map_builder(snapshot : str):
    """
    Function to build Reference Lookup Dict
    """

    lines = snapshot.splitlines()
    
    parent_ref = None
    stack = []
    

    for line in lines:

        # Search refs 
        ref_match = re.search(r"^(\s*).+\[ref=([^\]]+)\]", line)

        if not ref_match:
            continue
        
        indentation_len = len(ref_match.group(1))
        ref_id = ref_match.group(2)
        
        ref_map[ref_id] = {"role" : "", "name" : ""}

        # Pattern to lookup role of refs
        role_match_group = re.search(r"^\s*-\s*([a-z]+)", line)

        if not role_match_group:
            del ref_map[ref_id]
            continue

        role = role_match_group.group(1)

        if role not in VALID_ROLES:
            del ref_map[ref_id]
            continue

        # name / heading of refs
        name_match = re.search(r'"(.+)"', line)

        if name_match:
            name = name_match.group(1)
            
        name = ""

        
        while stack and ref_map[stack[-1]].get("indentation_len") >= indentation_len:
            stack.pop()
            
        parent_ref = stack[-1] if stack else None
        
        stack.append(ref_id)
        
        ref_map[ref_id] = {"role" : role, "name" : name, "indentation_len" : indentation_len, "parent_ref" : parent_ref}
        
       

class BestSS(BaseModel):
    index : int = Field (description="Give Index of best observation from all the observation")
    reason : str = Field (description="Reason to select this observation")



async def test_run(playwright : Playwright):
    chromium = playwright.chromium
    browser = await chromium.launch_persistent_context(user_data_dir=browser_profile_path, executable_path=browser_executable_path, headless=False, args=["--profile-directory=Profile 2"] )
    page = await browser.new_page()
    await page.goto("https://www.youtube.com/")
    await page.wait_for_load_state("load")

    snapshot = await page.locator("body").aria_snapshot(mode="ai")
    ref = input("Enter Ref Id")
    locator = page.locator(f"aria-ref={ref}")
    await locator.click(timeout=10000)
    
    full_content = await page.content()

    print(f"Content : {full_content[:1000]}\n\n")

    snapshot = re.sub( r"\s*\[cursor=[^\]]*\]", "",snapshot) #Removing [cursor=...]
    snapshot = re.sub(r"(?m)^[\t\s]*-[\s]*\/url.*\n?\r?", "", snapshot) #Removing [urls]
    snapshot = re.sub(r"(?m)^[\t\s]*-\s*(generic|listitem).+\n?\r?", "", snapshot) #Removing [generic, listitem]

    ref_map_builder(snapshot)

    lenght_ss = len(snapshot)

    print(f"Snapshot length : {lenght_ss}\n\n")
    print(f"Snapshot : \n\n{snapshot[:1000]}\n\n")

    divided_snapshots = {}

    rem = lenght_ss%10000
    parts = lenght_ss//10000

    if rem > 0 :
        parts  = parts+1


    SYSTEM_PROMPT = """You are Observer Agent. You are part of browser automation system which has tools like click, type, goback, navigate.
    On each step you are provided with series of snapshot. As agent you have to observe one snapshot at a time. And Give best candidate ref id among it which can help reach our goal.
    For example : 
    Best_Candidate : Button with ref e10 because this is product page.
    Best_Candidate : Searchbox with ref e20 because this can be used to search product.

    Anything along those lines.

    NOTE : As There will be multiple snapshot you can also give your observation as PASS.

    Recent Snapshot : 

    {snapshot}
    """

    observation = ""

    for i in range(parts):
        startIndex = i*10000
        endIndex = startIndex+10000

        if parts-1 == i:
            endIndex = rem

        result = await make_observation(snapshot, SYSTEM_PROMPT, startIndex, endIndex)
        divided_snapshots[i+1] = snapshot[startIndex : endIndex]
        observation = observation + (f"{i+1} : {result.content}\n\n")


    Check_Prompt = "These are all the observations of various web snapshot. You have to decide which observation is best direction to achieve Goal. Keyword PASS means in that observation there was nothing relevant to goal\n\n All observations : {obs}"


    model = ChatOllama(model=MODEL_ID, temperature=0,  num_predict=1024)
    model_with_ss = model.with_structured_output(BestSS)

    messages = [SystemMessage(content=Check_Prompt.format(obs=observation)),
                HumanMessage(content = "Goal : Order Zoro poster on amazon india")]

    result = await model_with_ss.ainvoke(messages)
    print(f"Final Obs Index {result.index}\n Reason : {result.reason}\n\n")

    best_ss = divided_snapshots[result.index]

    print(f" SS : {best_ss:5000}\n\n")
    
    locators = page.get_by_role(role="button", name="More actions")
    
    try :
        
        await locators.click(timeout=10000)
    except Exception as e:
        
        print((f"Error {e}\n\n"))


    input("Press Enter to close....")
    await browser.close()

async def make_observation(snapshot, SYSTEM_PROMPT, i:int, j:int):

    model = ChatOllama(model=MODEL_ID, temperature=0,  num_predict=1024)

    messages = [SystemMessage(content=SYSTEM_PROMPT.format(snapshot=snapshot[i:j])),
                HumanMessage(content = "Goal : Order Zoro poster on amazon india")]

    result = await model.ainvoke(messages)

    print(f"Observation : {result.content}\n\n")
    return result

async def main():
    async with async_playwright() as playwright:
        await test_run(playwright)


asyncio.run(main())