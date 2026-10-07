import os
import time
from dotenv import load_dotenv
from crewai import Agent, Task,Crew, Process,LLM
from crewai.tools import tool
from ddgs import DDGS


load_dotenv()
API_KEY=os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini/gemini-3.5-flash-lite")

if not API_KEY:
    raise ValueError("\nNO API KEY FOUND\n")

llm=LLM(model=MODEL_NAME,
        api_key=API_KEY,
        temperature=0.1)

#THE TOOL

@tool("Web Search")
def web_search(query:str)->str:
    """Searches the web and returns the top results with title, link and snippet.
    Use this to find current facts, recent news or anything you are unsure about.
    query: a short, specific search query, e.g. 'UPI transactions 2026 statistics'"""

    try:
        results=DDGS().text(query,max_result=5)
    except Exception as e:
        return f"Search failed:{e}"
    if not results:
        return "no result found"
    
    output=[]

    for r in results:
        output.append(f"Title: {r['title']}\nLink: {r['href']}\nSnippet: {r['body']}\n")
    return "\n".join(output)


#Agent for researching 
research_agent=Agent(
    role="Research Analyst",
    goal=(
        "Research the topic and produce a structured research brief "
        "that another agent can use to write a report."
    ),
    backstory=(
        "You are an experienced research analyst. You find the key concepts, "
        "organise them logically and cut unnecessary detail. "
        "You do NOT write the final report."
    ),
tools=[web_search], #only for this agent
llm=llm,
verbose=True,
allow_delegation=False
)
#Agent for Writing a technical report
writer = Agent(
    role="Technical Report Writer",
    goal="Turn research findings into a clear, well-structured Markdown report.",
    backstory=(
        "You are a technical writer who converts research into reports "
        "students can easily understand. You do NOT do new research."
    ),
    llm=llm,
    verbose=True,
    allow_delegation=False
)

#TASK:1
research_task=Task(
name="research_task",
    description=(
        "Research the topic: {topic}\n\n"
        "Prepare a research brief covering:\n"
        "1. What it is\n"
        "2. Key concepts and terminology\n"
        "3. Why it matters\n"
        "4. How it works\n"
        "5. Examples or applications\n"
        "6. Key takeaways\n"
        "7.A list of source links you used\n\n"
        "Do NOT write the final report."
    ),
    expected_output="A structured research brief usable by a report writer.",
    agent=research_agent
)

#TASK:2
report_task = Task(
    name="report_task",
    description=(
        "Using ONLY the Research Analyst's findings, write a report on {topic} "
        "for students, with these sections:\n"
        "# {topic}\n"
        "## What is it?\n"
        "## Why does it matter?\n"
        "## How does it work?\n"
        "## Practical Examples\n"
        "## Key Takeaways"
    ),
    expected_output="A polished Markdown report with all the sections above.",
    agent=writer,
    context=[research_task],        # ⭐ receives Task 1's output
    output_file="final_report.md",  # ⭐ saves the result to a file
    markdown=True
)

# 6. The Crew
crew = Crew(
    agents=[research_agent, writer],
    tasks=[research_task, report_task],
    process=Process.sequential,     # ⭐ run the tasks in order
    verbose=True
)

# 7. Run it
topic = input("\nWhat should the two-agent crew investigate? ")
start = time.time()
result = crew.kickoff(inputs={"topic": topic})
end = time.time()

# 8. Results
print("\n" + "=" * 60)
print("TASK 1: RESEARCH BRIEF")
print("=" * 60)
print(result.tasks_output[0])

print("\n" + "=" * 60)
print("TASK 2: FINAL REPORT")
print("=" * 60)
print(result.tasks_output[1])

print(f"\n⏱️ {end - start:.2f}s | 🤖 {MODEL_NAME}")
print("📄 Saved as final_report.md")
print(f"🔢 Token usage: {result.token_usage}")
