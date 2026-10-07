import os
from datetime import date
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool          # ⭐ new import

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini/gemini-3.5-flash-lite")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env")

llm = LLM(model=MODEL_NAME, api_key=API_KEY, temperature=0.2)


# ---------- TOOLS ----------

@tool("Get Today's Date")
def get_today_date() -> str:
    """Returns today's date. Use this whenever you need the current date."""
    return date.today().isoformat()


@tool("Save To File")
def save_to_file(filename: str, content: str) -> str:
    """Saves text content to a file in the current folder.
    filename: name of the file, e.g. notes.md
    content: the full text to write into the file"""
    with open(filename, "w") as f:
        f.write(content)
    return f"Saved {len(content)} characters to {filename}"


# ---------- AGENT ----------

note_writer = Agent(
    role="Study Notes Writer",
    goal="Write short, clear study notes and save them to a file.",
    backstory="You write crisp revision notes for students.",
    tools=[get_today_date, save_to_file],    # ⭐ give the agent its tools
    llm=llm,
    verbose=True
)

# ---------- TASK ----------

notes_task = Task(
    description=(
        "Write short study notes on: {topic}\n\n"
        "1. Get today's date using your tool and put it at the top.\n"
        "2. Write the notes in Markdown (max 200 words).\n"
        "3. Save them using your Save To File tool as notes.md."
    ),
    expected_output="Confirmation that notes.md was saved, plus the notes.",
    agent=note_writer
)

crew = Crew(agents=[note_writer], tasks=[notes_task], verbose=True)

topic = input("\nTopic for study notes? ")
result = crew.kickoff(inputs={"topic": topic})
print("\n===== RESULT =====\n", result)