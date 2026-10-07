#####
# This code is the first basic agent using CrewAI.
# Created to demonstrate the basic functionality of what an AI Agent can do.
# This uses CrewAI as we at PESU CIE - Agentic AI workshop are using this framework to start with
# The instructions to run this in your local environment is below.
# This code assumes that you are using LMStudio with Qwen 3.5 9b.
# If you use any other model, say Bonsai 8B or Mistral or Gemma4 E2B etc, add that in the env file.
# Students are expected to run this as-is OR do their own tweaks and get it to work on their systems.
#####
#python -m venv .venv
#source .venv/bin/activate
#python -m pip install -U pip
#pip install crewai

import os
import time
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM


load_dotenv()
API_KEY=os.getenv("GEMINI_API_KEY")
MODEL_NAME=os.getenv("GEMINI_MODEL","gemini/gemini-3.5-flash-lite")

if not API_KEY:
    raise ValueError ("GEMINI_API_KEY not found in .env")

print(f"CURRENT MODEL:{MODEL_NAME}")
llm=LLM(
    model=MODEL_NAME,
    api_key=API_KEY,
    temperature=0.7 #lets go a bit creative we will update this after checking the output   
)
research_agent=Agent(
    role="Research Analyst",
    goal=(
        "Research a given topic and give me simple answers"
        "Also give recomendation to dig deep into that topic"
    ),
    backstory=(
        "You are a careful research analyst who specializes "
        "in explaining technology and business topics. "
        "You simplify complex ideas without losing technical accuracy."
    ),
    llm=llm,
    verbose=True
)

research_task=Task(
    description=(
        "Research the topic: {topic}\n\n" # the topic gets fetched from crew.kickoff
        "Provide a beginner-friendly explanation.\n\n"
        "Cover:\n"
        "1. What the topic is\n"
        "2. Why it is important\n"
        "3. One practical example\n"
        "4.Further related topics\n"
        "Keep the explanation concise."
    ),
    expected_output=(
         "A concise research brief containing:\n"
        "- What is it?\n"
        "- Why does it matter?\n"
        "- Practical Example\n"
        "- Related topics"
    ),
    agent=research_agent
)

crew=Crew(
    agents=[research_agent],
    tasks=[research_task],
    verbose=True
)
topic=input("\nWhat would you like the research agent to investigate? ")

#record the starting time
start_time=time.time()
print("\n the agent has began it's work!")

#stores the result with topic given as input
result = crew.kickoff(inputs={"topic": topic})

end_time = time.time()
print("\n" + "=" * 60)
print("RESEARCH RESULT")
print("=" * 60)
print(result)
print(f"\n⏱️ Time taken: {end_time - start_time:.2f} seconds with {MODEL_NAME}")