import os
from typing import Any, Tuple
import streamlit as st
from crewai import Agent, Crew, LLM, Process, Task, TaskOutput
from crewai.knowledge.source.text_file_knowledge_source import (
    TextFileKnowledgeSource,
)
from crewai_tools import SerperDevTool
from dotenv import load_dotenv

# Load environment variables strictly from the .env file
load_dotenv()

# Verify environment variables exist behind the scenes (no UI prompts)
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables or .env file.")
if not os.getenv("SERPER_API_KEY"):
    raise ValueError("SERPER_API_KEY not found in environment variables or .env file.")

# --- CUSTOM DATABASE STORAGE CONFIGURATION ---
CUSTOM_DB_DIR = os.path.abspath("./my_custom_db")
os.environ["CREWAI_STORAGE_DIR"] = CUSTOM_DB_DIR

# Page configuration
st.set_page_config(
    page_title="Hyundai Customer Support Multi-Agent System",
    page_icon="🚗",
    layout="wide",
)

st.title("Hyundai Customer Support Multi-Agent System (RAG + Guardrails)")
st.markdown(
    "This application processes customer queries sequentially using three specialized agents powered by **gpt-4o-mini**, "
    "backed by **Hyundai Official Policy Knowledge Base RAG**, **Custom DB Memory Storage**, and **Automated Task Guardrails**."
)

# --- 1. KNOWLEDGE DIRECTORY & TEXT FILE SETUP ---
KNOWLEDGE_DIR = "knowledge"
os.makedirs(KNOWLEDGE_DIR, exist_ok=True)

POLICY_FILENAME = "hyundai_policy.txt"
POLICY_PATH = os.path.join(KNOWLEDGE_DIR, POLICY_FILENAME)

# Automatically create the default Hyundai Policy text file if it doesn't exist
if not os.path.exists(POLICY_PATH):
    default_policy_content = """[HYUNDAI MOTOR INDIA - CUSTOMER SUPPORT & SERVICE POLICY v3.1]
1. Warranty Policy: Comprehensive vehicle warranty covers mechanical and electrical components as per specific model terms. Warranty claims require periodic maintenance records at authorized service centers.
2. Roadside Assistance (RSA): 24x7 complimentary RSA is provided during the basic warranty period (Toll-free: 1800-102-4645). Covers towing, flat tire replacement, dead battery jump-start, key lock-out, and fuel delivery up to 5 litres.
3. Customer Care & Contact: Main customer support helpline is 1800-114-645. WhatsApp support is available for quick assistance and service registration.
4. Bluelink & Connected Car Support: Dedicated support is available for connected car features, remote diagnostics, and infotainment system troubleshooting via the Bluelink helpdesk.
5. Periodic Maintenance Service: Recommended every 10,000 km or 1 year, whichever comes first, to maintain warranty validity and optimal vehicle performance.
"""
    with open(POLICY_PATH, "w", encoding="utf-8") as f:
        f.write(default_policy_content)

# Initialize the Text File Knowledge Source using 'file_paths' list
rag_knowledge_source = TextFileKnowledgeSource(
    file_paths=[POLICY_FILENAME],
    metadata={"source": "hyundai_policy_kb", "category": "customer_support"},
)

# User input text box
user_query = st.text_input(
    "Enter your Hyundai customer support query or task:",
    placeholder="e.g., What is covered under Hyundai Roadside Assistance and what is the toll-free number?",
)

if st.button("Run Hyundai Support Crew", type="primary"):
    if not user_query.strip():
        st.error("Please enter a valid query before running the crew.")
    elif not os.path.exists(POLICY_PATH):
        st.error(f"⚠️ Policy file not found at path: {POLICY_PATH}")
    else:
        with st.spinner(
            "Processing query via Hyundai Policy RAG, Web Search & Guardrails..."
        ):
            try:
                # 1. Initialize Tools & Explicit LLM
                search_tool = SerperDevTool()
                support_llm = LLM(model="openai/gpt-4o-mini", temperature=0.2)

                # 2. Define the Three Agents using gpt-4o-mini
                assistant_agent = Agent(
                    role="Hyundai Assistant",
                    goal="Answer the user's query accurately using the official Hyundai policy knowledge base.",
                    backstory=(
                        "You are an expert Hyundai customer support representative equipped with "
                        "direct access to official Hyundai policy documents via high-speed vector retrieval. "
                        "You provide precise, grounded answers based strictly on Hyundai guidelines."
                    ),
                    llm=support_llm,
                    verbose=True,
                )

                web_search_agent = Agent(
                    role="Web Search Assistant",
                    goal="Search the web to gather complementary external information or verify industry/Hyundai standards related to the query.",
                    backstory=(
                        "You are a skilled automotive researcher who browses the web to pull up "
                        "real-time context or additional details that enrich the Hyundai customer support response."
                    ),
                    tools=[search_tool],
                    llm=support_llm,
                    verbose=True,
                )

                entry_agent = Agent(
                    role="Entry Agent",
                    goal=(
                        "Compile the original user query, Answer 1 (from Hyundai RAG Assistant), "
                        "and Answer 2 (from Web Search Assistant), append them into a local text file "
                        "named support_log.txt, and present both answers clearly."
                    ),
                    backstory=(
                        "You are a meticulous data logging and synthesis specialist. "
                        "Your job is to record all customer interactions into a persistent log file "
                        "and structure the final report for the user interface."
                    ),
                    llm=support_llm,
                    verbose=True,
                )

                # 3. Define Tasks corresponding to Sequential Process
                task_1 = Task(
                    description=f"Answer the following user query using the Hyundai policy knowledge base: '{user_query}'",
                    expected_output="A direct, factual answer to the query derived from the Hyundai policy text document via vector similarity search.",
                    agent=assistant_agent,
                )

                task_2 = Task(
                    description=f"Search the web regarding the user query: '{user_query}', and provide an enriched external perspective.",
                    expected_output="An external web-researched perspective or supporting details related to the query.",
                    agent=web_search_agent,
                )


                # --- Guardrail Function Definition ---
                def support_response_guardrail(
                    result: TaskOutput,
                ) -> Tuple[bool, Any]:
                    output_text = result.raw if result and result.raw else ""
                    if len(output_text.strip()) < 30:
                        return (
                            False,
                            "The output is too brief. Please provide a detailed response containing both Answer 1 and Answer 2.",
                        )
                    return (True, output_text)


                task_3 = Task(
                    description=(
                        "Take the original user query, the output from the Assistant (Answer 1), "
                        "and the output from the Web Search Assistant (Answer 2). Append these entries "
                        "into a local text file named `support_log.txt` with clear timestamps or section separators. "
                        "Then, return a well-formatted summary containing both Answer 1 and Answer 2."
                    ),
                    expected_output="Confirmation that the log file was updated successfully, along with a structured breakdown containing both Answer 1 and Answer 2.",
                    agent=entry_agent,
                    guardrail=support_response_guardrail,
                    guardrail_max_retries=3,
                )

                # 4. Assemble the Crew with Knowledge Source & Memory
                support_crew = Crew(
                    agents=[assistant_agent, web_search_agent, entry_agent],
                    tasks=[task_1, task_2, task_3],
                    process=Process.sequential,
                    knowledge_sources=[rag_knowledge_source],
                    memory=True,
                    embedder={
                        "provider": "openai",
                        "config": {"model": "text-embedding-3-small"},
                    },
                    verbose=True,
                )

                # 5. Kick off crew execution
                result = support_crew.kickoff(inputs={"query": user_query})

                # Display Results in Streamlit UI
                st.success("Hyundai support crew execution completed successfully!")

                st.markdown("### 📋 Final Output & Answers")
                st.markdown(str(result))

                # Ensure the final output is also explicitly saved to support_log.txt as a fallback download asset
                log_file_path = "support_log.txt"
                if not os.path.exists(log_file_path):
                    with open(log_file_path, "w", encoding="utf-8") as f:
                        f.write(f"Query: {user_query}\n\nFinal Output:\n{str(result)}")

                # Handle and display file logs for download
                if os.path.exists(log_file_path):
                    with open(log_file_path, "r", encoding="utf-8") as f:
                        log_contents = f.read()
                    with st.expander("📁 View `support_log.txt` Contents"):
                        st.code(log_contents, language="text")
                    st.download_button(
                        label="📥 Download Support Log (.txt)",
                        data=log_contents,
                        file_name="support_log.txt",
                        mime="text/plain",
                    )

            except Exception as e:
                st.error(f"An error occurred during crew execution: {str(e)}")