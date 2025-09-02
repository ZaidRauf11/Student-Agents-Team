import streamlit as st
from dotenv import load_dotenv
import os

# Agno imports
from agno.agent import Agent
from agno.team import Team
from agno.models.groq import Groq
from agno.tools.googlesearch import GoogleSearchTools
from agno.tools.arxiv import ArxivTools
from agno.tools.hackernews import HackerNewsTools

# Load environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# ------------------------
# Define Agents
# ------------------------

news_agent = Agent(
    name="News Analyst",
    model=Groq(id="qwen/qwen3-32b"),
    role="Find recent news on sustainability initiatives",
    tools=[GoogleSearchTools(), ArxivTools()],
    instructions="Search for city-level green projects in the past year.",
    show_tool_calls=True,
    markdown=True,
)

policy_agent = Agent(
    name="Policy Reviewer",
    model=Groq(id="qwen/qwen3-32b"),
    role="Summarize government policies",
    tools=[GoogleSearchTools()],
    instructions="Search official sites for city policy updates.",
    show_tool_calls=True,
    markdown=True,
)

innovation_agent = Agent(
    name="Innovations Scout",
    model=Groq(id="qwen/qwen3-32b"),
    role="Find innovative greentech ideas",
    tools=[GoogleSearchTools(), HackerNewsTools()],
    instructions="Look for urban sustainability and smart city innovations.",
    show_tool_calls=True,
    markdown=True,
)

data_agent = Agent(
    name="Data Analyst",
    model=Groq(id="qwen/qwen3-32b"),
    role="Analyze sustainability datasets and trends",
    tools=[GoogleSearchTools()],
    instructions="Find and summarize sustainability-related statistics and reports.",
    show_tool_calls=True,
    markdown=True,
)

# ------------------------
# Create Team (with members argument)
# ------------------------

team_agent = Team(
    name="Student Agents Team",
    model=Groq(id="qwen/qwen3-32b"),
    members=[news_agent, policy_agent, innovation_agent, data_agent],
    instructions="Coordinate specialized agents to provide comprehensive insights.",
    show_tool_calls=True,
    markdown=True,
)

# ------------------------
# Streamlit Interface
# ------------------------

st.set_page_config(page_title="Student Agents Team", layout="wide")

st.title("🎓 Student Agents Team")
st.write("Choose either a single agent or the full task force to answer your query.")

# Sidebar for agent selection
st.sidebar.header("Choose an Agent")
agent_choice = st.sidebar.radio(
    "Select which agent to use:",
    ["News Analyst", "Policy Reviewer", "Innovations Scout", "Data Analyst", "Full Team"],
)

# Main input

user_query = st.text_input("💬 Enter your query:")

if st.button("Run Analysis"):

    if not user_query.strip():
        st.warning("⚠️ Please enter a query before running.")
    else:
        with st.spinner(f"🔍 Running {agent_choice}..."):
            try:
                # Select agent based on choice
                if agent_choice == "News Analyst":
                    response = news_agent.run(user_query)
                    title = "📰 News Analyst Result"
                elif agent_choice == "Policy Reviewer":
                    response = policy_agent.run(user_query)
                    title = "📜 Policy Reviewer Result"
                elif agent_choice == "Innovations Scout":
                    response = innovation_agent.run(user_query)
                    title = "💡 Innovations Scout Result"
                elif agent_choice == "Data Analyst":
                    response = data_agent.run(user_query)
                    title = "📊 Data Analyst Result"
                else:  # Full Team
                    response = team_agent.run(user_query)
                    title = "🌍 Full Team Proposal"

                # Extract only human-readable content
                if hasattr(response, "content"):
                    readable_output = response.content
                else:
                    readable_output = str(response)

                st.subheader(title)
                st.markdown(readable_output)

            except Exception as e:
                st.error(f"Error: {e}")









