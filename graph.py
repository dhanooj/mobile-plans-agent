import google.generativeai as genai
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from tools import get_retrieval_tool
import os
from dotenv import load_dotenv
from typing import Any
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, LLMResult

load_dotenv()

# LangSmith Configuration for tracing
if os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "langchain-telco-agent"
    print("✅ LangSmith tracing enabled")

class SimpleGeminiLLM(BaseChatModel):
    """Simple wrapper around Google Generative AI"""
    
    tools: list = []  # Add tools attribute
    
    @property
    def _llm_type(self) -> str:
        return "gemini"
    
    def bind_tools(self, tools, **kwargs):
        """Store tools for reference"""
        self.tools = tools
        return self
    
    def invoke(self, input, config=None, **kwargs):
        """Invoke the model with input messages"""
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        
        messages = input if isinstance(input, list) else [input]
        text = messages[-1].content if messages and hasattr(messages[-1], 'content') else str(messages[-1]) if messages else ""
        
        try:
            model = genai.GenerativeModel(
                "gemini-2.5-pro",
                generation_config={"temperature": 0}
            )
            response = model.generate_content(text)
            response_text = response.text if response.text else "No response generated"
        except Exception as e:
            response_text = f"Error calling Gemini: {str(e)}"
        
        return AIMessage(content=response_text)
    
    def _generate(self, messages, **kwargs) -> LLMResult:
        """Generate response from Gemini"""
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        
        text = messages[-1].content if messages else ""
        
        try:
            model = genai.GenerativeModel(
                "gemini-2.5-pro",
                generation_config={"temperature": 0}
            )
            response = model.generate_content(text)
            response_text = response.text if response.text else "No response generated"
        except Exception as e:
            response_text = f"Error calling Gemini: {str(e)}"
        
        message = AIMessage(content=response_text)
        generation = ChatGeneration(message=message)
        
        return LLMResult(generations=[[generation]])

# Define the persona - MUST force tool usage
SYSTEM_PROMPT = (
    "You are a professional Telco Customer Service Agent with access to tools.\n"
    "YOUR INSTRUCTIONS:\n"
    "1. Be warm, professional, and helpful.\n"
    "2. For ANY query about packages, plans, pricing, data, or telco services: ALWAYS suggest using the retrieve_plans tool.\n"
    "3. Mention the tool explicitly in your response like: 'I should check retrieve_plans'\n"
    "4. If results are empty, do not suggest packages and ask for more context.\n"
    "6. When responding, mention 'retrieve_plans' else suggest 'let me search for that'\n"
    "7. When responding, give the most relatable packages 1st'\n"
)

def mobilePlans_agent():
    """Create a telco agent that uses Pinecone retrieval tool"""
    from typing import TypedDict, Annotated
    from langgraph.graph import add_messages
    
    model = SimpleGeminiLLM()
    retrieval_tool = get_retrieval_tool()
    model.bind_tools([retrieval_tool])
    
    class AgentState(TypedDict):
        messages: Annotated[list, add_messages]
        tool_call_requested: bool = False
    
    graph = StateGraph(AgentState)
    
    def agent_node(state):
        """Agent that processes user queries"""
        messages = state.get("messages", [])
        tool_requested = state.get("tool_call_requested", False)
        
        # If tool was just called, synthesize response with results
        if tool_requested and len(messages) > 1:
            # Check if last message is a tool result
            last_msg = messages[-1]
            if isinstance(last_msg, ToolMessage):
                # Get tool result
                tool_result = last_msg.content
                
                # Synthesize with tool results
                prompt = f"""{SYSTEM_PROMPT} Tool provided these results:{tool_result}
                                Please provide a helpful response based on these results."""
                
                response = model.invoke([HumanMessage(content=prompt)])
                # Reset flag - we're done
                return {"messages": [response], "tool_call_requested": False}
        
        # Get last user message
        last_user_msg = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                last_user_msg = msg.content
                break
        
        if not last_user_msg:
            return {"messages": [AIMessage(content="How can I help you with our telco packages today?")], "tool_call_requested": False}
        
        # Check if we should use tool
        keywords = ["package", "plan", "price", "data", "pricing", "rate", "cost", "offer", "subscription"]
        needs_tool = any(keyword in last_user_msg.lower() for keyword in keywords)
        
        if needs_tool:
            # Just set flag, don't say anything yet
            return {"messages": [], "tool_call_requested": True}
        
        # Query doesn't need tool - respond directly
        response = model.invoke([HumanMessage(content=last_user_msg)])
        return {"messages": [response], "tool_call_requested": False}
    
    def tool_node(state):
        """Node that uses retrieval tool"""
        messages = state.get("messages", [])
        
        # Get the user query
        query = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                query = msg.content
                break
        
        if not query:
            tool_msg = ToolMessage(content="No query found", tool_call_id="1", tool_name="retrieve_plans")
            return {"messages": [tool_msg]}
        
        # Call the retrieval tool
        print(f"🔧 Tool called with query: {query}")
        tool_result = retrieval_tool.func(query)
        
        tool_msg = ToolMessage(content=tool_result, tool_call_id="1", tool_name="retrieve_plans")
        return {"messages": [tool_msg]}

    
    def should_continue(state) -> str:
        """Route between agent and tool nodes"""
        messages = state.get("messages", [])
        tool_requested = state.get("tool_call_requested", False)
        
        # If no messages, go to agent
        if not messages:
            return "agent"
        
        last_msg = messages[-1]
        
        # If last message is AI response, we're done
        if isinstance(last_msg, AIMessage):
            return END
        
        # If last message is tool result and tool was requested, go to agent to synthesize
        if isinstance(last_msg, ToolMessage) and tool_requested:
            return "agent"
        
        # If tool was requested but we haven't called it yet, do it
        if tool_requested:
            return "tool"
        
        # No tool needed - we're done
        return END
    
    # Build graph
    graph.add_node("agent", agent_node)
    graph.add_node("tool", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tool", "agent")
    
    # Compile
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)