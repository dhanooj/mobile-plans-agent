from graph import mobilePlans_agent

def run_chat():
    agent = mobilePlans_agent()
    # Unique ID for the current user session
    config = {"configurable": {"thread_id": "user_12345"}}
    
    print("--- Telco Assistant Active ---")
    
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            break
            
        inputs = {"messages": [("user", user_input)]}
        
        # Stream the response and collect final message
        final_response = None
        for event in agent.stream(inputs, config=config):
            for key, value in event.items():
                if isinstance(value, dict) and "messages" in value:
                    messages = value["messages"]
                    if messages:
                        last_msg = messages[-1]
                        # Check for final AI message
                        if hasattr(last_msg, 'type') and last_msg.type == "ai":
                            final_response = last_msg.content
        
        if final_response:
            print(f"Assistant: {final_response}\n")

if __name__ == "__main__":
    run_chat()