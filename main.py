import os
import csv
import time
import warnings
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from google import genai

# 1. SETUP & SILENCE WARNINGS
warnings.filterwarnings("ignore")
load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
MODEL_ID = "gemini-flash-latest"

# 2. STATE DEFINITION
class AgentState(TypedDict):
    order_index: int      # Tracking our place in the CSV
    found_data: str       # Current row data
    decision: str         # AI status
    valid_status: bool    # Validator result
    logs: List[str]

# 3. NODES

def investigator_node(state: AgentState):
    idx = state.get("order_index", 0)
    print(f"\n--- 🔍 INVESTIGATOR: Fetching Row {idx + 1} ---")
    
    try:
        with open("orders.csv", mode="r") as f:
            reader = list(csv.DictReader(f))
            if idx < len(reader):
                row = reader[idx]
                data_str = f"ID: {row['id']}, Status: {row['status']}, Reason: {row['reason']}"
                return {"found_data": data_str, "order_index": idx, "logs": [f"Investigator: Read Order {row['id']}"]}
            return {"found_data": "EOF"}
    except FileNotFoundError:
        print("❌ Error: orders.csv not found!")
        return {"found_data": "ERROR"}

def auditor_node(state: AgentState):
    print("--- 🤔 AUDITOR: Analyzing Logic ---")
    prompt = f"Data: {state['found_data']}. Reply APPROVED or REJECTED. One word only."
    try:
        response = client.models.generate_content(model=MODEL_ID, contents=prompt)
        decision = response.text.strip().upper().replace("*", "")
    except:
        decision = "ERROR"
    return {"decision": decision, "logs": state["logs"] + [f"Auditor: {decision}"]}

def validator_node(state: AgentState):
    print("--- 🛡️ VALIDATOR: Red-Teaming ---")
    prompt = f"Data: {state['found_data']}. Auditor said {state['decision']}. Is this correct? YES/NO."
    try:
        response = client.models.generate_content(model=MODEL_ID, contents=prompt)
        is_valid = "YES" in response.text.strip().upper()
    except:
        is_valid = False
    
    final_decision = state["decision"] if is_valid else "HUMAN_REVIEW_REQUIRED"
    return {"decision": final_decision, "valid_status": is_valid, "logs": state["logs"] + [f"Validator: Verified={is_valid}"]}

def reporter_node(state: AgentState):
    print("--- 📝 REPORTER: Logging Result ---")
    file_exists = os.path.isfile("results_log.csv")
    with open("results_log.csv", "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Index", "Decision", "Verified"])
        writer.writerow([state['order_index'], state['decision'], state.get('valid_status', False)])
    
    # Increment index for the next loop
    return {"order_index": state["order_index"] + 1}

# 4. ROUTING LOGIC

def should_continue(state: AgentState):
    """Check if there are more rows to process."""
    with open("orders.csv", "r") as f:
        total_rows = len(list(csv.DictReader(f)))
    return "continue" if state["order_index"] < total_rows else "end"

def audit_routing(state: AgentState):
    """Skip validation if already rejected."""
    return "verify" if state["decision"] == "APPROVED" else "skip"

# 5. GRAPH CONSTRUCTION

workflow = StateGraph(AgentState)

workflow.add_node("investigator", investigator_node)
workflow.add_node("auditor", auditor_node)
workflow.add_node("validator", validator_node)
workflow.add_node("reporter", reporter_node)

workflow.set_entry_point("investigator")
workflow.add_edge("investigator", "auditor")
workflow.add_conditional_edges("auditor", audit_routing, {"verify": "validator", "skip": "reporter"})
workflow.add_edge("validator", "reporter")

# Recursive loop back to start
workflow.add_conditional_edges("reporter", should_continue, {"continue": "investigator", "end": END})

app = workflow.compile()

# 6. ANALYTICS & EXECUTION

def generate_final_chart():
    """Generates a visual bar chart of the swarm's performance."""
    if not os.path.exists("results_log.csv"): return
    
    decisions = []
    with open("results_log.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            decisions.append(row["Decision"])
    
    counts = {status: decisions.count(status) for status in set(decisions)}
    
    plt.figure(figsize=(8, 5))
    colors = ['#4CAF50' if k == 'APPROVED' else '#F44336' for k in counts.keys()]
    plt.bar(counts.keys(), counts.values(), color=colors)
    plt.title("Sentinel Swarm: Batch Results")
    plt.xlabel("Decision Status")
    plt.ylabel("Count")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig("swarm_summary.png")
    print("\n📊 Analytics generated: swarm_summary.png")

if __name__ == "__main__":
    print("🚀 Sentinel Swarm: Batch ETL Mode Active")
    
    # Cleanup old run data
    if os.path.exists("results_log.csv"): os.remove("results_log.csv")
    
    # Run Swarm
    app.invoke({"order_index": 0, "found_data": "", "decision": "", "valid_status": False, "logs": []})
    
    print("\n✅ All orders processed.")
    generate_final_chart()