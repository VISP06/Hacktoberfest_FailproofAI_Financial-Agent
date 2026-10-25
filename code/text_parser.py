import os
import json
import re
import time
import pandas as pd
from litellm import completion
from dotenv import load_dotenv

load_dotenv()

def extract_json_from_text(text_str):
    if not text_str:
        return {}
    cleaned = re.sub(r'```(?:json)?\s*', '', text_str, flags=re.IGNORECASE).strip()
    cleaned = cleaned.rstrip('`').strip()
    
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return {}

def rule_based_fallback(message_text):
    text_lower = str(message_text).lower()
    cancel_keywords = ["cancel", "cancelled", "canceling", "stop", "nevermind", "disregard", "don't proceed", "dont proceed"]
    
    for kw in cancel_keywords:
        if kw in text_lower:
            return {"action": "cancel", "details": f"Rule-based match: {kw}"}
            
    return {"action": "none", "details": "no action"}

def analyze_message_batch(batch_df, max_retries=3):
    model_name = os.getenv("BUY_OR_WAIT_MODEL", "gemini/gemini-1.5-flash")
    
    messages_text = ""
    for _, row in batch_df.iterrows():
        messages_text += f"[{row['message_id']}]: {row['message_text']}\n"
        
    prompt = f"""
    Analyze these financial messages. For each message, does the user explicitly cancel an event, delay an event, or state something else?
    Return ONLY a valid JSON object where keys are exact message IDs, and values are objects with "action" and "details".
    "action" must be strictly one of: "cancel", "delay", "none".
    
    Messages:
    {messages_text}
    """
    
    messages = [{"role": "user", "content": prompt}]
    
    for attempt in range(max_retries):
        try:
            response = completion(
                model=model_name,
                messages=messages,
                response_format={"type": "json_object"}
            )
            result_str = response.choices[0].message.content
            parsed_json = extract_json_from_text(result_str)
            
            if parsed_json:
                return parsed_json
            else:
                print(f"  [!] JSON sanitize retry (Attempt {attempt + 1}/{max_retries})")
                
        except Exception as e:
            # UNMASKED ERROR LOGGING
            print(f"  [!] API Error: {type(e).__name__} - {str(e)}")
            print(f"  [!] Retrying in 3s... (Attempt {attempt + 1}/{max_retries})")
            time.sleep(3)
                
    return None

def parse_messages(dataset_dir="dataset"):
    print("=== Running Text Perception Layer (Deterministic + VLM Hybrid) ===")
    
    messages_path = os.path.join(dataset_dir, "messages.csv")
    out_path = os.path.join(dataset_dir, "parsed_messages.csv")
    
    if not os.path.exists(messages_path):
        print(f"Missing {messages_path}")
        return
        
    messages_df = pd.read_csv(messages_path, skipinitialspace=True)
    messages_df.columns = messages_df.columns.str.strip()
    
    existing_results = {}
    if os.path.exists(out_path):
        try:
            existing_df = pd.read_csv(out_path, skipinitialspace=True)
            existing_df.columns = existing_df.columns.str.strip()
            for _, row in existing_df.iterrows():
                act = str(row.get("action", ""))
                if act in ["cancel", "delay", "none"]:
                    existing_results[str(row["message_id"])] = {
                        "message_id": str(row["message_id"]),
                        "related_event_id": str(row.get("related_event_id", "")),
                        "action": act,
                        "details": str(row.get("details", ""))
                    }
        except Exception:
            existing_results = {}

    unparsed_df = messages_df[~messages_df["message_id"].astype(str).isin(existing_results.keys())]
    print(f"Total messages: {len(messages_df)} | Already parsed: {len(existing_results)} | Remaining to process: {len(unparsed_df)}")
    
    if unparsed_df.empty:
        print("All messages successfully parsed!")
        return

    batch_size = 10
    
    for i in range(0, len(unparsed_df), batch_size):
        batch = unparsed_df.iloc[i:i+batch_size]
        print(f"-> Processing batch {i//batch_size + 1} ({len(batch)} messages)...")
        
        batch_results = analyze_message_batch(batch)
        
        for _, row in batch.iterrows():
            msg_id = str(row['message_id'])
            rel_event = str(row.get('related_event_id', ''))
            msg_text = str(row.get('message_text', ''))
            
            if batch_results and msg_id in batch_results:
                analysis = batch_results[msg_id]
            else:
                analysis = rule_based_fallback(msg_text)
                print(f"   [Fallback applied for {msg_id} -> {analysis['action']}]")
            
            existing_results[msg_id] = {
                "message_id": msg_id,
                "related_event_id": rel_event,
                "action": analysis.get("action", "none"),
                "details": analysis.get("details", "")
            }
        
        out_df = pd.DataFrame(list(existing_results.values()))
        out_df.to_csv(out_path, index=False)
        print(f"   [Progress saved: {len(existing_results)}/{len(messages_df)} messages]")
        time.sleep(2)
        
    print(f"\nCompleted parsing all messages into {out_path}")

if __name__ == "__main__":
    parse_messages()