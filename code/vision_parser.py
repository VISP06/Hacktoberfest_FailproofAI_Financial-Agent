import os
import json
import base64
import time
import pandas as pd
from litellm import completion
from dotenv import load_dotenv

load_dotenv()

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_receipt_amount(image_path, max_retries=10):
    model_name = os.getenv("BUY_OR_WAIT_MODEL", "gemini/gemini-3.5-flash")
    
    prompt = (
        "Extract the total final amount from this receipt. "
        "Return ONLY a valid JSON object with a single key 'amount' containing the float value. "
        "Example: {\"amount\": 125.50}"
    )
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url", 
                    "image_url": {"url": f"data:image/png;base64,{encode_image(image_path)}"}
                }
            ]
        }
    ]
    
    for attempt in range(max_retries):
        try:
            response = completion(
                model=model_name,
                messages=messages,
                response_format={"type": "json_object"}
            )
            result_str = response.choices[0].message.content
            result = json.loads(result_str)
            amt = float(result.get("amount", 0.0))
            if amt > 0:
                return amt
        except Exception as e:
            wait_time = min(20, 2 ** (attempt + 1))
            print(f"  [!] API busy or error. Retrying in {wait_time}s... (Attempt {attempt + 1}/{max_retries})")
            time.sleep(wait_time)
                
    print(f"  [!] Failed to extract amount from {image_path} after {max_retries} attempts.")
    return 0.0

def patch_missing_amounts(dataset_dir="dataset"):
    print("=== Running Vision Perception Layer (With Caching & Pacing) ===")
    
    events_df = pd.read_csv(f"{dataset_dir}/financial_events.csv", skipinitialspace=True)
    events_df.columns = events_df.columns.str.strip()
    
    images_df = pd.read_csv(f"{dataset_dir}/images.csv", skipinitialspace=True)
    images_df.columns = images_df.columns.str.strip()
    
    # Load existing patched file if present to skip already-extracted images
    patched_path = os.path.join(dataset_dir, "patched_financial_events.csv")
    if os.path.exists(patched_path):
        existing_df = pd.read_csv(patched_path, skipinitialspace=True)
        existing_df.columns = existing_df.columns.str.strip()
        for idx, row in existing_df.iterrows():
            if pd.notna(row['amount']) and float(row['amount']) > 0:
                events_df.at[idx, 'amount'] = float(row['amount'])

    missing_amounts = events_df[events_df['amount'].isna() | (events_df['amount'] == 0)]
    print(f"Found {len(missing_amounts)} events still needing amounts.")
    
    for idx, row in missing_amounts.iterrows():
        event_id = row['event_id']
        match = images_df[images_df['related_event_id'] == event_id]
        if not match.empty:
            image_id = match.iloc[0]['image_id']
            image_path = os.path.join(dataset_dir, "media", "images", f"{image_id}.png")
            
            if os.path.exists(image_path):
                print(f"-> Analyzing {image_id}.png for event {event_id}...")
                extracted_amt = get_receipt_amount(image_path)
                print(f"   Extracted Amount: {extracted_amt}")
                events_df.at[idx, 'amount'] = extracted_amt
                
                # Pace calls slightly to respect rate limits
                time.sleep(2)
            else:
                print(f"  [!] Missing file: {image_path}")
                
    events_df.to_csv(patched_path, index=False)
    print(f"\nSaved complete dataset to {patched_path}")

if __name__ == "__main__":
    patch_missing_amounts()