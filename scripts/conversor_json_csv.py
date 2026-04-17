import json
import pandas as pd

def convertir_json_to_df(file_obj):
    try:
        # Assuming the utf-16 decoding or normal json if it's already an object
        if hasattr(file_obj, 'read'):
            raw_data = file_obj.read()
            # Handle bytes (streamlit uploader) vs str
            if isinstance(raw_data, bytes):
                try:
                    text_data = raw_data.decode("utf-16")
                except:
                    text_data = raw_data.decode("utf-8")
                gross_data = json.loads(text_data)
            else:
                gross_data = json.loads(raw_data)
        else:
            gross_data = json.load(open(file_obj, 'r', encoding="utf-16"))
        
        gross_data = gross_data.get("result", gross_data)
        
        if 'data' in gross_data and 'data' in gross_data['data']:
            records = gross_data['data']['data']
        else:
            records = gross_data
            
        if records:
            df = pd.DataFrame(records)
            return df
        else:
            return pd.DataFrame()
            
    except json.JSONDecodeError as e:
        print(f"Error parseando JSON: {e}")
        return None
