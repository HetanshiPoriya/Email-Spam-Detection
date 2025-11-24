import os
import pandas as pd
import glob

def combine_files():
    base_dir = 'pu1_encoded/lemm_stop'
    data = []
    
    # Iterate through part1 to part10
    for i in range(1, 11):
        folder_name = f'part{i}'
        folder_path = os.path.join(base_dir, folder_name)
        
        if not os.path.exists(folder_path):
            print(f"Warning: {folder_path} does not exist. Skipping.")
            continue
            
        print(f"Processing {folder_name}...")
        
        # Get all .txt files in the folder
        files = glob.glob(os.path.join(folder_path, '*.txt'))
        
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                filename = os.path.basename(file_path)
                data.append({
                    'folder': folder_name,
                    'filename': filename,
                    'content': content
                })
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    output_file = os.path.join('pu1_encoded', 'combined_lemm_stop.csv')
    df.to_csv(output_file, index=False)
    print(f"Successfully saved {len(df)} records to {output_file}")

if __name__ == "__main__":
    combine_files()
