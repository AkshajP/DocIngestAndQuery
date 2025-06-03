import os
import pandas as pd
from pathlib import Path

def walk_folders_dfs(start_path, start_count=1):
    """
    Perform depth-first search on folder structure and count files in order of appearance.
    Processes folders first (going deep), then files in current directory.
    Excludes files starting with dot (.) or dot underscore (._).
    
    Args:
        start_path (str): Path to the starting folder
        start_count (int): Starting count number
    
    Returns:
        pandas.DataFrame: DataFrame with file paths and order numbers
    """
    file_data = []
    counter = start_count
    
    def dfs_recursive(current_path):
        nonlocal counter
        
        try:
            # Get all items in current directory
            items = sorted(os.listdir(current_path))
            
            # Separate files and directories
            files = []
            directories = []
            
            for item in items:
                item_path = os.path.join(current_path, item)
                if os.path.isfile(item_path):
                    # Exclude files starting with . or ._
                    if not (item.startswith('.') or item.startswith('._')):
                        files.append(item_path)
                elif os.path.isdir(item_path):
                    directories.append(item_path)
            
            # First, recursively process subdirectories (depth-first)
            for dir_path in directories:
                dfs_recursive(dir_path)
            
            # Then, add all files in current directory
            for file_path in files:
                file_data.append({
                    'file_path': file_path,
                    'order': counter
                })
                counter += 1
                
        except PermissionError:
            print(f"Permission denied: {current_path}")
        except Exception as e:
            print(f"Error processing {current_path}: {e}")
    
    # Start the recursive DFS
    dfs_recursive(start_path)
    
    # Create DataFrame
    df = pd.DataFrame(file_data)
    return df

def walk_folders_dfs_alternative(start_path, start_count=1):
    """
    Alternative implementation using pathlib for folders-first DFS traversal.
    Excludes files starting with dot (.) or dot underscore (._).
    Note: The recursive method above is better suited for this specific ordering.
    
    Args:
        start_path (str): Path to the starting folder
        start_count (int): Starting count number
    
    Returns:
        pandas.DataFrame: DataFrame with file paths and order numbers
    """
    file_data = []
    counter = start_count
    
    def collect_paths_dfs(path):
        """Collect all file paths in folders-first DFS order."""
        path_obj = Path(path)
        all_files = []
        
        try:
            # Get subdirectories first
            subdirs = sorted([p for p in path_obj.iterdir() if p.is_dir()])
            
            # Recursively collect from subdirectories first
            for subdir in subdirs:
                all_files.extend(collect_paths_dfs(subdir))
            
            # Then add files from current directory (excluding hidden files)
            files = sorted([p for p in path_obj.iterdir() 
                          if p.is_file() and not (p.name.startswith('.') or p.name.startswith('._'))])
            all_files.extend([str(f) for f in files])
            
        except PermissionError:
            print(f"Permission denied: {path}")
        except Exception as e:
            print(f"Error processing {path}: {e}")
        
        return all_files
    
    # Collect all file paths in the correct order
    all_file_paths = collect_paths_dfs(start_path)
    
    # Create the data with order numbers
    for file_path in all_file_paths:
        file_data.append({
            'file_path': file_path,
            'order': counter
        })
        counter += 1
    
    # Create DataFrame
    df = pd.DataFrame(file_data)
    return df

def append_to_csv(df, filename):
    """
    Append dataframe to existing CSV or create new one if it doesn't exist.
    Adjusts the starting count based on existing data.
    """
    if os.path.exists(filename):
        # Read existing CSV to get the last order number
        existing_df = pd.read_csv(filename, index_col=0)
        
        if not existing_df.empty:
            # Get the last order number and adjust new df
            last_order = existing_df['order'].max()
            df['order'] = df['order'] - df['order'].min() + last_order + 1
        
        # Append without header since file already exists
        df.to_csv(filename, mode='a', header=False, index=True)
        print(f"Appended {len(df)} rows to existing {filename}")
    else:
        # Create new file with header
        df.to_csv(filename, index=True)
        print(f"Created new {filename} with {len(df)} rows")
        
# Usage example
if __name__ == "__main__":
    # Example usage - processes folders first (deepest first), then files
    folder_path = '/Volumes/Extreme SSD/[N] Respondent Expert Report Exhibits/Exhibits of 2nd Independent Quantum Expert Report by Mr. Michael Davies/Exhibit RX0MD-2-16 - Claim 1'
    start_count = 55
    
    print("Method 1: Recursive DFS (Folders first, then files) - Excluding hidden files")
    df1 = walk_folders_dfs(folder_path, start_count)
    print(df1.head(10))
    print(f"Total files found: {len(df1)}")
    
    print("\nMethod 2: Pathlib-based DFS - Excluding hidden files")
    df2 = walk_folders_dfs_alternative(folder_path, start_count)
    print(df2.head(10))
    print(f"Total files found: {len(df2)}")
    print(f"Last order number: {df2.iloc[-1]['order']}")
    
    # Expected output order (hidden files excluded):
    # /folder/subfolder/a.txt     1    (but not .hidden.txt or ._temp.txt)
    # /folder/subfolder/b.txt     2 
    # /folder/subfolder2/c.txt    3
    # /folder/file1.txt           4    (but not .DS_Store or ._metadata)
    # /folder/file2.txt           5
    # /file3.txt                  6
    
    # Save to CSV if needed
    df2.to_csv('Quantum_report_file_order_dfs5.csv', index=True)