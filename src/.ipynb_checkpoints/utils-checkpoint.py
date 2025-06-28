def load_data(csv_path:str) -> pandas.DataFrame:
    import os
    import pandas as pd
    
    cwd = os.getcwd()
    
    return pd.read_csv(os.path.join(cwd, csv_path))