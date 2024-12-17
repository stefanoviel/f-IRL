import pandas as pd
import glob
import re
import os

class ExperimentDataProcessor:
    def __init__(self, base_path):
        self.base_path = base_path
        self.cache_dir = os.path.join('plots', 'cached_data')
        os.makedirs(self.cache_dir, exist_ok=True)

    @staticmethod
    def _extract_q(folder_name):
        match = re.search(r'q(\d+)_', folder_name)
        return int(match.group(1)) if match else None

    @staticmethod
    def _extract_clip(folder_name):
        match = re.search(r'qstd(\d+\.\d+)', folder_name)
        return float(match.group(1)) if match else None

    @staticmethod
    def _extract_seed(folder_name):
        match = re.search(r'seed(\d+)', folder_name)
        return int(match.group(1)) if match else None

    def _get_cache_path(self):
        # Create a unique cache file name based on the base path
        env_name = self.base_path.split('/')[1]
        exp_name = self.base_path.split('/')[2]
        method = self.base_path.split('/')[3]
        return os.path.join(self.cache_dir, f'{env_name}_{exp_name}_{method}_data.csv')

    def load_data(self, use_cache=True):
        cache_path = self._get_cache_path()
        
        if use_cache and os.path.exists(cache_path):
            print(f"Loading cached data from {cache_path}")
            return pd.read_csv(cache_path)

        data_rows = []
        folders = glob.glob(f'{self.base_path}2024_*_seed*')
        
        # Extract environment name from base_path
        env_name = self.base_path.split('/')[1]

        for folder in folders:
            q = self._extract_q(folder)
            clip = self._extract_clip(folder)
            seed = self._extract_seed(folder)

            print(f"Processing folder: {folder}, q={q}, clip={clip}, seed={seed}")

            if q is not None:
                try:
                    df = pd.read_csv(f'{folder}/progress.csv')
                    # Add experiment metadata to each row
                    df['q'] = q
                    df['clip'] = clip
                    df['folder'] = folder
                    df['episode'] = df['Itration'] * 5000  # Convert iterations to episodes
                    df['environment'] = env_name  # Add environment name
                    df['seed'] = seed
                    data_rows.append(df)
                except Exception as e:
                    print(f"Error reading file in folder: {folder}: {e}")

        if not data_rows:
            raise ValueError(f"No data found in {self.base_path}")

        # Combine all data into a single DataFrame
        combined_df = pd.concat(data_rows, ignore_index=True)
        
        # Save to cache
        combined_df.to_csv(cache_path, index=False)
        print(f"Cached data saved to {cache_path}")
        
        return combined_df


    @classmethod
    def process_experiments(cls, root_path, output_dir):
        """
        Process all experiments in the root path and save their data to output_dir.
        
        Args:
            root_path (str): Path containing multiple experiment folders
                           (e.g., "logs/Ant-v5/")
            output_dir (str): Directory where processed data will be saved
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all method folders (e.g., rkl, cisl)
        exp_folders = []
        for env_folder in glob.glob(os.path.join(root_path, "*/")):
            for exp_folder in glob.glob(os.path.join(env_folder, "exp-*/")):
                for method_folder in glob.glob(os.path.join(exp_folder, "*/")):
                    if os.path.isdir(method_folder):
                        exp_folders.append(method_folder)
        
        print(f"Found {len(exp_folders)} experiment folders to process")
        
        for exp_folder in exp_folders:

            try:
                # Create processor for this experiment
                processor = cls(exp_folder)
                
                # Load and process the data
                df = processor.load_data(use_cache=False)  # Force reprocessing
                
                # Create output filename based on experiment path
                parts = exp_folder.strip('/').split('/')
                env_name = parts[-4]  # e.g., "Ant-v5"
                exp_name = parts[-3]  # e.g., "exp-16"
                method = parts[-2]    # e.g., "rkl"
                
                output_file = os.path.join(output_dir, f'{env_name}_{exp_name}_{method}_data.csv')
                
                # Save processed data
                df.to_csv(output_file, index=False)
                # print(f"Processed and saved data to {output_file}")
                
            except Exception as e:
                print(f"Error processing {exp_folder}: {e}")


if __name__ == "__main__":
    root_path = "logs"
    output_dir = "processed_data"
    ExperimentDataProcessor.process_experiments(root_path, output_dir)
