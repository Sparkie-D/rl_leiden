import os
import pandas as pd
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='/home/ubuntu/duxinghao/clone/data/lineage_trace_data')
    parser.add_argument('--output_file', type=str, default='/home/ubuntu/duxinghao/clone/data/concat')
    return parser.parse_args()

def concatenate_cnv_files(input_dir, output_file):
    cnv_files = [f for f in os.listdir(input_dir) if f.endswith('CNV.csv')]
    if not cnv_files:
        raise ValueError("No CNV CSV files found in the specified directory.")
    
    concatenated_data = []
    meta_data = []

    for file in cnv_files:
        file_path = os.path.join(input_dir, file)
        df = pd.read_csv(file_path, index_col=0)
        print(file_path, df.shape)

        # Append the origin file name to each column name for tracking
        df.columns = [col.strip() for col in df.columns]
        meta_data.extend([(col, file) for col in df.columns])

        # Concatenate data
        concatenated_data.append(df)

    # Create the meta DataFrame
    concatenated_data = pd.concat(concatenated_data, axis=1)

    # Fill missing values with global mode
    global_mode = concatenated_data.mode().iloc[0]
    concatenated_data.fillna(global_mode, inplace=True)

    meta_df = pd.DataFrame(meta_data, columns=["Column", "Source File"])
    print(meta_df.shape, concatenated_data.shape)
    
    # Save concatenated data and meta information
    concatenated_data.to_csv(f"{output_file}_concatenated.csv")
    meta_df.to_csv(f"{output_file}_meta.csv", index=False)
    print(f"Files concatenated and saved at .csv and {output_file}_meta.csv")

if __name__ == '__main__':
    args = get_args()
    concatenate_cnv_files(args.input_dir, args.output_file)
