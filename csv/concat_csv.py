import pandas as pd
# concat given csv files into one csv file
def concat_csv(csv_files, output_file):
    dfs = []
    for csv_file in csv_files:
        dfs.append(pd.read_csv(csv_file))
    df = pd.concat(dfs)
    df.to_csv(output_file, index=False)
    print("Concatenated", len(csv_files), "csv files into", output_file)
    
    
if __name__=="__main__":
    csv_files = [
        "roof_houses_all_output.csv",
        "roof_oberbayern-houses-orientation.csv"
    ]
    concat_csv(csv_files, "roof_houses_all_final.csv")