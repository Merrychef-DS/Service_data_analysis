# import pandas as pd
#
# # File paths
# organized_file_path = "x12_part_list_no_duplicates.csv"  # Replace with your actual path
# processed_file_path = "Processed_Service_Incident_Summary_Keywords_for_ConneX_12.csv"  # Replace with your actual path
#
# # Load the CSV files
# organized_df = pd.read_csv(organized_file_path)
# processed_df = pd.read_csv(processed_file_path)
#
# # Convert part numbers to string and strip any spaces for consistent matching
# organized_df['Part Number'] = organized_df['Part Number'].astype(str).str.strip()
# processed_df['Part Number'] = processed_df['Part Number'].astype(str).str.strip()
#
# # Merge the two dataframes based on part number, keeping the keywords for n=3
# merged_df = pd.merge(organized_df, processed_df[['Part Number', 'Keywords for n=3']], on='Part Number', how='left')
#
# # Save the merged file to a new CSV
# output_file = "organized_with_keywords2.csv"
# merged_df.to_csv(output_file, index=False)
#
# print(f"Data merged and saved to {output_file}")

import pandas as pd

# Load both CSV files into pandas DataFrames
first_csv = pd.read_csv('C:/Users/mn1006/Downloads/x12_part_list_no_duplicates 1.csv')
second_csv = pd.read_csv("R:/K_Archive_/Claim data analysis/OUTPUT_CSV_CPS_TESSRACT/Tesract_and_cps_data_2024-05-29 10-11-54.csv")

# Merge the DataFrames on 'Part Number' from first_csv and 'FSRL_Part_Num' from second_csv
merged_df = pd.merge(first_csv, second_csv[['FSRL_Part_Num', 'FSR_Solution']],
                     left_on='Part Number', right_on='FSRL_Part_Num', how='left')

# Export the result to a new CSV file
merged_df.to_csv('merged_file_with_solution.csv', index=False)

print("The merged file with the FSR_Solution column has been successfully exported!")


