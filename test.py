import serial
import pandas as pd
import time

# Set up the serial connection (adjust COM port and baud rate as needed)
arduino = serial.Serial(port = 'COM8', baudrate = 9600)  # Replace 'COM3' with your Arduino port (Linux: '/dev/ttyUSB0')

# Initialize an empty list to store the data
data_list = []

try:
    while True:
        if arduino.in_waiting:

            # Read the incoming data from Arduino and decode it
            data = arduino.readline().decode('utf-8').strip()
            print(f"Received: {data}")  # For debugging

            # Capture the current timestamp
            timestamp = time.strftime('%H:%M:%S')  # Format: YYYY-MM-DD HH:MM:SS

            # Append the timestamp and data to the list
            data_list.append([timestamp] + data.split(','))  # Assuming data is comma-separated

            # Optional: Stop after a certain number of readings (e.g., 1000 rows)
            if len(data_list) > 1000:
                break
except KeyboardInterrupt:
    print("Data collection stopped.")

# Create a DataFrame with the collected data (including timestamp)
df = pd.DataFrame(data_list, columns=['Timestamp', 'Data1:RED'])  # Adjust column names and number of columns as per your data
# df = pd.DataFrame(data_list, columns=['Timestamp', 'Data1:IR', 'Data2:RED', 'Data3:SpO2'])  # Adjust column names and number of columns as per your data
# df = pd.DataFrame(data_list, columns=[ 'Data1:IR', 'Data2:BPM','Data3:Avg BPM','Data4:Finger'])

# Save the DataFrame to Excel
df.to_excel(r'C:\Users\Soumyadeep\Desktop\Local Docs\Healthcare\PPG DATA\arduino_data_with_time_trans_10.xlsx', index=False)

# Close the serial connection
arduino.close()

# # 100 -> 15s
# # 1000 -> 2m 15s


# # Load the two Excel files
# file1 = r"C:\Users\Soumyadeep\Desktop\Local Docs\Healthcare\PPG DATA\dummy01.xlsx"
# file2 = r"C:\Users\Soumyadeep\Desktop\Local Docs\Healthcare\PPG DATA\dummy02.xlsx"

# df1 = pd.read_excel(file1)
# df2 = pd.read_excel(file2)

# # Display the first few rows of each DataFrame to verify
# # print(df1.head())
# # print(df2.head())

# # Ensure timestamps are in a consistent format
# # # Example format: 'YYYY-MM-DD HH:MM:SS' (adjust to match your actual format)
# # df1['Timestamp'] = pd.to_datetime(df1['Timestamp'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
# # df2['Timestamp'] = pd.to_datetime(df2['Timestamp'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

# # Rename columns to avoid conflicts
# df1.rename(columns={'Timestamp': 'time_1', 'Data1:IR': 'ir_1', 'Data2:RED': 'red_1', 'Data3:SpO2': 'spo2_1'}, inplace=True)
# df2.rename(columns={'Timestamp': 'time_2', 'Data1:IR': 'ir_2', 'Data2:RED': 'red_2', 'Data3:SpO2': 'spo2_2'}, inplace=True)

# # Display the first few rows to verify changes
# # print(df1.head())
# # print(df2.head())

# # Merge the DataFrames on their time columns
# merged_df = pd.merge_asof(df1, df2, left_on='time_1', right_on='time_2', direction='nearest')

# # # Drop duplicate time columns if needed
# merged_df.drop(columns=['time_2'], inplace=True)

# # Fill NaN values (if any) after merging
# merged_df.ffill(inplace=True)

# # Combine the IR and Red values by averaging them (or use another method as needed)
# merged_df['ir_combined'] = (merged_df['ir_1'] + merged_df['ir_2']) / 2
# merged_df['red_combined'] = (merged_df['red_1'] + merged_df['red_2']) / 2

# # Display the first few rows of the merged DataFrame
# print(merged_df.head())
