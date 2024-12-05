# Script: process_energies.py

input_file = "energies.txt"
output_file = "plot_data.txt"

# Open the input file and output file
with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    for line in infile:
        # Split the line at 'output:::'
        parts = line.split('output:::')
        if len(parts) < 2:
            continue
        
        # Extract the x-value from the first column (float before '/')
        x_value = parts[0].split('/')[0].strip()
        
        # Extract the energy (last part of the line after "Total energy:")
        energy = parts[1].split('Total energy:')[-1].strip()
        
        # Write the extracted values to the output file
        outfile.write(f"{x_value} {energy}\n")

print(f"Processed data saved to {output_file}")


