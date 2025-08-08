#!/bin/bash

# Philippe Joly 2025-06-02
# This BASH script is designed to convert A/E-CHAIM database data to csv format. 
# More precisely it extracts A/E-CHAIM maximal F2 electron density at recorded times.
#
# Uage:
# Ensure to check all the paths in CHAIM_extract.c Makefile and this file to match your environment. load gcc/11.4.0
# Run
#     ./CHAIM_db_to_csv.sh <db path> 



DIRECTORY="${1:-/project/s/sievers/philj0ly/CHAIM/A-CHAIM_User_Release-6.0.2/For_Mohan}"
A_CSV="/project/s/sievers/philj0ly/CHAIM/CSV/ACHAIM_nmf2.csv"
E_CSV="/project/s/sievers/philj0ly/CHAIM/CSV/ECHAIM_nmf2.csv"

echo "datetime,nmf2" > "$A_CSV"
echo "datetime,nmf2" > "$E_CSV"

FC=0

for FILE in "$DIRECTORY"/output_t-02h_*_v5.1.0.db; do
    # Extract the datetime part using regex pattern
    if [[ $FILE =~ output_t-02h_([0-9]{6}_[0-9]{6})_v5\.1\.0\.db$ ]]; then
        DATETIME="${BASH_REMATCH[1]}"
        
        OUTPUT=$(./CHAIM_extract.o "$FILE")
        A_RESULT=$(echo "$OUTPUT" | awk '{print $1}')
        E_RESULT=$(echo "$OUTPUT" | awk '{print $2}')
        
        echo "$DATETIME,$A_RESULT" >> "$A_CSV"
        echo "$DATETIME,$E_RESULT" >> "$E_CSV"
        FC=$((FC+1))
    fi
done

echo "$FC maximal F2 electron densities Extracted and stored in"
echo $A_CSV
echo $E_CSV

exit 0