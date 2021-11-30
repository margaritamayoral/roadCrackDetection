import json 
import os 
import psycopg2

conn = psycopg2.connect(host="dbremote.ckjgcig5seif.ca-central-1.rds.amazonaws.com",
	database="label_db", 
	user="master", 
	password="$treetScan!")
cur = conn.cursor()
cur.execute("SELECT * FROM label.labels")

print("======================================================================")
print("fetch one row of data from database...")
print("======================================================================\n")
one_feat = cur.fetchone()
print(one_feat)
print("\n")
print("======================================================================")
print("fetch all rows of data from database...")
print("======================================================================\n")
all_feat = cur.fetchall()
print(all_feat)
print("\n")

#Get the file name for the new file to write
f = open("masks.json","w+")

# If the file name exists, write a JSON string into the file.
if f:
    	# Writing JSON DATA
	json.dump(all_feat, f)


conn.close()
