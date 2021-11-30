import os
import psycopg2
import config_0

###################### WRITE CONFIG TO DATABASE ##############################

conn = psycopg2.connect(host="streetscan-test.csm0qufoemtp.us-east-2.rds.amazonaws.com",
                        database="streetscan",
                        user="sballesteros",
                        password="tatanSBR22")

cur = conn.cursor()
#cur.execute(open("model_dbs.sql", "r").read())

##INSERT INTO MODELPARAMS
query = " INSERT INTO model_params (model_id, model_name, seed_val, framework, architecture) VALUES (DEFAULT,%s,%s,%s,%s) RETURNING model_id"
record_to_insert = (config_0.model_params['model_name'],
                    config_0.model_params['seed_val'],
                    config_0.model_params['framework'],
                    config_0.model_params['architecture'] )
cur.execute(query, record_to_insert)
inserted_model_id = cur.fetchone()[0]

##INSERT INTO SYSTEM
query = " INSERT INTO system (system_id, computer_name, computer_id, processor, memory, location, activation_date) VALUES (DEFAULT,%s,%s,%s,%s,%s,CURRENT_DATE) RETURNING system_id"
record_to_insert = (config_0.system_params['computer_name'],
                    config_0.system_params['computer_id'],
                    config_0.system_params['processor'],
                    float(config_0.system_params['memory']),
                    config_0.system_params['location'] )
cur.execute(query, record_to_insert)
inserted_system_id = cur.fetchone()[0]


##INSERT INTO MODELTRAINING
query = " INSERT INTO model_training (model_id, preproc_id, system_id, run_start_date) VALUES (%s, %s, %s, CURRENT_DATE)"
record_to_insert = (inserted_model_id,
                    config_0.training_params['preproc_id'],
                    inserted_system_id)
cur.execute(query, record_to_insert)


conn.commit()
conn.close()

######################## CREATE CONFIG FOR THIS MODEL ############################

config_name = 'config_' + str(inserted_model_id) + '.py'
with open("config_0.py") as f:
    with open(config_name, "w") as f1:
        for line in f:
            f1.write(line)
