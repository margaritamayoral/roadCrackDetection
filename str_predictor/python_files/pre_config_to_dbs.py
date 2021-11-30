import os
import psycopg2
import pre_config

###################### WRITE CONFIG TO DATABASE ##############################

conn = psycopg2.connect(
    host="streetscan-test.csm0qufoemtp.us-east-2.rds.amazonaws.com",
    database="streetscan",
    user="sballesteros",
    password="tatanSBR22")

cur = conn.cursor()
#cur.execute(open("model_dbs.sql", "r").read())


query = " INSERT INTO preprocess (preproc_id, postproc_id, class_ids, num_classes, train_data_ids, test_data_ids, test_train_split, num_test_imgs, num_train_imgs, seed_val) VALUES (DEFAULT,DEFAULT,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING preproc_id"
record_to_insert = (pre_config.preprocess['class_ids'],
                    pre_config.preprocess['num_classes'],
                    pre_config.preprocess['train_data_ids'],
                    pre_config.preprocess['test_data_ids'],
                    pre_config.preprocess['test_train_split'],
                    pre_config.preprocess['num_test_imgs'],
                    pre_config.preprocess['num_train_imgs'],
                    pre_config.preprocess['seed_val'])
cur.execute(query, record_to_insert)

inserted_preproc_id = cur.fetchone()[0]

conn.commit()
conn.close()

######################## CREATE CONFIG FOR THIS MODEL ############################

config_name = 'pre_config_' + str(inserted_preproc_id) + '.py'
with open("pre_config.py") as f:
    with open(config_name, "w") as f1:
        for line in f:
            f1.write(line)
