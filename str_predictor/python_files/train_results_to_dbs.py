import os
import psycopg2

def write_to_dbs(loss, mean_iou, val_loss, val_mean_iou, model_id):
	print(f"model id: {model_id}")
	###################### WRITE CONFIG TO DATABASE ##############################

	conn = psycopg2.connect(host="streetscan-test.csm0qufoemtp.us-east-2.rds.amazonaws.com",
       	                	database="streetscan",
                        	user="sballesteros",
                        	password="tatanSBR22")

	cur = conn.cursor()

	"""
	query = 'ALTER TABLE model_training ADD COLUMN loss float[]'
	cur.execute(query)
	query = 'ALTER TABLE model_training ADD COLUMN mean_iou float[]'
	cur.execute(query)
	query = 'ALTER TABLE model_training ADD COLUMN val_loss float[]'
	cur.execute(query)
	query = 'ALTER TABLE model_training ADD COLUMN val_mean_iou float[]'
	cur.execute(query)
	
	query = '''SELECT * FROM model_training'''
	cur.execute(query)
	print(cur.fetchall())

	conn.commit()

	"""

	##INSERT INTO MODEL_TRAINING
	query = " UPDATE model_training SET loss=%s, mean_iou=%s, val_loss=%s, val_mean_iou=%s WHERE model_id=%s"
	record_to_insert = (loss,
			mean_iou,
			val_loss,
			val_mean_iou,
			model_id)
	cur.execute(query, record_to_insert)

	
	query = '''SELECT * FROM model_training WHERE model_id=%s'''
	cur.execute(query, (model_id, ))
	print("Model updated:")
	print(cur.fetchall())
	conn.commit()
	conn.close()


