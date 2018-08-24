import json
import os
from config import JSON_PATH, DATA_PATH

class CSVUtility():
    """This class is for preprocessing the input data for feeding to the model"""
    def __init__(self, input_df, department_column, doc_id_column):
        self.input_df = input_df
        self.department_column = department_column
        self.doc_id_column = doc_id_column


    def generate_processed_csv(self):
        print ("Initial Shape of document_csv: %s" %str(self.input_df.shape))
        print ("Department Counts: \n")
        print (self.input_df[self.department_column].value_counts())

        # Dropping Dublicates
        self.input_df = self.input_df.drop_duplicates()

        print ("\nShape of document_csv after dropping dublicates: %s" % str(self.input_df.shape))

        for i, row in self.input_df.iterrows():
            json_payload = json.load(open(JSON_PATH + "/" + str(row[self.doc_id_column]) + '.json', 'rb'))
            description = json_payload['jd_information']['description']
            self.input_df.at[i, 'description'] = str(description)

        save_path = os.path.join(DATA_PATH, 'processed_csv')

        self.input_df.to_csv(save_path)

        print("\nProcessed CSV created at %s" %save_path)

        return self.input_df