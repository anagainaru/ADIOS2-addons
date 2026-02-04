import argparse
import sqlite3
import numpy as np
import json

from hpc_campaign.info import format_info
from hpc_campaign.ls import ls
from hpc_campaign.manager import Manager
import adios2


class CampaignSqlite:
    separator = "/"
    prefix_elements = 2
    dataset_list = []

    def __init__(self, db, cursor):
        self.db = db
        self.cursor = cursor
        self.create_tables()

    def set_dataset_list(self, campaign, path):
        manager = Manager(archive=str(campaign))
        if path != None:
            manager = Manager(archive=str(campaign), campaign_store=str(path))
        info_data = manager.info(
                False, #list_replicas
                False, #list_files
                False, #show_deleted
                False #show_checksum
                )
        # info data is a InfoResult, which has datasets: list[DatasetInfo]
        # class DatasetInfo:
        # id: int
        # uuid: str
        # name: str
        # mod_time: int
        # del_time: int
        # file_format: str
        # replicas: list[ReplicaInfo] = field(default_factory=list)
        self.dataset_list = [d.name for d in info_data.datasets]

    def _extract_info(self, entry):
        #print(entry, self.dataset_list)
        for dataset in self.dataset_list:
            if entry[:len(dataset)] == dataset:
                # + 1 so we skip the separator
                # print(entry[len(dataset) + 1:], dataset)
                return dataset, entry[len(dataset) + 1:]

        # identify the file that the variable/attribute belongs to
        temp = entry.split(self.separator)
        prefix = self.prefix_elements
        if len(temp) <= prefix: # couting the entry name as well
            prefix = len(temp) - 1
        file = self.separator.join(temp[:prefix])
        value = self.separator.join(temp[prefix:])
        return file, value

    def _get_variable_data(self, variables, f):
        variable_data = {}
        attribute_data = {}
        for var in variables:
            file, var_name = self._extract_info(var)
            # json files will be open and the information inside become attributes
            if var_name.endswith(".json"):
                content = f.read(var)
                json_attributes = json.loads("".join(chr(code) for code in content))
                json_file = file+self.separator+var_name
                if json_file not in attribute_data:
                    attribute_data[json_file] = {}
                for atr in json_attributes:
                    attribute_data[json_file][atr] = json_attributes[atr]
            else: # otherwise we add variable data
                if file not in variable_data:
                    variable_data[file] = {}
                variable_data[file][var_name] = variables[var]
        return variable_data, attribute_data

    def _get_attribute_data(self, attributes, f, attribute_data={}):
        for atr in attributes:
            file, atr_name = self._extract_info(atr)
            read_value = f.read_attribute(atr)
            if file not in attribute_data:
                attribute_data[file] = {}
            if atr_name.endswith(".json"):
                # transform from np.array to json content
                variable_data[file][atr_name] = json.loads("".join(chr(code) for code in read_value))
            else:
                attribute_data[file][atr_name] = read_value
        return attribute_data

    def create_tables(self):
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS variables (
                campaign TEXT,
                dataset TEXT,
                variable TEXT,
                property TEXT
            )
        """)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS attributes (
                campaign TEXT,
                dataset TEXT,
                attribute TEXT
            )
        """)
        self.db.commit()

    def add_campaign_to_collection(self, campaign, path=None):
        print("Adding", campaign, "to the collection")
        self.set_dataset_list(campaign, path)

        file = campaign
        if path is not None:
            file = path + "/" + file

        with adios2.FileReader(file) as f:
            # add variables to the collection
            variables = f.available_variables()
            variable_data, attribute_data = self._get_variable_data(variables, f)
            for dataset in variable_data:
                for var in variable_data[dataset]:
                    value = variable_data[dataset][var]
                    self.create_variable_entry(campaign, dataset, var, value)

            # add attributes to the collection
            attributes = f.available_attributes()
            attribute_data = self._get_attribute_data(attributes, f, attribute_data)
            for dataset in attribute_data:
                for atr in attribute_data[dataset]:
                    value = attribute_data[dataset][atr]
                    # take care of the cases not supported by MongoDB
                    if isinstance(value, np.ndarray):
                        value = value.tolist()
                    if isinstance(value, np.float32):
                        value = float(value)
                    if isinstance(value, (np.uint32, np.uint64, np.uint8)):
                        value = int(value)
                    self.create_attribute_entry(campaign, dataset, {atr : value})

    def create_variable_entry(self, campaign, dataset, var, value):
        self.cursor.execute("INSERT INTO variables VALUES (?, ?, ?, ?)",
                          (campaign, dataset, var, json.dumps(value)))
        self.db.commit()

    def create_attribute_entry(self, campaign, dataset, attribute):
        self.cursor.execute("INSERT INTO attributes VALUES (?, ?, ?)",
                          (campaign, dataset, json.dumps(attribute)))
        self.db.commit()

    def print_tables(self):
        print("ATTRIBUTES")
        cursor = self.cursor.execute("SELECT * FROM attributes")
        names = [description[0] for description in cursor.description]
        print(names)
        for row in cursor:
            print(row)

        print("VARIABLES")
        cursor = self.cursor.execute("SELECT * FROM variables")
        names = [description[0] for description in cursor.description]
        print(names)
        for row in cursor:
            print(row)

    def drop_tables(self):
        self.cursor.execute("DROP TABLE IF EXISTS attributes")
        self.cursor.execute("DROP TABLE IF EXISTS variables")
        self.db.commit()


# find all campaigns files in the default campaign-store folder
def find_campaign_list(campaign_name, path):
    if path is None:
        return ls(str(campaign_name))
    return ls(str(campaign_name), campaign_store=str(path))

# python campaigns2sqlite.py campaign_names sqlite_db_file
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reading data from hpc campaigns and adding this information to a sqlite database")
    parser.add_argument('campaign_name', help="Regular expression for campaign file names")
    parser.add_argument('database', help="Name of the sqlite database file")
    parser.add_argument('--path', default=None, help="Custom path to the campaign files (default path to the campaign-store folder is used by default)")
    parser.add_argument('--remove', action='store_true', help="Remove sqlite tables")
    parser.add_argument('--print', action='store_true', help="Print sqlite tables")
    args = parser.parse_args()

    # Connect to Sqlite
    db = sqlite3.connect(args.database)
    cursor = db.cursor()

    # find all campaign files that match the pattern
    campaign_files = find_campaign_list(args.campaign_name, args.path)
    cm = CampaignSqlite(db, cursor)

    if args.remove == True:
        cm.drop_tables()

    for campaign in campaign_files:
        cm.add_campaign_to_collection(campaign, path=args.path)

    if args.print == True:
        cm.print_tables()

    db.close()
