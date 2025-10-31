import argparse
import pymongo
import numpy as np
import json

import hpc_campaign
import adios2


class CampaignMongoDB:
    separator = "/"
    prefix_elements = 2

    def __init__(self, collectionAtr, collectionVar):
        self.collectionVar = collectionVar
        self.collectionAtr = collectionAtr

    def _extract_info(self, entry):
        # identify the file that the variable/attribute belongs to
        temp = entry.split(self.separator)
        prefix = self.prefix_elements
        if len(temp) <= prefix: # couting the entry name as well
            prefix = len(temp) - 1
        file = self.separator.join(temp[:prefix])
        value = self.separator.join(temp[prefix:])
        return file, value

    # the function returns a list of variables with their metadata
    # as well as a list of attributes in case there are json files saved as variables
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

    # read variable list from the campaign files and add them in the mongoDB collection
    # for json files, parst input configuration and add them in the collection
    def add_campaign_to_collection(self, campaign):
        print("Adding", campaign, "to the collection")
        path = ""
        if args.path is not None:
            path = self.path + separator

        with adios2.FileReader(campaign) as f:
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
        doc = {
            "campaign": campaign,
            "dataset": dataset,
            "variable": var,
            "property": value,
        }
        result = collectionVar.insert_one(doc)
        return result.inserted_id

    def create_attribute_entry(self, campaign, dataset, attribute):
        doc = {
            "campaign": campaign,
            "dataset": dataset,
            "attribute": attribute
        }
        result = collectionAtr.insert_one(doc)
        return result.inserted_id

    def print_collections(self):
        print("ATTRIBUTES")
        for doc in collectionAtr.find():
            print(doc)
        print("VARIABLES")
        for doc in collectionVar.find():
            print(doc)

    def drop_collections(self):
        collectionAtr.drop()
        collectionVar.drop()


# find all campaigns files in the default campaign-store folder
def find_campaign_list(campaign_name, path):
    if path is None:
        return hpc_campaign.List(campaign_name)
    return hpc_campaign.List(campaign_name, path)

# python campaign2mongodb.py campaign_names mongodb_client mongodb_db mongodb_collection
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reading data from hpc campaigns and adding this information to a mongodb collection")
    parser.add_argument('campaign_name', help="Regular expression for campaign file names")
    parser.add_argument('client', help="MongoDB client daemon")
    parser.add_argument('database', help="Name of the MongoDB database")
    parser.add_argument('collection', help="Name of the MongoDB collection in the database")
    parser.add_argument('--path', default=None, help="Custom path to the campaign files (default path to the campaign-store folder is used by default)")
    parser.add_argument('--remove', action='store_true', help="Remove mongoDB collection")
    parser.add_argument('--print', action='store_true', help="Print mongoDB collection")
    args = parser.parse_args()

    # Connect to MongoDB
    client = pymongo.MongoClient(args.client)
    db = client[args.database]
    collectionAtr = db[args.collection+"Atr"]
    collectionVar = db[args.collection+"Var"]

    # find all campaign files that match the pattern
    campaign_files = find_campaign_list(args.campaign_name, args.path)
    cm = CampaignMongoDB(collectionAtr, collectionVar)

    if args.remove == True:
        cm.drop_collections()

    for campaign in campaign_files:
        cm.add_campaign_to_collection(campaign)

    if args.print == True:
        cm.print_collections()
