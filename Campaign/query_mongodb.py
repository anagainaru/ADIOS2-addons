import argparse
import pymongo
import adios2


class CampaignQueryDB:

    def __init__(self, collectionAtr, collectionVar, groupby="campaign"):
        self.collectionVar = collectionVar
        self.collectionAtr = collectionAtr

        self.groupby = [groupby]
        if groupby == "both":
            self.groupby = ["campaign", "dataset"]


    def _process_value(self, value):
        # Try to convert to an int, then a float, otherwise keep as string
        try:
            processed_value = int(value)
        except ValueError:
            try:
                processed_value = float(value)
            except ValueError:
                processed_value = value # Keep as string if conversion fails
        return processed_value

    def query_attributes(self, keyword, value):
        # Find all outputs where the attribute keywork == value
        # {'_id': ObjectId('..'), 'campaign': 'picongpu_laptop_02.aca', 'dataset': 'input_files/pypicongpu2.json', 'attribute': {'moving_window': {'move_point': 0.98, 'stop_iteration': None}}}
        pipeline = [
            {
                "$match": {
                    "attribute."+keyword: value
                }
            },
            # Group the results to extract unique campaign/dataset names
            {
                "$group": {
                     "_id": ["$campaign", "$dataset"]
                }
            }
        ]
        # Execute the pipeline to get the list of campaign and dataset IDs
        result = self.collectionAtr.aggregate(pipeline)
        result_tuple_ids = [doc['_id'] for doc in result]
        return result_tuple_ids

    def query_variables(self, keyword, value):
        # Find all outputs where the variable keywork has value between Min/Max
        # {'_id': ObjectId('...'), 'campaign': 'picongpu_laptop_02.aca', 'dataset': 'run03/output', 'variable': 'data/0/particles/electrons/weighting', 'property': {'AvailableStepsCount': '1', 'Max': '0', 'Min': '0', 'Shape': '21780', 'SingleValue': 'false', 'Type': 'double'}}
        pipeline = [
            {
                "$match": {
                    'variable' : keyword,
                    "$expr": {
                        "$and": [
                            { "$gte": [ { "$toDouble": "$property.Max" }, value ] },
                            { "$lte": [ { "$toDouble": "$property.Min" }, value] }
                        ]
                    }
                }
            },
            # Group the results to extract unique campaign/dataset names
            {
                "$group": {
                     "_id": ["$campaign", "$dataset"]
                }
            }
        ]
        # Execute the pipeline to get the list of campaign and dataset IDs
        result = self.collectionVar.aggregate(pipeline)
        result_tuple_ids = [doc['_id'] for doc in result]
        return result_tuple_ids

    def query_collection(self, variables, keyword=None, value=None):
        # if there is no keyword return all campaigns and datasets that contain the desired variables 
        if keyword is None or value is None:
            result = collectionVar.find({'variable': { '$in': variables }},
                                        {"campaign": 1, "dataset": 1, "variable": 1, "_id": 0})
            return self.aggregate_query_result(result)

        processed_value = self._process_value(value)
        result_tuple_ids = self.query_attributes(keyword, processed_value)
        # no attribute fits the query
        if not result_tuple_ids:
            # look for variables that fit the query (assuming we can compare Min/Mx to the value)
            if not isinstance(processed_value, str):
                result_tuple_ids = self.query_variables(keyword, processed_value)

        if not result_tuple_ids:
            print("No campaigns found to match the query.")
            return {}

        eligible_campaign_ids = [i[0] for i in result_tuple_ids]
        eligible_dataset_ids = [i[1] for i in result_tuple_ids]

        # Construct the final query to get variables from the selected campaigns/datasets
        final_query = {
            "variable": { '$in': variables },
        }
        if "campaign" in self.groupby:
            final_query["campaign"] = { "$in": eligible_campaign_ids }
        if "dataset" in self.groupby:
            final_query["dataset"] = { "$in": eligible_dataset_ids }
        result = collectionVar.find(final_query, {"campaign": 1, "dataset": 1, "variable": 1, "_id": 0})
        return self.aggregate_query_result(result)

    # input = [{'campaign': 'picongpu_laptop_03.aca', 'dataset': 'run04/output', 'variable': 'data/0/fields/B/x'}]
    # output[campaign][dataset] = list of variables
    def aggregate_query_result(self, entries):
        result_dict = {}
        for item in entries:
            campaign = item['campaign']
            dataset = item['dataset']
            variable = item['variable']

            # 1. Get or create the inner dictionary for the campaign
            # result_dict.setdefault(campaign, {}) returns the dictionary for the campaign
            # or creates it as {} if it doesn't exist.
            dataset_dict = result_dict.setdefault(campaign, {})

            # 2. Get or create the list for the dataset within that campaign
            # dataset_dict.setdefault(dataset, []) returns the list for the dataset
            # or creates it as [] if it doesn't exist.
            variable_list = dataset_dict.setdefault(dataset, [])

            # 3. Append the variable to the list
            variable_list.append(variable)
        return result_dict

# read variable from partial name from all campaign files selected
# results[campaign][dataset] = [variable list]
def read_from_campaigns(results):
    print(results)
    for campaign in results:
        print("From", campaign)
        files = results[campaign]
        datasetnames = ",".join([dataset for dataset in files])
        adios = adios2.Adios()
        io = adios.declare_io("BPWriter")
        io.set_parameter("include-dataset", datasetnames)
        with adios2.FileReader(io, campaign) as f:
            variable_list = [f+"/"+var for f in files for var in files[f]]
            for var in variable_list:
                data = f.inquire_variable(var)
                print("READ", var, "shape:", data.shape())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query a MongoDB collection, get read_variables from all campaigns and datasets that contain keyword equal to value")
    parser.add_argument('client', help="MongoClient daemon")
    parser.add_argument('database', help="Name of the atabase")
    parser.add_argument('collection', help="Name of the collection in the database")
    parser.add_argument('--read_variables', required=True, nargs='*', default=None, help="List of variable to extract")
    parser.add_argument('--keyword', default=None, help="The variable/attribute used for the query condition (optional)")
    parser.add_argument('--value', default=None, help="The value requested for the keyword attribute/variable (optional)")
    parser.add_argument('--group-by', default='campaign', choices=['campaign', 'dataset', 'both'],
                        help="Find variables in datasets based on queries grouped by campaign/datasets or both")
    args = parser.parse_args()

    if args.keyword == None and args.value != None:
        print("If value is provided, the keyword attribute/variable needs to be provided too")
        exit(1)

    if args.keyword != None and args.value == None:
        print("If the keyword is provided, the value needs to be provided too")
        exit(1)

    # Connect to MongoDB
    client = pymongo.MongoClient(args.client)
    db = client[args.database]
    collectionAtr = db[args.collection+"Atr"]
    collectionVar = db[args.collection+"Var"]

    cm = CampaignQueryDB(collectionAtr, collectionVar, args.group_by)

    entries = cm.query_collection(args.read_variables, args.keyword, args.value)
    read_from_campaigns(entries)
