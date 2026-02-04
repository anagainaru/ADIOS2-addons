import argparse
import sqlite3
import adios2
import json


class CampaignQuerySqlite:

    def __init__(self, db, cursor, groupby="campaign"):
        self.db = db
        self.cursor = cursor

        self.groupby = [groupby]
        if groupby == "both":
            self.groupby = ["campaign", "dataset"]


    def query_attributes(self, keyword, value):
        # Find all outputs where the attribute keywork == value
        query = f"""
            SELECT DISTINCT campaign, dataset
            FROM attributes
            WHERE json_extract(attribute, '$.{keyword}') = ?
        """
        self.cursor.execute(query, (value,))
        return self.cursor.fetchall()

    def query_variables(self, keyword, value):
        # Find all outputs where the variable keywork has value between Min/Max
        query = """
            SELECT DISTINCT campaign, dataset
            FROM variables
            WHERE variable = ?
              AND CAST(json_extract(property, '$.Min') AS REAL) <= ?
              AND CAST(json_extract(property, '$.Max') AS REAL) >= ?
        """
        self.cursor.execute(query, (keyword, value, value))
        return self.cursor.fetchall()

    def query_collection(self, variables, keyword=None, value=None):
        # if there is no keyword return all campaigns and datasets that contain the desired variables
        if keyword is None or value is None:
            query = "SELECT campaign, dataset, variable FROM variables WHERE variable IN ({seq})".format(seq=','.join(['?']*len(variables)))
            self.cursor.execute(query, variables)
            result = self.cursor.fetchall()
            return self.aggregate_query_result(result)

        try:
            processed_value = float(value)
            result_tuple_ids = self.query_variables(keyword, processed_value)
        except ValueError:
            processed_value = value
            result_tuple_ids = self.query_attributes(keyword, processed_value)


        if not result_tuple_ids:
            print("No campaigns found to match the query.")
            return {}

        eligible_campaign_ids = [i[0] for i in result_tuple_ids]
        eligible_dataset_ids = [i[1] for i in result_tuple_ids]

        # Construct the final query to get variables from the selected campaigns/datasets
        final_query = "SELECT campaign, dataset, variable FROM variables WHERE variable IN ({seq})".format(seq=','.join(['?']*len(variables)))
        final_query_params = list(variables)

        if "campaign" in self.groupby:
            final_query += " AND campaign IN ({seq})".format(seq=','.join(['?']*len(eligible_campaign_ids)))
            final_query_params.extend(eligible_campaign_ids)
        if "dataset" in self.groupby:
            final_query += " AND dataset IN ({seq})".format(seq=','.join(['?']*len(eligible_dataset_ids)))
            final_query_params.extend(eligible_dataset_ids)

        self.cursor.execute(final_query, final_query_params)
        result = self.cursor.fetchall()
        return self.aggregate_query_result(result)

    # input = [('campaign1', 'dataset1', 'variable1'), ('campaign1', 'dataset1', 'variable2')]
    # output[campaign][dataset] = list of variables
    def aggregate_query_result(self, entries):
        result_dict = {}
        for campaign, dataset, variable in entries:
            dataset_dict = result_dict.setdefault(campaign, {})
            variable_list = dataset_dict.setdefault(dataset, [])
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
    parser = argparse.ArgumentParser(description="Query a sqlite database, get read_variables from all campaigns and datasets that contain keyword equal to value")
    parser.add_argument('database', help="Name of the sqlite database file")
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

    # Connect to Sqlite
    db = sqlite3.connect(args.database)
    cursor = db.cursor()

    cm = CampaignQuerySqlite(db, cursor, args.group_by)

    entries = cm.query_collection(args.read_variables, args.keyword, args.value)
    read_from_campaigns(entries)
    db.close()
