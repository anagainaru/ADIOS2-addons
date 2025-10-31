import argparse
import numpy as np
import glob
import hpc_campaign
import adios2

# find all campaigns files in a given path
def find_campaigns_at_path(campaign_name, path):
    if campaign_name[-3:] != "aca":
        if campaign_name[:-1] != "*":
            campaign_name += "*"
        campaign_name += ".aca"
    if campaign_name[0] != "*":
        campaign_name = "*" + campaign_name
    return glob.glob(path+"/"+campaign_name)

# find all campaigns files in the default campaign-store folder
def find_campaigns_default_path(campaign_name):
    return hpc_campaign.List(campaign_name)

# read variable list from all campaign files selected
def read_from_all_campaigns(campaign_files, read_variables):
    for campaign in campaign_files:
        print("Reading", campaign)
        with adios2.FileReader(campaign) as f:
            variable_list = f.available_variables()
            for var in read_variables:
                print("READ", " ".join([v for v in variable_list if v.endswith(var)]))

def query_satisfied(read_value, value):
    return value == str(read_value)

# read variable list from the campaign files that have the given keywork attribute
# (and optionally check that the attribute has a predefined value)
def read_campaign_files(campaign_files, keyword, value, read_variables):
    for campaign in campaign_files:
        print("Reading", campaign)
        with adios2.FileReader(campaign) as f:
            attributes = f.available_attributes()
            # select all entries containing the keyword 
            attribute_list = [a for a in attributes if a.endswith(keyword)]
            selected_attributes = []
            # if a value is selected, check the value for all the attributes
            if value != None:
                for attribute in attribute_list:
                    read_value = f.read_attribute(attribute)
                    if query_satisfied(read_value, value):
                        selected_attributes.append(attribute)
            else:
                selected_attributes = attribute_list

            # extract the name of the dataset in the selected attribute list
            # e.g. /run01/slurm002/pressure/unit for keyword pressure/unit will return /run01/slurm002/
            unique_datasets = set([var[: -len(keyword)] for var in selected_attributes])
            for var in read_variables:
                # check variable list for the identified datasets to make sure the required one exists
                variable_list = f.available_variables()
                read_vars = [dataset+var for dataset in unique_datasets if dataset+var in variable_list]
                if len(read_vars) > 0:
                    print("READ", " ".join(read_vars))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reading data from hpc campaigns based on attribute/value matching.")
    parser.add_argument('name', help="Partial name of the campaign file")
    parser.add_argument('--read_variables', required=True, nargs='*', default=None, help="List of variable to extract")
    parser.add_argument('--keyword', default=None, help="The variable/attribute used for the query condition (optional)")
    parser.add_argument('--value', default=None, help="The value requested for the keyword attribute/variable (optional)")
    parser.add_argument('--path', default=None, help="Path to the campaign files (optional)")
    args = parser.parse_args()

    if args.keyword == None and args.value != None:
        print("If value is provided, the keyword attribute/variable needs to be provided too")
        exit(1)

    print(" - Inspecing campaigns with names containing:", args.name)
    # find all campaign files that match the pattern
    campaign_files = []
    if args.path != None:
        print("   - Located in folder:", args.path)
        campaign_files = find_campaigns_at_path(args.name, args.path)
    else:
        print("   - Located in the default campaign folder")
        campaign_files = find_campaigns_default_path(args.name)

    print(" - Read variables/attributes:", args.read_variables)
    if args.keyword != None:
        print("   - For files containing attribute/variable:", args.keyword)
    if args.value != None:
        print("   - With the keywork containing value:", args.value)

    if len(campaign_files) == 0:
        print("No campaign files found")
        exit(1)
    print(len(campaign_files), "campaigns found")
    
    if args.keyword == None:
        # read all datasets that contain the variables of interest
        read_from_all_campaigns(campaign_files, args.read_variables)
    else:
        # filter the file list to those that contain the keyword attribute
        # and the keyword has the specified value
        read_campaign_files(campaign_files, args.keyword, args.value, args.read_variables)
