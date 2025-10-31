## Queries over campaigns

Queries over ADIOS variables inside campaigns can be done in the same way ADIOS bp files can be queried.
The campaign API has a way to find all campaigns in the default folder that contains a string:

```python
campaigns = hpc_campaign.List("partial_name")
# returns ["partial_name_001", "another_partial_name", "partial_name_002", ...]
```

The Campaign engine in ADIOS can read campaign files in the default folder without a path being provided.

**Python example script for querying campaign files**

```
usage: query_campaigns.py [-h] --read_variables [READ_VARIABLES ...] [--keyword KEYWORD] [--value VALUE] [--path PATH] name

Reading data from hpc campaigns based on attribute/value matching.

positional arguments:
  name                  Partial name of the campaign file

options:
  -h, --help            show this help message and exit
  --read_variables [READ_VARIABLES ...]
                        List of variable to extract
  --keyword KEYWORD     The variable/attribute used for the query condition (optional)
  --value VALUE         The value requested for the keyword attribute/variable (optional)
  --path PATH           Path to the campaign files (optional)
```

The name (or partial name) of a campaign is needed (`*` will include all aca files) as well as the variables that will be read from the files that fit the query.
The default folder is the campaign-store path provided in the config.yaml file associated to the hpc-campaign software.

At the minimum the script needs to be executed in the following way:

```
$ python query_campaigns.py picongpu_laptop --read_variables derive/derived/energy
 - Inspecing campaigns with names containing: picongpu_laptop
   - Located in the default campaign folder
 - Read variables/attributes: ['derive/derived/energy']
2 campaigns found
Reading picongpu_laptop_02.aca
READ run03/derive/derived/energy
Reading picongpu_laptop_01.aca
READ run01/derive/derived/energy run02/derive/derived/energy
```
This will parse all campaign files containing `picongpu_laptop` (in this case 2 campaign files found) and reads all variables called `derive/derived/energy` within those campaigns.
Three variables are found in the three runs (run 1 and 2 from campaign 1 and run 3 from campaign 2).

A custom `path` can be provided for where campaigns are stored.

```
$ python query_campaigns.py picongpu_laptop --read_variables derive/derived/energy --path /new/path/
 - Inspecing campaigns with names containing: picongpu_laptop
   - Located in the default campaign folder
 - Read variables/attributes: ['derive/derived/energy']
No campaign files found
```

If a keyword is provided, only the files that contain the attribute given by the keyword will be considered for reading.
E.g. if the third run would not contain attribute `test` and the other two runs contain it, the example above will only read the first two runs.
If the value is also provided, an additional filtering is imposed.
For now the script compares (as strings) the value stored in the attributes that contain the keywork with the value provided by the value entry and looks for the desired read variable(s) only for those files that fit this query.

```
$ python query_campaigns.py picongpu_laptop --read_variables derive/derived/energy derive/derived/poynting --keyword input/customuserinput/minimum_weight  --value 0.04
 - Inspecing campaigns with names containing: picongpu_laptop
   - Located in the default campaign folder
 - Read variables/attributes: ['derive/derived/energy', 'derive/derived/poynting']
   - For files containing attribute/variable: input/customuserinput/minimum_weight
   - With the keywork containing value: 0.04
2 campaigns found
Reading picongpu_laptop_02.aca
Reading picongpu_laptop_01.aca
READ run02/derive/derived/energy
READ run02/derive/derived/poynting

$ python query_campaigns.py picongpu_laptop --read_variables derive/derived/energy derive/derived/poynting --keyword input/customuserinput/minimum_weight   --value 0.03
 - Inspecing campaigns with names containing: picongpu_laptop
   - Located in the default campaign folder
 - Read variables/attributes: ['derive/derived/energy', 'derive/derived/poynting']
   - For files containing attribute/variable: input/customuserinput/minimum_weight
   - With the keywork containing value: 0.03
2 campaigns found
Reading picongpu_laptop_02.aca
READ run03/derive/derived/energy
READ run03/derive/derived/poynting
Reading picongpu_laptop_01.aca
READ run01/derive/derived/energy
READ run01/derive/derived/poynting
```

The first and third run are using `input/customuserinput/minimum_weight` of 0.03 while the second is using 0.04

## Using MongoDB

Installing MongoDB
```
brew tap mongodb/brew
brew install mongodb-community
brew services start mongodb-community
pip install pymongo
```

Assumptions on how campaigns are created
- All data will one campaign uses the same input configuration
- One campaign can have multiple outputs for one or multiple runs
- Multiple campaigns can have the same input configuration

The `campaign2mongodb.py` can be used to parse campaigns in a path and add all variables to a MongoDB collection.
This collection can be queried using the `query_mongodb.py` script.

### Campaign to MongoDB

The script works on regular expressions for campaign names (or partial names).
- If no path is selected, the default path to the campaign-store is used.
- If the remove option is not used, the campaigns are concatenated to existing collections. 
- The print option, prints the entire collection (and not just the entries related to the campaign)

```
usage: campaign2mongodb.py [-h] [--path PATH] [--remove] [--print] campaign_name client database collection

Reading data from hpc campaigns and adding this information to a mongodb collection

positional arguments:
  campaign_name  Regular expression for campaign file names
  client         MongoDB client daemon
  database       Name of the MongoDB database
  collection     Name of the MongoDB collection in the database

options:
  -h, --help     show this help message and exit
  --path PATH    Custom path to the campaign files (default path to the campaign-store folder is used by default)
  --remove       Remove mongoDB collection
  --print        Print mongoDB collection
```

Example:
```
python campaign2mongodb.py picongpu_laptop mongodb://localhost:27017/ PIConGPU_db laptop --remove
Adding picongpu_laptop_03.aca to the collection
Adding picongpu_laptop_02.aca to the collection
Adding picongpu_laptop_01.aca to the collection
```

Running the script, two collections are created, one with variables and one with attributes.
The entries in the attributes collection use the keywords: campaign, dataset, attribute ({name:value}) and the entries in the variables collection use campaign, dataset, variable and property.

Attribute keywords in the attributes collection can be dictionaries with sub-attributes:
```json
{‘_id’: ObjectId(‘..’), ‘campaign’: ‘picongpu_laptop_02.aca’, ‘dataset’: ‘input_files/pypicongpu2.json’, ‘attribute’: {‘moving_window’: {‘move_point’: 0.98, ‘stop_iteration’: None}}}
```

Variable keywods have the name of the variable and the property have the Min, Max and other metadata associated to variables.
```json
{'_id': ObjectId('...'), 'campaign': 'picongpu_laptop_02.aca', 'dataset': 'run03/output', 'variable': 'data/0/particles/electrons/weighting', 'property': {'AvailableStepsCount': '1', 'Max': '0', 'Min': '0', 'Shape': '21780', 'SingleValue': 'false', 'Type': 'double'}}
```

### Query campaigns using MongoDB

To query the mongoDB collections and the campaigns sored inside, the `query_mongodb.py` script.
- The script reads variables (not attributes)
- The query can be applied on attributes (checking for a specific value)
- The query can also be applied on variables (the value is in between the min and max of the variable)

```
usage: query_mongodb.py [-h] --read_variables [READ_VARIABLES ...] [--keyword KEYWORD] [--value VALUE] [--group-by {campaign,dataset,both}] client database collection

Query a MongoDB collection, get read_variables from all campaigns and datasets that contain keyword equal to value

positional arguments:
  client                MongoClient daemon
  database              Name of the atabase
  collection            Name of the collection in the database

options:
  -h, --help            show this help message and exit
  --read_variables [READ_VARIABLES ...]
                        List of variable to extract
  --keyword KEYWORD     The variable/attribute used for the query condition (optional)
  --value VALUE         The value requested for the keyword attribute/variable (optional)
  --group-by {campaign,dataset,both}
                        Find variables in datasets based on queries grouped by campaign/datasets or both
```

To quey the database we can specify only the variable names in which case all datasets containing that variables will be selected or can take a keyword (name of the attribute or variable) and value.
In the example `data/0/fields/B/x` is a variable and `customuserinput.minimum_weight` is an attribute.

```bash
$ python query_mongodb.py mongodb://localhost:27017/ PIConGPU_db laptop --read_variables data/0/fields/B/x data/0/fields/B/y
Reading picongpu_laptop_03.aca
READ run04/output/data/0/fields/B/x shape: [256, 32, 32]
READ run04/output/data/0/fields/B/y shape: [256, 32, 32]
Reading picongpu_laptop_02.aca
READ run03/output/data/0/fields/B/x shape: [256, 32, 32]
READ run03/output/data/0/fields/B/y shape: [256, 32, 32]
Reading picongpu_laptop_01.aca
READ run01/output/data/0/fields/B/x shape: [256, 32, 32]
READ run01/output/data/0/fields/B/y shape: [256, 32, 32]
READ run02/output/data/0/fields/B/x shape: [256, 32, 32]
READ run02/output/data/0/fields/B/y shape: [256, 32, 32]

$ python query_mongodb.py mongodb://localhost:27017/ PIConGPU_db laptop --read_variables data/0/fields/B/x data/0/fields/B/y --keyword customuserinput.minimum_weight --value 0.01
No campaigns found to match the query.

$ python query_mongodb.py mongodb://localhost:27017/ PIConGPU_db laptop --read_variables data/0/fields/B/x data/0/fields/B/y --keyword customuserinput.minimum_weight --value 0.04
Reading picongpu_laptop_03.aca
READ run04/output/data/0/fields/B/x shape: [256, 32, 32]
READ run04/output/data/0/fields/B/y shape: [256, 32, 32]
Reading picongpu_laptop_02.aca
READ run03/output/data/0/fields/B/x shape: [256, 32, 32]
READ run03/output/data/0/fields/B/y shape: [256, 32, 32]
```

We can also query based on variable properties, e.g. if the value provided is between the Min and Max of the variable with the name given by the keyword.

```
$ python query_mongodb.py mongodb://localhost:27017/ PIConGPU_db laptop --read_variables data/0/fields/B/x data/0/fields/B/y --keyword data/0/particles/electrons/position/y --value 0 --group-by both
Reading picongpu_laptop_01.aca
READ run01/output/data/0/fields/B/x shape: [256, 32, 32]
READ run01/output/data/0/fields/B/y shape: [256, 32, 32]
READ run02/output/data/0/fields/B/x shape: [256, 32, 32]
READ run02/output/data/0/fields/B/y shape: [256, 32, 32]
```
