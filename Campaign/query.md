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
$ python query_attribute_read_variable.py -h
usage: query_attribute_read_variable.py [-h] --read_variables [READ_VARIABLES ...] [--keyword KEYWORD] [--value VALUE] [--path PATH] name

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
$ python query_attribute_read_variable.py picongpu_laptop --read_variables derive/derived/energy
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
$ python query_attribute_read_variable.py picongpu_laptop --read_variables derive/derived/energy --path /new/path/
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
$ python query_attribute_read_variable.py picongpu_laptop --read_variables derive/derived/energy derive/derived/poynting --keyword input/customuserinput/minimum_weight  --value 0.04
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

$ python query_attribute_read_variable.py picongpu_laptop --read_variables derive/derived/energy derive/derived/poynting --keyword input/customuserinput/minimum_weight   --value 0.03
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
