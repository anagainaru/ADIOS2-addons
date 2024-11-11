# Campaign management system

The campaign management includes:
- Access to variables across multiple runs
- Remote access
- Setting the accuracy of a read

## ADIOS build

For now, the ADIOS branch that has accuracy enabled is in Norbert's repo:
https://github.com/pnorbert/ADIOS2/tree/remote_compression

**Dependencies:**
 - ZFP 1.0.0 https://github.com/LLNL/zfp.git, tag 1.0.1
 - SQLite 3  (libsqlite3-dev on Ubuntu)

On MAC that had python both through Homebrew and external, use a virtual environment to install a Python library that isn't in Homebrew:
```
$ python3 -m venv ~/work/kits/pipve
$ source  ~/work/kits/pipve/bin/activate
```

The following pip3 packages may need to be installed: paramiko, python-dateutil, python-yaml, sqlite3, python-tk@3.13 and dataclasses

```
$ pip3 install pyyaml
Successfully installed pyyaml-6.0.2
$ pip3 install python-dateutil
Successfully installed python-dateutil-2.9.0.post0 six-1.16.0
$ brew install python-tk@3.13
$ pip3 install paramiko
Successfully installed bcrypt-4.2.0 cffi-1.17.1 cryptography-43.0.3 paramiko-3.5.0 pycparser-2.22 pynacl-1.5.0
```

ZFP was build with default parameters. ADIOS2 build uses the following options:
```
    -D ADIOS2_USE_ZFP=ON
    -D ADIOS2_USE_Campaign=ON
    -D ADIOS2_USE_Derived_Variable=ON
    -D ADIOS2_BUILD_EXAMPLES=ON
    -D ADIOS2_USE_PYTHON=ON
    -D PYTHON_EXECUTABLE=/usr/local/bin/python3
    -D Python_FIND_STRATEGY=LOCATION
    -D BUILD_TESTING=OFF 
```

## Campaign manager and connector
 
Repo: https://github.com/ornladios/hpc-campaign

The campaign manager requires two configuration files: `hosts.yaml` and `adios2.yaml`
 
Create the `~/.config/adios2/hosts.yaml` file:
```
# Host configurations for accessing remote data in Adios Campaign Archives (.aca)
# Top level dictionary: name should be the same as the hostnames used across the projects to identify hosts
# For each host, define a list of access methods
# An access method is a dictionary with elements depending on the protocol
 
 
OLCF:
  dtn-ssh:
      protocol: ssh
      host: dtn.olcf.ornl.gov
      user: YOURUSERNAME
      authentication: passcode
      serverpath: ~pnorbert/dtn/sw/adios2/bin/adios2_remote_server -background -report_port_selection -v -v -l /ccs/home/YOURUSERNAME/log.adios2_remote_server
      verbose: 1
```

Create the `~/.config/adios2/adios2.yaml` file:
```
Campaign:
  active: true
  hostname: LAP131864
  campaignstorepath: ~/dropbox/adios-campaign-store
  cachepath: /tmp/campaign
  verbose: 0
```

The hostname, campaignstorepath and cachepath need to be changed for each system.
 
### Campaign files

Campaign files need to be stored into the `campaignstorepath` path. 
You can use subdirectories to organize campaigns but not required.

```
$ python3 hpc_campaign_manager/hpc_campaign_manager.py info frontier_s3d_001.aca
info archive
ADIOS Campaign Archive, version 0.1, created on 2024-08-26 12:42:05.855606
hostname = OLCF   longhostname = frontier.olcf.ornl.gov
    dir = /lustre/orion/csc143/proj-shared/s3d/demo_single_step/run
        dataset = ../data/ptj.field.bp     created on 2024-08-13 09:11:29
```

If subdirectories are used, the path from `campaignstorepath` should be provided.
```
e.g.
$ python3 ~/Software/hpc-campaign/source/hpc_campaign_manager/hpc_campaign_manager.py  info csc143/demoproject/frontier_s3d_001.aca
$ bpls -l csc143/demoproject/frontier_s3d_001.aca
$ bpls -l csc143/gray-scott/ensemble1/run-488.aca
```

The connector is not needed for just inspecting the metadata (variable info and location).
 
### Connector

The connector needs to be ran before reading the remote data:
```
hpc-campaign/source/hpc_campaign_connector$ python3 ./hpc_campaign_connector.py -c ~/.config/adios2/hosts.yaml -p  30000
 
SSH Tunnel Server:  127.0.0.1 30000
```

The remote server will timeout in an hour, but the connecter script will not know it. After one hour the python local instance needs to be killed.
 
 
### Testing remote access

ADIOS2 can be ran as if the data was local.

Example using bpls
```
$ bpls -l run-488.aca -d gs.bp/U -s "99,62,62,62" -c "1,4,4,4" -n 4
  double   gs.bp/U        200*{128, 128, 128} = 0.0813067 / 1
    slice (99:99, 62:65, 62:65, 62:65)
BP5Reader::PerformRemoteGets: send out requests
BP5Reader::PerformRemoteGets: wait for responses
ReadResponseHandler: response size = 78 operator type [7]
BP5Reader::PerformRemoteGets: done with one response
BP5Reader::PerformRemoteGets: completed
    (99,62,62,62)    0.892578 0.900391 0.900391 0.892578
    (99,62,63,62)    0.900391 0.908203 0.908203 0.900391
    (99,62,64,62)    0.900391 0.908203 0.908203 0.900391
    (99,62,65,62)    0.892578 0.900391 0.900391 0.892578
    (99,63,62,62)    0.900391 0.908203 0.908203 0.900391
    (99,63,63,62)    0.908203 0.916016 0.916016 0.908203
    (99,63,64,62)    0.908203 0.916016 0.916016 0.908203
    (99,63,65,62)    0.900391 0.908203 0.908203 0.900391
    (99,64,62,62)    0.900391 0.908203 0.908203 0.900391
    (99,64,63,62)    0.908203 0.916016 0.916016 0.908203
    (99,64,64,62)    0.908203 0.916016 0.916016 0.908203
    (99,64,65,62)    0.900391 0.908203 0.908203 0.900391
    (99,65,62,62)    0.892578 0.900391 0.900391 0.892578
    (99,65,63,62)    0.900391 0.908203 0.908203 0.900391
    (99,65,64,62)    0.900391 0.908203 0.908203 0.900391
    (99,65,65,62)    0.892578 0.900391 0.900391 0.892578
 
```

Example using the python API
```python
>>> import adios2
>>> f = adios2.FileReader("run-488.aca")
>>> f.available_variables()
{'ckpt.bp/U': {'AvailableStepsCount': '1', 'Max': '1', 'Min': '0.0964156', 'Shape': '8, 66, 66, 66', 'SingleValue': 'false', 'Type': 'double'}, 'ckpt.bp/V': {'AvailableStepsCount': '1', 'Max': '0.478857', 'Min': '2.44533e-48', 'Shape': '8, 66, 66, 66', 'SingleValue': 'false', 'Type': 'double'}, 'ckpt.bp/step': {'AvailableStepsCount': '1', 'Max': '1400', 'Min': '1400', 'Shape': '', 'SingleValue': 'true', 'Type': 'int32_t'}, 'gs.bp/U': {'AvailableStepsCount': '200', 'Max': '1', 'Min': '0.0813067', 'Shape': '128, 128, 128', 'SingleValue': 'false', 'Type': 'double'}, 'gs.bp/V': {'AvailableStepsCount': '200', 'Max': '0.674805', 'Min': '0', 'Shape': '128, 128, 128', 'SingleValue': 'false', 'Type': 'double'}, 'gs.bp/step': {'AvailableStepsCount': '200', 'Max': '2000', 'Min': '10', 'Shape': '', 'SingleValue': 'true', 'Type': 'int32_t'}, 'pdf.bp/U/bins': {'AvailableStepsCount': '200', 'Max': '0.999175', 'Min': '0.0813067', 'Shape': '1000', 'SingleValue': 'false', 'Type': 'double'}, 'pdf.bp/U/pdf': {'AvailableStepsCount': '200', 'Max': '16384', 'Min': '0', 'Shape': '128, 1000', 'SingleValue': 'false', 'Type': 'double'}, 'pdf.bp/V/bins': {'AvailableStepsCount': '200', 'Max': '0.67413', 'Min': '0', 'Shape': '1000', 'SingleValue': 'false', 'Type': 'double'}, 'pdf.bp/V/pdf': {'AvailableStepsCount': '200', 'Max': '16384', 'Min': '0', 'Shape': '128, 1000', 'SingleValue': 'false', 'Type': 'double'}, 'pdf.bp/step': {'AvailableStepsCount': '200', 'Max': '2000', 'Min': '10', 'Shape': '', 'SingleValue': 'true', 'Type': 'int32_t'}}
>>> u = f.read("ckpt.bp/U")
ReadResponseHandler: response size = 18399744 operator type [127]
>>> vu = f.inquire_variable("ckpt.bp/V")
>>> vu.set_accuracy(0.1, 0.0, False)
>>> u = f.read(vu)
ReadResponseHandler: response size = 1299478 operator type [7]
```

### Inspecting the logs

On the dtn node, there should be a log.adios2_remote_server file based on the session:

```
$ cat /ccs/home/YOURUSERNAME/log.adios2_remote_server
Listening on Port 37003
Max threads = 8
Got an open request (mode RandomAccess) for file /lustre/orion/csc143/world-shared/kmehta/gray-scott-ensemble/campaign/Du-0.2-Dv-0.1-F-0.01-k-0.05/ckpt.bp
Reading var U with absolute error 0 in norm 0
Returning 17.5 MB for Get<double>(U) start = {0,0,0,0} count = {8,66,66,66}
Reading var V with absolute error 0.1 in norm 0
Returning 1.2 MB for Get<double>(V) start = {0,0,0,0} count = {8,66,66,66}
closing ADIOS file "/lustre/orion/csc143/world-shared/kmehta/gray-scott-ensemble/campaign/Du-0.2-Dv-0.1-F-0.01-k-0.05/ckpt.bp" total sent 18.8 MB in 2 Get()s
```

 

