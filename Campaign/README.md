# Campaign management system

The campaign management readme includes:
- Access to variables across multiple runs
- Remote access (remote and local side set-ups)
- Setting the accuracy of a read (primary and derived variables)

The versions on ADIOS2 on the remote and local sides needs to be the same. For querying on accuracy, this version has to be the one in Norbert's repo (https://github.com/pnorbert/ADIOS2) the `remote_compression` branch. 

## Remote side (where the data resides)

### Setup remote access

Create a folder for installing adios2 (e.g. `~/dtn/sw/adios2`) and install ADIOS2 to this location with the options needed for the remote access (derived variables, compression etc. if they are used to answer a request). 

**Frontier:** For the DTN nodes on Frontier, login to the dtn nodes (dtn.olcf.ornl.gov) and use the following parameters to install ADIOS2:

```cmake
        -DCMAKE_INSTALL_PREFIX=/path/to/dtn/sw/adios2 \
        -DCMAKE_EXE_LINKER_FLAGS="-Wl,--allow-multiple-definition" \
        -DCMAKE_CXX_STANDARD_LIBRARIES=-lstdc++fs \
        -DBUILD_SHARED_LIBS=OFF \
        -DADIOS2_BUILD_EXAMPLES=OFF \
        -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
        -DBUILD_TESTING=OFF \
        -DCMAKE_PREFIX_PATH="/ccs/home/pnorbert/dtn/sw/zfp" \
        -DADIOS2_USE_Fortran=OFF \
        -DADIOS2_USE_DataMan=OFF \
        -DADIOS2_USE_HDF5=OFF \
        -DADIOS2_USE_Python=OFF \
        -DADIOS2_USE_MGARD=OFF \
        -DADIOS2_USE_AWSSDK=OFF \
        -DADIOS2_USE_SST=ON \
        -DADIOS2_USE_Derived_Variable=ON \
        -DADIOS2_INSTALL_GENERATE_CONFIG=OFF \
```

To check the configuration of an installation use `bpls`
```
$ ~/dtn/sw/adios2/bin/bpls -vV
blps: ADIOS file introspection utility

Build configuration:
ADIOS version: 2.10.0
C++ Compiler:  GNU 8.5.0
Target OS:     Linux-4.18.0-553.16.1.el8_10.x86_64
Target Arch:   x86_64
Available engines = 8: BP3, BP4, BP5, SST, Inline, MHS, Null, Skeleton
Available operators = 1: ZFP
Available features = 6: BP3BP4BP5MHS, SST, ZFP, O_DIRECT, SYSVSHMEM, DERIVED_VARIABLE
```

### Create the campaign files

**Dependencies:**

 - SQLite 3  (libsqlite3-dev on Ubuntu)
   
The remote server needs to have an `adios2.yaml` file in ` ~/.config/adios2/` with the location of the campaign files and caches (for writing the cache path is not used).

```
Campaign:
  active: true
  hostname: OLCF
  campaignstorepath: /path/to/adios-campaign-store
  cachepath: /path/to/campaign-cache
  verbose: 0
```

**Frontier:** I used the default modules on Frontier (script from adios2 repo) to build ADIOS2 with derived variables and campaign management ON. 
```
module load PrgEnv-gnu-amd/8.5.0
module load craype-accel-amd-gfx90a
module load cmake/3.27.9
module load cray-python/3.11.5

$ ./install-kokkos-frontier/bin/bpls -vV
blps: ADIOS file introspection utility

Build configuration:
ADIOS version: 2.10.0
C++ Compiler:  GNU 12.3.0
Target OS:     Linux-5.14.21-150500.55.49_13.0.57-cray_shasta_c
Target Arch:   x86_64
Available engines = 9: BP3, BP4, BP5, SST, SSC, Inline, MHS, Null, Skeleton
Available operators = 2: BZip2, PNG
Available features = 18: BP3BP4BP5DATAMAN, MHS, SST, FORTRAN, MPI, PYTHON, BZIP2, PNG, O_DIRECT, SODIUM, SYSVSHMEM, ZEROMQ, PROFILING, DERIVED_VARIABLE, GPU_SUPPORT, KOKKOS, KOKKOS_HIP, CAMPAIGN
```

Applications ran normally and create `bp` files. While the `adios2.yaml` file exists (and the active field is set to true), there will be an additional folder created wherever the execution took place with sqlite files.

```
$ ls -la
drwxr-sr-x  5 againaru csc143 99328 Nov 15 12:36 .
drwxr-sr-x 11 againaru csc143 99328 Nov 15 10:40 ..
drwxr-sr-x  2 againaru csc143 99328 Nov 15 12:39 .adios-campaign
drwxr-sr-x  2 againaru csc143 99328 Nov 15 12:22 gs-derived.bp
```

**Create/Update campaign files**

Repo: https://github.com/ornladios/hpc-campaign
- Contains the `hpc_campaign_manager.py` file to create/update campaign files

If the create command is called from the same folder where the `.adios-campaign` files are, the `bp` files are identified automatically. Otherwise, the path to the file needs to be given explicitly.

```
$ ./hpc-campaign/source/hpc_campaign_manager/hpc_campaign_manager.py create gray-scott-derived-run1.aca -f /path/to/gray-scott-ensemble/Du-0.2-Dv-0.1-F-0.01-k-0.05/gs-derived.bp
Create archive
Inserted host OLCF into database, rowid = 1
Inserted directory /path/to/gray-scott-ensemble/Du-0.2-Dv-0.1-F-0.01-k-0.05 into database, rowid = 1
Process entry gs-derived.bp:
Add dataset gs-derived.bp to archive

$ ls /path/to/adios-campaign-store
gray-scott-derived-run1.aca

$ ./hpc-campaign/source/hpc_campaign_manager/hpc_campaign_manager.py info gray-scott-derived-run1.aca
info archive
ADIOS Campaign Archive, version 0.1, created on 2024-11-15 15:29:18.552364
hostname = OLCF   longhostname = frontier.olcf.ornl.gov
    dir = /path/to/gray-scott-ensemble/Du-0.2-Dv-0.1-F-0.01-k-0.05
        dataset = gs-derived.bp     created on 2024-11-15 12:02:03
```

Other bp files can be updated to the campaign management file by running the update option in the hpc_campaign_manager.py script.

```
$ ./hpc-campaign/source/hpc_campaign_manager/hpc_campaign_manager.py update gray-scott-derived-run1.aca -f ckpt.bp
update archive
Found host OLCF in database, rowid = 1
Inserted directory /path/to/gray-scott-ensemble/campaign/Du-0.2-Dv-0.1-F-0.01-k-0.05 into database, rowid = 2
Process entry ckpt.bp:
Add dataset ckpt.bp to archive
```

Each folder will have a separate entry containing all the bp files added from the folder.

```
./hpc-campaign/source/hpc_campaign_manager/hpc_campaign_manager.py info gray-scott-derived-run1.aca
info archive
ADIOS Campaign Archive, version 0.1, created on 2024-11-15 15:36:32.881251
hostname = OLCF   longhostname = frontier.olcf.ornl.gov
    dir = /path/to/first/gray-scott-ensemble/Du-0.2-Dv-0.1-F-0.01-k-0.05
        dataset = gs-derived.bp     created on 2024-11-15 12:02:03
    dir = /path/to/second/gray-scott-ensemble/campaign/Du-0.2-Dv-0.1-F-0.01-k-0.05
        dataset = ckpt.bp     created on 2024-04-18 14:21:57
        dataset = pdf.bp     created on 2024-04-18 14:22:50
```

## Local side (where the query occurs)

The aca files need to be copied from the remote local to the local machine.

```
$ scp USERNAME@remote.location:/path/to/remote/adios-campaign-store/*.aca /path/to/localadios-campaign-store/
gray-scott-derived-run1.aca                                                          100%   68KB 260.2KB/s   00:00
```

### Campaign manager
 
Repo: https://github.com/ornladios/hpc-campaign

The local campaign manager requires two configuration files: `hosts.yaml` and `adios2.yaml`
 
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
      serverpath: ~{REMOTE_USERNAME}/dtn/sw/adios2/bin/adios2_remote_server -background -report_port_selection -v -v -l /ccs/home/YOURUSERNAME/log.adios2_remote_server
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

### Inspecting campaign files

Campaign files need to be stored into the `campaignstorepath` path. 

```
$ du -h ~/dropbox/adios-campaign-store/*.aca
1.0M	frontier_s3d_001.aca
 68K	gray-scott-derived-run1.aca
120K	run-488.aca
```

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

```
$ ./bin/bpls gray-scott-derived-run1.aca
  double   ckpt.bp/U                    {8, 66, 66, 66}
  double   ckpt.bp/V                    {8, 66, 66, 66}
  int32_t  ckpt.bp/step                 scalar
  double   gs-derived.bp/U              200*{128, 128, 128}
  double   gs-derived.bp/V              200*{128, 128, 128}
  double   gs-derived.bp/derived/sumUV  200*{128, 128, 128}
  int32_t  gs-derived.bp/step           200*scalar
  double   pdf.bp/U/bins                200*{1000}
  double   pdf.bp/U/pdf                 200*{128, 1000}
  double   pdf.bp/V/bins                200*{1000}
  double   pdf.bp/V/pdf                 200*{128, 1000}
  int32_t  pdf.bp/step                  200*scalar
```

The `--show-derived` option is currently only working for bp files and not aca files.

### Connector

The connector needs to be ran before reading the remote data:
```
hpc-campaign/source/hpc_campaign_connector$ python3 ./hpc_campaign_connector.py -c ~/.config/adios2/hosts.yaml -p 30000
 
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

## Accuracy based queries

### ADIOS build

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

The `PYTHONPATH` environment variable needs to point to `path/to/adios2/install/lib/python{VERSION}/site-packages`

Setting the accuracy will compress the data using zfp on the remote side and only send the data that matches the accuracy requested. The query works for derived variables as well, the remote server reads all the primary data, computes the derived data, compresses the derived data using zfp and transfers the compressed data. Example using the python API

```python
>>> import adios2
>>> f = adios2.FileReader("gray-scott-derived-run1.aca")
>>> u = f.read("ckpt.bp/U")
ReadResponseHandler: response size = 18399744 operator type [127]
>>> vu = f.inquire_variable("ckpt.bp/V")
>>> vu.set_accuracy(0.1, 0.0, False)
>>> u = f.read(vu)
ReadResponseHandler: response size = 1299478 operator type [7]
>>> sumuv = f.inquire_variable("gs-derived.bp/derived/sumUV")
>>> sumuv.set_accuracy(0.1, 0.0, False)
>>> u = f.read(sumuv)
ReadResponseHandler: response size = 157366 operator type [7]
```
