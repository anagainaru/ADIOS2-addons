# Campaign management

Using the HPC-Campaign tool (https://github.com/ornladios/hpc-campaign)

Instructions to install: https://hpc-campaign.readthedocs.io/en/latest/installation.html

```
(.venv) $ git clone https://github.com/ornladios/hpc-campaign.git
(.venv) $ cd hpc-campaign
(.venv) $ pip3 install -e .
(.venv) $ hpc_campaign list
```

Create a folder `~/.config/hpc-campaign/` that will store all the data used in the campaign management.

**Configuration options**

The yaml file needs to be updated to include parths/names used in the campaign setup.
Intructions on what needs to be added to this yaml file: https://hpc-campaign.readthedocs.io/en/latest/installation.html#setup

```
Campaign:
  hostname: OLCF
  campaignstorepath: /path/to/campaign-store
  cachepath: /path/to/campaign-cache
  verbose: 0
```

### Usage

Create a new campaign file using:
```
$ hpc_campaign manager picongpu_olcf_001.aca create
=============================
Create new archive /path/to/campaign-store/picongpu_olcf_001.aca
```

Adding HDF-5 or ADIOS-2 datasets to a campaign file:
```
$ hpc_campaign manager picongpu_olcf_001.aca dataset run/002_lwfa/simOutput/openPMD/simOutput_000000.bp
=============================
Found host OLCF in database, rowid = 1
Found directory /lustre/orion/csc143/proj-shared/againaru/picongpu with hostID 1 in database, rowid = 1
Process entry run/002_lwfa/simOutput/openPMD/simOutput_000000.bp:
Add dataset run/002_lwfa/simOutput/openPMD/simOutput_000000.bp to archive
Add replica run/002_lwfa/simOutput/openPMD/simOutput_000000.bp to archive
AddReplicaToArchive(host=1, dir=1, key=0, name=run/002_lwfa/simOutput/openPMD/simOutput_000000.bp dsid=2, time=1760282400000000000, size=5315508)
Replica rowid = 2
```

Adding text files to the campaign:
```
$ hpc_campaign manager picongpu_olcf_001.aca text lwfa_picongpu_newstii001/pypicongpu.json
=============================
Found host OLCF in database, rowid = 1
Found directory /lustre/orion/csc143/proj-shared/againaru/picongpu with hostID 1 in database, rowid = 1
Process entry lwfa_picongpu_newstii001/pypicongpu.json:
Add dataset lwfa_picongpu_newstii001/pypicongpu.json to archive
Add replica lwfa_picongpu_newstii001/pypicongpu.json to archive
AddReplicaToArchive(host=1, dir=1, key=0, name=lwfa_picongpu_newstii001/pypicongpu.json dsid=1, time=1760293106000000000, size=40134)
Replica rowid = 1
```

Inspecting the contents of a campaign file:
```
$ hpc_campaign manager picongpu_olcf_001.aca info
=============================
ADIOS Campaign Archive, version 0.5, created on Oct 15 13:51

Hosts and directories:
  OLCF   longhostname = frontier.olcf.ornl.gov
     1. /lustre/orion/csc143/proj-shared/againaru/picongpu

Other Datasets:
    e3d3551d8dae3bf69a1f87e9192a3908  TEXT   Oct 12 14:18   lwfa_picongpu_newstii001/pypicongpu.json
    66d2231240693f158543149e859d3b3e  ADIOS  Oct 12 11:20   run/002_lwfa/simOutput/openPMD/simOutput_000000.bp
    7781eca52f7e3d41b19dbace8dcaed0d  ADIOS  Oct 12 12:52   run/002_lwfa/simOutput/openPMD/simOutput_fields_000000.bp
...
    f6ca0d94d13030e6aba34dbb877a8d8e  ADIOS  Oct 12 12:52   run/002_lwfa/simOutput/openPMD/simOutput_fields_150000.bp
```

Using BPLS on a campaign file
```
$ ./bin/bpls -l /lustre/orion/csc143/proj-shared/againaru/campaign-store/picongpu_olcf_001.aca
  char      lwfa_picongpu_newstii001/pypicongpu.json                                                                       {40134} = A / Z
  uint64_t  run/002_lwfa/simOutput/openPMD/simOutput_000000.bp/data/0/particles/en_all/particlePatches/extent/x            {256} = 0 / 0
  uint64_t  run/002_lwfa/simOutput/openPMD/simOutput_000000.bp/data/0/particles/en_all/particlePatches/extent/y            {256} = 0 / 0
  uint64_t  run/002_lwfa/simOutput/openPMD/simOutput_000000.bp/data/0/particles/en_all/particlePatches/extent/z            {256} = 0 / 0
  uint64_t  run/002_lwfa/simOutput/openPMD/simOutput_000000.bp/data/0/particles/en_all/particlePatches/numParticles        {256} = 0 / 0
  uint64_t  run/002_lwfa/simOutput/openPMD/simOutput_000000.bp/data/0/particles/en_all/particlePatches/numParticlesOffset  {256} = 0 / 0
  uint64_t  run/002_lwfa/simOutput/openPMD/simOutput_000000.bp/data/0/particles/en_all/particlePatches/offset/x            {256} = 0 / 0
  uint64_t  run/002_lwfa/simOutput/openPMD/simOutput_000000.bp/data/0/particles/en_all/particlePatches/offset/y            {256} = 0 / 0
  uint64_t  run/002_lwfa/simOutput/openPMD/simOutput_000000.bp/data/0/particles/en_all/particlePatches/offset/z            {256} = 0 / 0
  float     run/002_lwfa/simOutput/openPMD/simOutput_fields_000000.bp/data/000000/fields/B/x                               {512, 1920, 1} = 0 / 0
  float     run/002_lwfa/simOutput/openPMD/simOutput_fields_000000.bp/data/000000/fields/B/y                               {512, 1920, 1} = 0 / 0
  float     run/002_lwfa/simOutput/openPMD/simOutput_fields_000000.bp/data/000000/fields/B/z                               {512, 1920, 1} = 0 / 0
  float     run/002_lwfa/simOutput/openPMD/simOutput_fields_000000.bp/data/000000/fields/E/x                               {512, 1920, 1} = 0 / 0
  float     run/002_lwfa/simOutput/openPMD/simOutput_fields_000000.bp/data/000000/fields/E/y                               {512, 1920, 1} = 0 / 0
  float     run/002_lwfa/simOutput/openPMD/simOutput_fields_000000.bp/data/000000/fields/E/z                               {512, 1920, 1} = 0 / 0
  float     run/002_lwfa/simOutput/openPMD/simOutput_fields_000000.bp/data/000000/fields/e_all_chargeDensity               {512, 1920, 1} = 0 / 0
  float     run/002_lwfa/simOutput/openPMD/simOutput_fields_000000.bp/data/000000/fields/e_all_energyDensity               {512, 1920, 1} = 0 / 0
  float     run/002_lwfa/simOutput/openPMD/simOutput_fields_000000.bp/data/000000/fields/en_all_chargeDensity              {512, 1920, 1} = 0 / 0
  float     run/002_lwfa/simOutput/openPMD/simOutput_fields_000000.bp/data/000000/fields/en_all_energyDensity              {512, 1920, 1} = 0 / 0
  float     run/002_lwfa/simOutput/openPMD/simOutput_fields_000000.bp/data/000000/fields/n_all_chargeDensity               {512, 1920, 1} = 0 / 0
  float     run/002_lwfa/simOutput/openPMD/simOutput_fields_000000.bp/data/000000/fields/n_all_energyDensity               {512, 1920, 1} = 0 / 0
...
  float     run/002_lwfa/simOutput/openPMD/simOutput_fields_150000.bp/data/150000/fields/B/x                               {512, 1920, 1} = 0 / 0
  float     run/002_lwfa/simOutput/openPMD/simOutput_fields_150000.bp/data/150000/fields/B/y                               {512, 1920, 1} = 0 / 0
  float     run/002_lwfa/simOutput/openPMD/simOutput_fields_150000.bp/data/150000/fields/B/z                               {512, 1920, 1} = 0 / 0
  float     run/002_lwfa/simOutput/openPMD/simOutput_fields_150000.bp/data/150000/fields/E/x                               {512, 1920, 1} = 0 / 0
  float     run/002_lwfa/simOutput/openPMD/simOutput_fields_150000.bp/data/150000/fields/E/y                               {512, 1920, 1} = 0 / 0
  float     run/002_lwfa/simOutput/openPMD/simOutput_fields_150000.bp/data/150000/fields/E/z                               {512, 1920, 1} = 0 / 0
  float     run/002_lwfa/simOutput/openPMD/simOutput_fields_150000.bp/data/150000/fields/e_all_chargeDensity               {512, 1920, 1} = 0 / 0
  float     run/002_lwfa/simOutput/openPMD/simOutput_fields_150000.bp/data/150000/fields/e_all_energyDensity               {512, 1920, 1} = 0 / 0
  float     run/002_lwfa/simOutput/openPMD/simOutput_fields_150000.bp/data/150000/fields/en_all_chargeDensity              {512, 1920, 1} = 0 / 0
  float     run/002_lwfa/simOutput/openPMD/simOutput_fields_150000.bp/data/150000/fields/en_all_energyDensity              {512, 1920, 1} = 0 / 0
  float     run/002_lwfa/simOutput/openPMD/simOutput_fields_150000.bp/data/150000/fields/n_all_chargeDensity               {512, 1920, 1} = 0 / 0
  float     run/002_lwfa/simOutput/openPMD/simOutput_fields_150000.bp/data/150000/fields/n_all_energyDensity               {512, 1920, 1} = 0 / 0
```
