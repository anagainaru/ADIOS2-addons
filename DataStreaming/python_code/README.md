# Running python SST example

**For one writer one reader**
Add the path to the install adios python API then run the two codes.
```
export PYTHONPATH=/Users/95j/work/adios/ADIOS2-init/install/lib/python3.9/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=/Users/95j/work/adios/ADIOS2-init/install/lib:$LD_LIBRARY_PATH

python writer.py 100 1 &
python reader.py 100 1
```

**For one writer multiple readers**

The *RendezvousReaderCount* parameter needs to be set to 2 in the adios configuration XML file.
```
$ vim adios.xml
<parameter key="RendezvousReaderCount" value="2"/>
```

Two instances of the reader need to be executed.
```
export PYTHONPATH=/Users/95j/work/adios/ADIOS2-init/install/lib/python3.9/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=/Users/95j/work/adios/ADIOS2-init/install/lib:$LD_LIBRARY_PATH

python writer.py 100 1 & 
python reader.py 100 1 & 
python reader.py 100 1
```
