## Valgrind

```
wget https://sourceware.org/pub/valgrind/valgrind-3.20.0.tar.bz2
```

Massif visualizer examples
<img width="774" alt="Screen Shot 2023-02-13 at 6 48 53 PM" src="https://user-images.githubusercontent.com/16229479/218600259-5c44a112-e03d-46cd-888a-4c562069708d.png">


##  HeapTrack

**Install the data collector**

Install all the dependencies for HeapTrack
- Boost

```python
wget https://boostorg.jfrog.io/artifactory/main/release/1.81.0/source/boost_1_81_0.tar.bz2
tar -xf boost_1_81_0.tar.bz2
./bootstrap.sh --prefix=/home/ana/kits/boost_1_81_0/install
# consider using the --show-libraries and --with-libraries=library-name-list options to limit the long wait you'll experience if you build everything
./b2 install

export CMAKE_PREFIX_PATH=/home/ana/kits/boost_1_81_0/install
```

Install the heaptrack package

```python
git clone git@github.com:KDE/heaptrack.git
# in the build directory in heaptrack
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j4
cmake -D CMAKE_INSTALL_PREFIX=/home/ana/kits/heaptrack/install ..
make install

ls /home/ana/kits/heaptrack/install/bin
> heaptrack  heaptrack_print
```

**Install the analyzer GUI**

Install QT5
```
wget https://download.qt.io/archive/qt/5.15/5.15.8/single/qt-everywhere-opensource-src-5.15.8.tar.xz
./configure --prefix=/path/to/qt5/install
make
make install
```

Install the heaptrack package with GUI

```python
git clone git@github.com:KDE/heaptrack.git
# in the build directory in heaptrack
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j4
cmake -D CMAKE_INSTALL_PREFIX=/home/ana/kits/heaptrack/install -D HEAPTRACK_BUILD_GUI=TRUE ..
make install
```

## Run Heaptrack

```
$HOME/kits/heaptrack/install/bin/heaptrack ./adios2-cuda-basis
```
