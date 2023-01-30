
## Install HeapTrack

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
makeinstall

ls /home/ana/kits/heaptrack/install/bin
> heaptrack  heaptrack_print
```

**Install the analyzer GUI**
