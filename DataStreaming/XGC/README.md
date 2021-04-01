# Running XGC-Avocado on Summit

### Build codes

Building and running the codes coupled will require EFFIS, ADIOS and VTK-M. 
For EFFIS and VTK-M, the intructions here will use the libraries installed in Eric's folders in `/gpfs/alpine/csc143/world-shared/esuchyta/spack/opt/spack/linux-rhel7-power9le/gcc-9.3.0/`. ADIOS will use the implementation in `/ccs/home/againaru/adios/ADIOS2-init/install`.

**XGC**

Root path: `/gpfs/alpine/csc143/proj-shared/againaru/xcg`

Download the latest XGC code
```
git clone https://github.com/PrincetonUniversity/XGC-Devel.git
cd XGC-Devel
mkdir XGC-bin
cd XGC-bin
```
Changes to the `XGC-Devel/CMake/find_dependencies_summit.cmake`:
- Add `set(XGC_COMPILE_DEFINITIONS NO_TASKMAP)`
- Update the path to ADIOS2 from `set(ADIOS2_ROOT "/gpfs/alpine/world-shared/csc143/jyc/summit/sw/adios2/devel/pgi")`
to `set(ADIOS2_ROOT "/ccs/home/againaru/adios/ADIOS2-init/install")`

Import the modules and set environmental variables
```
module load pgi/19.10
module load pgi-cxx14
module load spectrum-mpi
module load cmake
module load cuda/10.1.243

module load netlib-lapack/3.8.0
module load hypre/2.13.0
module load fftw/3.3.8
module load hdf5/1.10.4
module load python/2.7.15

export XGC_PLATFORM=summit
export NVCC_WRAPPER_DEFAULT_COMPILER=pgc++
effis=/gpfs/alpine/csc143/world-shared/esuchyta/spack/opt/spack/linux-rhel7-power9le/pgi-19.10/effis-develop-rgzjnggufdzpus7pqco2wza3hjzlyxty
```

Build XGC with EFFIS turned on
```
cmake \
        -DCMAKE_CXX_COMPILER=/gpfs/alpine/world-shared/phy122/lib/install/summit/kokkos/pgi19.10/bin/nvcc_wrapper \
        -DCMAKE_Fortran_COMPILER=mpif90 \
        -DXGC_USE_CABANA=ON \
        -DBUILD_TESTING=OFF \
        -DBUILD_KERNELS=OFF \
        -DEFFIS=ON \
        -DCMAKE_PREFIX_PATH=$effis \
        ../XGC-Devel
make -j4
```

Software versions
```
CMAKE version
cmake/3.18.2 (L,D)
```

Or use the `build-xgc.sh` script placed in the same folder where `XGC-Devel` is stored.


**VTK-Avocado**

Root path: `/gpfs/alpine/csc143/proj-shared/againaru/avocado`

Download the latest VTK Avocado code
```
git clone git@gitlab.kitware.com:suchyta1/vtk-avocado.git
mkdir build
cd build
```

Build the code and set the variables to the EFFIS, ADIOS and VTK-M softwares
```
module load gcc/9.3.0
module load spectrum-mpi
module load cmake
module load python/3.7.0

effis=/gpfs/alpine/csc143/world-shared/esuchyta/spack/opt/spack/linux-rhel7-power9le/gcc-9.3.0/effis-develop-trvkjzyfbsh3d3zfds66h3wahd7ylj4o
vtkm=/gpfs/alpine/csc143/world-shared/esuchyta/spack/opt/spack/linux-rhel7-power9le/gcc-9.3.0/vtk-m-master-672m33oblm5xiazctojrd4a3v4vowk33
cuda=/sw/summit/cuda/11.1.0
adios=/ccs/home/againaru/adios/ADIOS2-init/install
```

Build VTK-Avocado
```
cmake \
        -DCMAKE_VERBOSE_MAKEFILE=ON \
        -DCMAKE_PREFIX_PATH="$effis;$vtkm;$cuda;$adios" \
        -DEFFIS=ON \
        ../vtk-avocado

make -j4
``

Software versions
```
CMAKE version
cmake/3.18.2 (L,D)
```

Or use the `build-avocado.sh` script placed in the same folder where `vtk-avocado` is stored.


### Run the coupled codes

Root path: `/gpfs/alpine/csc143/proj-shared/againaru/xcg`

Set-up the SPACK environment and load EFFIS
```
source /gpfs/alpine/csc143/proj-shared/againaru/xcg/spack-setup-env.sh
spack load effis +compose
```

EFFIS requires `yaml` files to define the properties of a run.
This repo contains two such configuration files:
- `xgc-small-separate.yaml` for a small demo running XGC with Avocado through BP files
- `xgc-small-mpmd.yaml` for a small demo running XGC with Avocado the SSC engine files

To run the examples
```
effis-compose.py {config_file.yaml}
effis-submit runs/demo-small-1
bjobs --list
effis-view runs/demo-small-1 xgc vtkm
effis-view runs/demo-small-1 login-proc
```

