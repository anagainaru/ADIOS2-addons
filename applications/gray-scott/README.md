# Gray-Scott reaction diffusion model

Changes to the code in the ADIOS2-Examples repo are focusing on replacing the way u and v are stored from `std::vector<double>` to `Kokkos::View<double ***>`. The files in this repo are the only ones with updated code compared to the cpp version.

**Current limitations of the code:**
- The layout of the arrays is fixed to `LayoutLeft` (due to the l2i function that is used by the exchange functions)
- The whole arrays (u and v) are copied from the GPU to the CPU to exchange the bouderies

**Running the code:**

On summit the following modules need to be imported
```
module load gcc/10.2
module load cuda/11.5
module load cmake/3.23
```

Compiling the code from the `ADIOS2-Examples` repo for a GPU architecture:
```
 cmake -DKokkos_ROOT=/path/to/kokkos/install -DADIOS2_DIR=/path/to/adios/install/lib64/cmake/adios2 -D CMAKE_CUDA_ARCHITECTURES=70 -D CMAKE_CXX_STANDARD=17 -D CMAKE_CXX_EXTENSIONS=OFF -DCMAKE_CXX_COMPILER=$HOME/kokkos/kokkos/bin/nvcc_wrapper -DCMAKE_C_COMPILER=gcc  ..
```
Running using MPI (on Summit jsrun):
```
mpirun -n 3 ./bin/adios2-gray-scott-kokkos settings-files.json
python ../source/cpp/gray-scott/plot/gsplot.py -i gs.bp
```
<img width="536" alt="Screenshot 2023-06-16 at 2 21 07 PM" src="https://github.com/anagainaru/ADIOS2-addons/assets/16229479/9d90e49b-dc53-4753-aa4c-0e2ec29fdd9d">

