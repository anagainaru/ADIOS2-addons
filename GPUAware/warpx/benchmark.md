# OpenPMD benchmark

Instructions from [https://openpmd-api.readthedocs.io/en/0.13.3/usage/benchmarks.html](https://openpmd-api.readthedocs.io/en/0.13.3/usage/benchmarks.html)

## Writing
Code in: `examples/8a_benchmark_write_parallel.cpp`

This benchmark writes a few meshes and particles, either 1D, 2D or 3D.

The meshes are viewed as grid of mini blocks. As an example, the mini blocks dimension can be [16, 32, 32].

Next we define the grid based on the mini block. say, [32, 32, 16]. Then our actual mesh size is [16x32, 32x32, 32x16].

Here is a sample input file (“w.input”):
```
dim=3
balanced=true
ratio=1
steps=10
minBlock=16 32 32
grid=32 32 16
```

With the above input file, we will create an openPMD file with the above mesh using
- 3D data
- balanced load
- particle to mesh ratio = 1
- 10 iteration steps

*Note: All files generated are group based. i.e. One file per iteration.*

To run:

```
./8a_benchmark_write_parallel w.input
```

then the file generated are: `../samples/8a_parallel_3Db_*`

### Changes to the write benchmark
