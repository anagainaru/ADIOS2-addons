# DataMan engine

**Building ADIOS2 with DataMan**

Uses the ZeroMQ library for data transfer. For Summit/Frontier this means the `libzmq` must be loaded, i.e. adding the following line to the build scripts

```
module load libzmq/4.3.4
```

**DataMan with Kokkos**

DataMan can be used to stream GPU pointers and Kokkos Views between systems. An example of this can be found in `examples/hello/datamanKokkos`

