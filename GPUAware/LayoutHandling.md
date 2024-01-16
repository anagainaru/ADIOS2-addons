# Data layout in ADIOS2

ADIOS2 is row major so any data received will be stored in layout right format.

C++/C buffers use layout right, Fortran and GPU buffers layout left. ADIOS2 deals with Fortran codes by switching the dimensions.

## Current solution for layout mismatch

Variables will have two new fields to deal with GPUs:
- Memory space (where the buffer was allocated): GPU or CPU
- LayoutMismatch (if the layout of the memory space is the same as the layout used by ADIOS)

On the write side we fix the memory space of a variable during the first `Put`, succeding `Put`s on a different memory space will raise an error. If the memory space is set to GPU and the ADIOS layout is C++ there is a mismatch, otherwise there is no mismath between the two layouts. If there is a mismatch, we switch the dimensions of the shape, start and count of a variable and we switch every time a new shape is set on the variable.

On the read side, the `Shape` function will be able to receive a memory space (default Host) so that it returns switched dimensions. On Get, adios2 has the memory space so it can set the correct dimensions for the read buffer. 

The write memory space is stored as a variable attribute and can be used by the user to know the data is transposed. 

## Why do we need this?

If we write/read on the same memory space the sender/receiver buffers will have the same shape/content.
If the layouts are different, the read will not see the global data the same way as the writer. The next sections show examples.

### Writing from a CPU pointer

<img width="845" alt="Screenshot 2024-01-03 at 3 34 25 PM" src="https://github.com/anagainaru/ADIOS2-addons/assets/16229479/0d8b8687-48f8-4556-9ab7-1ec03b71036b">

Each process is writing a portion of the global array that is combined at the end based on the start and count values attached to a variable.

In the example above, the global shape is (2, 6) and each process is writing a (2,3) local array.

On the read side, inspecting the variable would return the dimensions in layout right format. The user is responsible with allocating a corresponding array (in the previous case a MDrange of dimensions 2,6 if reading from a single process).

**Reading the data using GPU buffers**

We will asume for the rest of the section that the data on storage was wrote by 3 processes using CPU buffers with a local array of size (3,2) -- global array of (3,6)

Inspecting the variable would return a global size of (3,6). If we want to distribute the global array among 3 processes (split on the second dimension) we would define a local adios variable of shape (3,2).

However, if the Kokkos::View is also a 3x2 array, the default layout on the GPU backend is layout left, while ADIOS2 is layout right so there will be a sramble in how the data is read from storage.

<img width="940" alt="Screenshot 2024-01-03 at 3 44 12 PM" src="https://github.com/anagainaru/ADIOS2-addons/assets/16229479/8c434856-267f-4551-9181-4530f0d6cb0e">

The dimensions for a Layout Left buffer will need to be fipped in order to get correct data from the storage

<img width="847" alt="Screenshot 2024-01-03 at 3 56 15 PM" src="https://github.com/anagainaru/ADIOS2-addons/assets/16229479/a5251c7b-2e6e-4baa-96bd-744be34299b0">

NOTE. The information will be transposed compared to what a CPU buffer would have.

### Writing from a GPU pointer

<img width="843" alt="Screenshot 2024-01-03 at 3 58 53 PM" src="https://github.com/anagainaru/ADIOS2-addons/assets/16229479/67e21c8e-0c5e-4bc4-b787-b57d3fd85874">

Two processes are writing a local 2x3 array into a 2x6 global array from a buffer using Layout Left. Since adios variables are using Layout Right, if the variables are using the same dimensions the data will be scrambled (top right in the figure).

Variable dimensions needs to be flipped in order for the data to be stored correctly (bottom right in the figure).


## Discussion

The `DefineVariable` function in adios2 is not templated over the memory space so variables are not pinned to a specific memory space until the `SetMemorySpace` call or until the first Put/Get on a buffer. 

```c++
    const adios2::Dims shape{Nx, Ny * size};
    const adios2::Dims start{0, Ny * rank};
    const adios2::Dims count{Nx, Ny};
    auto data = bpIO.DefineVariable<float>("bpFloats", shape, start, count);
```

On the write side:
- ADIOS2 can flip the dimensions without the user having to worry about it. However:
- The dimensions will not be flipped at DefineVariable but only on the first Put function
- If a user prints the bpFloats.Shape() before and after put it would get two different results: Nx, Ny and Ny, Nx
- `bpls` after write will show a shape that is flipped compared to what was written (Ny*size, Nx)

```c++
        bpWriter.BeginStep();
        bpWriter.Put(data, bpFloats.data()); // ADIOS2 knows now that the variable is using the GPU memory space
        bpWriter.EndStep();
```

On read side:
- We don't know that the user wants to read the data on GPU until the Get call which will mean that:
- ADIOS2 will return the Shape according to Layour Right (Ny*size, Nx)
- The buffer will have to flip the dimensions at allocation to be able to receive correct data (if using Layout Left)
