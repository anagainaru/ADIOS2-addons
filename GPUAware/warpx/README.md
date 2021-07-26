# WarpX

## Libraries

WarpX is using AMReX, openPMD and ADIOS.

- AMReX is a framework for block-structured adaptive mesh refinement ([link](https://github.com/AMReX-Codes/amrex))
- OpenPMD is the Open Standard for Particle-Mesh Data Files ([link](https://github.com/openPMD/openPMD-api))

### OpenPMD

openPMD is an open meta-data schema that provides meaning and self-description for data sets in science and engineering.
See [the openPMD standard](https://github.com/openPMD/openPMD-standard) for details of this schema.

This library provides a reference API for openPMD data handling. Since openPMD is a schema (or markup) on top of portable, hierarchical file formats, this library implements various backends such as HDF5, ADIOS1, ADIOS2 and JSON. 

**GPU buffers**

Using the example in `openPMD-api/examples/8a_benchmark_write_parallel.cpp` to test if openPMD works with GPU buffers.


## Data

GPU data question: data block on GPU is the entire block in output or do you slice and dice or reorganize it in the GPU->CPU->ADIOS buffering pipeline?
Meshes: PIConGPU with strides; WarpX/AMReX blocks are good:
https://github.com/ECP-WarpX/WarpX/blob/7d1fe27c286ca4ad870715353caf780a8e1c1fe6/Source/Diagnostics/WarpXOpenPMD.cpp#L995-L997

PIConGPU particles: nope (AoSoA)
WarpX/AMReX particles: for a subset of attributes (SoA)
Good kind: SoA
https://github.com/ECP-WarpX/WarpX/blob/7d1fe27c286ca4ad870715353caf780a8e1c1fe6/Source/Diagnostics/WarpXOpenPMD.cpp#L704-L714
Others are AoS :(
https://github.com/ECP-WarpX/WarpX/blob/7d1fe27c286ca4ad870715353caf780a8e1c1fe6/Source/Diagnostics/WarpXOpenPMD.cpp#L494-L506
Particles in PIConGPU:
https://github.com/ComputationalRadiationPhysics/picongpu/blob/131aad5211464c05cf240232dc1717e45a046a8e/include/picongpu/plugins/openPMD/writer/ParticleAttribute.hpp#L139
Mapped memory to concat or copy & concat
(the two strategies for data preparation are here:
https://github.com/ComputationalRadiationPhysics/picongpu/blob/131aad5211464c05cf240232dc1717e45a046a8e/include/picongpu/plugins/openPMD/WriteSpecies.hpp#L128
And here:
https://github.com/ComputationalRadiationPhysics/picongpu/blob/131aad5211464c05cf240232dc1717e45a046a8e/include/picongpu/plugins/openPMD/WriteSpecies.hpp#L198
)
Meshes in PIConGPU:
https://github.com/ComputationalRadiationPhysics/picongpu/blob/131aad5211464c05cf240232dc1717e45a046a8e/include/picongpu/plugins/openPMD/openPMDWriter.hpp#L1069

