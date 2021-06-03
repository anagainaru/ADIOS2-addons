# GPU aware ADIOS

**Goal of this research:** Use GPU buffers directly with `Put` functions and save one `memcpy` to the ADIOS internal buffers (illustrated in the following figures). 

<img width="734" alt="GPU aware ADIOS" src="https://user-images.githubusercontent.com/16229479/120682528-8e6cbb00-c46a-11eb-93a6-64e034dcaeb6.png">

**Code**

Changes to the code to allow ADIOS to receive buffers allocated in the GPU memory space in the Put function. Code is stored in the https://github.com/anagainaru/ADIOS2 repo in branch `gpu_copy_to_host`. Description of the changes can be found in this folder, here: [code_changes.md](https://github.com/anagainaru/ADIOS2-addons/blob/main/GPUAware/ADIOS.cuda/code_changes.md)

**Performance**

Initial performance results

<img width="500" alt="GPU aware ADIOS results" src="https://user-images.githubusercontent.com/16229479/120683014-1eab0000-c46b-11eb-8ff7-e799fa2db552.png">
