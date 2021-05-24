#include <Kokkos_Core.hpp>
#include <fstream>
#include <iostream>
#include <unistd.h>
#include <fcntl.h>
#include <iterator>
#include <chrono>

// needed by cuFile
#include "cufile.h"

// needed by checksums
#include "cufile_sample_utils.h"

using namespace std;

// Write data from GPU memory to a file
int gpu_to_storage(const char *file_name, Kokkos::View<float *> kokkos_buf,
		   int N, int nSteps){
  const size_t size = N * sizeof(float);
  void *gpumem_buf = (void *) kokkos_buf.data();
  
  int fd = open(file_name, O_CREAT | O_RDWR | O_DIRECT, 0664);
  if (fd < 0) {
      std::cerr << "write file open error : " << std::strerror(errno)
    	        << std::endl;
      return -1;
  }

  int ret = -1;
  CUfileError_t status;
  CUfileHandle_t fh;
  CUfileDescr_t desc;
  memset((void *)&desc, 0, sizeof(CUfileDescr_t));
  desc.handle.fd = fd;
  desc.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
  status = cuFileHandleRegister(&fh, &desc);
  if (status.err != CU_FILE_SUCCESS) {
	  std::cerr << "file register error: "
      << cuFileGetErrorString(status) << std::endl;
	  close(fd);
	return -1;
  }

  int idx;
  cudaGetDevice(&idx);
  std::cout << "Write data directly from GPU buffers "
            << " gpu id: " << idx << std::endl;
  auto start_time = std::chrono::steady_clock::now();
  auto end_time = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed_time;
  double elapsed_seconds = 0;
  for (int i=0; i < nSteps; i++)
  {
      start_time = std::chrono::steady_clock::now();
      ret = cuFileWrite(fh, gpumem_buf, size, i*size, 0);
      if (ret < 0)
      {
          std::cerr << "write failed : "
              << cuFileGetErrorString(errno) << std::endl;
          continue;
      }
      end_time = std::chrono::steady_clock::now();
      elapsed_time = end_time - start_time;
      elapsed_seconds += elapsed_time.count();

      Kokkos::parallel_for("updateBuffer", N, KOKKOS_LAMBDA(int i){
    	kokkos_buf(i) += 1;
      });
  }
  std::cout << "GDS write time: " << elapsed_seconds << " N "
	    << N << " steps " << nSteps << std::endl;
  return 0;
}

// Write data from CPU memory to a file
int cpu_to_storage(const char *file_name, Kokkos::View<float *> kokkos_buf,
		   int N, int nSteps)
{
  std::cout << "Write data by moving data to CPU space" << std::endl;
  const size_t size = N * sizeof(float);
  std::ofstream outfs(file_name);
  auto start_time = std::chrono::steady_clock::now();
  auto end_time = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed_time;
  double elapsed_seconds = 0;
  for (int i=0; i<nSteps; i++)
  {
      start_time = std::chrono::steady_clock::now();
      auto cpu_buf = Kokkos::create_mirror_view_and_copy(
	  Kokkos::HostSpace{}, kokkos_buf);
      outfs.write((char *) cpu_buf.data(), size);
      end_time = std::chrono::steady_clock::now();
      elapsed_time = end_time - start_time;
      elapsed_seconds += elapsed_time.count();

      Kokkos::parallel_for("updateBuffer", N, KOKKOS_LAMBDA(int i){
    	  kokkos_buf(i) += 1;
      });
  }
  std::cout << "CopyCPU write time: " << elapsed_seconds << " N "
	    << N << " steps " << nSteps << std::endl;
  return 0;
}

int main(int argc, char*argv[])
{
  const char *writef = "/mnt/nvme/write.dat";
  const char *writef_cpu = "/mnt/nvme/write_cpu.dat";
 
  if (argc < 3)
  {
      std::cout << "Usage: " << argv[0] << " array_size simulation_steps"
	        << std::endl;
      return -1;
  }
  const size_t N = atoi(argv[1]);
  int nSteps = atoi(argv[2]);
  
  // Initialize Kokkos to run on GPU with id 1
  int device_id = 1;
  Kokkos::InitArguments args;
  args.device_id = device_id;
  Kokkos::ScopeGuard gds(args);
  int idx;
  cudaGetDevice(&idx);
  std::cout << "GPU direct gpu id: " << idx << std::endl;

  Kokkos::View<float *> kokkos_buf( "gpu_buffer", N);
  // generate data
  Kokkos::parallel_for("initializeBuffer", N, KOKKOS_LAMBDA(int i){
    kokkos_buf(i) = i * 10 / N;
      });

  auto start = std::chrono::steady_clock::now(); 
  gpu_to_storage(writef, kokkos_buf, N, nSteps);
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;
  std::cout << "GDS total simulation time: " << elapsed_seconds.count();
  std::cout << " N " << N << " steps " << nSteps << std::endl;
  std::remove(writef);

  start = std::chrono::steady_clock::now(); 
  cpu_to_storage(writef_cpu, kokkos_buf, N, nSteps);
  end = std::chrono::steady_clock::now();
  elapsed_seconds = end-start;
  std::cout << "CopyCPU total simulation time: " << elapsed_seconds.count();
  std::cout << " N " << N << " steps " << nSteps << std::endl;
  std::remove(writef_cpu);
  return 0;
}
