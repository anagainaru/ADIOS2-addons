#include <Kokkos_Core.hpp>
#include <fstream>
#include <iostream>
#include <unistd.h>
#include <fcntl.h>
#include <iterator>

// needed by cuFile
#include "cufile.h"

// needed by checksums
#include "cufile_sample_utils.h"

using namespace std;

#define MAX_BUF_SIZE (31 * 1024 * 1024UL)

// Write data from GPU memory to a file
int gpu_to_storage(const char *file_name, void *gpumem_buf){
  const size_t size = MAX_BUF_SIZE;
  int fd = open(file_name, O_CREAT | O_RDWR | O_DIRECT, 0664);
if (fd < 0) {
  std::cerr << "write file open error : " << std::strerror(errno)
	  << std::endl;
  return -1;
}

  size_t ret = -1;
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
  std::cout << "GPU direct write memory of size :"
            << size << " gpu id: " << idx << std::endl;
  ret = cuFileWrite(fh, gpumem_buf, size, 0, 0);
  if (ret < 0) {
      std::cerr << "write failed : "
          << cuFileGetErrorString(errno) << std::endl;
      return -1;
   }

  return 0;
}

// Read data from NVME directly to the GPU memory space 
int storage_to_gpu(const char *file_name, void * gpumem_buf)
{
  const size_t size = MAX_BUF_SIZE;
  int fd = open(file_name, O_RDONLY | O_DIRECT);
  if (fd < 0) {
    std::cerr << "read file open error : " << file_name << " "
        << std::strerror(errno) << std::endl;
    return -1;
  }
  
  size_t ret = -1;
  CUfileError_t status;
  CUfileHandle_t fh;
  CUfileDescr_t desc;
  memset((void *)&desc, 0, sizeof(CUfileDescr_t));

  desc.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
  desc.handle.fd = fd;
  status = cuFileHandleRegister(&fh, &desc);
  if (status.err != CU_FILE_SUCCESS) {
    std::cerr << "file register error: "
        << cuFileGetErrorString(status) << std::endl;
    close(fd);
    return -1;
  }

  int idx;
  cudaGetDevice(&idx);
  std::cout << "GPU direct read memory of size :"
            << size << " gpu id: " << idx << std::endl;
  ret = cuFileRead(fh, gpumem_buf, size, 0, 0);
  if (ret < 0) {
    std::cerr << "read failed : "
	<< cuFileGetErrorString(errno) << std::endl;
    cuFileHandleDeregister(fh);
    close(fd);
    return -1;
  }

  cuFileHandleDeregister(fh);
  close (fd);
  return 0;
}

// Write data from CPU memory to a file
int cpu_to_storage(const char *file_name, void *cpumem_buf)
{
  const size_t size = MAX_BUF_SIZE;
  std::ofstream outfs(file_name);
  outfs.write((char *) cpumem_buf, size);
  cout << "CPU Writing memory of size :" << size << std::endl;
  return 0;
}

int main(int argc, char*argv[])
{
  const char *readf = "/mnt/nvme/read.dat";
  const char *writef = "/mnt/nvme/write.dat";
  const char *writef_cpu = "/mnt/nvme/write_cpu.dat";
  
  // Initialize Kokkos to run on GPU with id 3
  int ret, device_id = 3;
  Kokkos::InitArguments args;
  args.device_id = device_id;
  Kokkos::ScopeGuard gds(args);

  // Read data into GPU memory then write it to NVME
  // twice using the GPU and CPU
  const size_t size = MAX_BUF_SIZE;
  Kokkos::View<char*> kokkos_buf( "gpu_buffer", size );
  void *gpumem_buf = (void *) kokkos_buf.data();

  storage_to_gpu(readf, gpumem_buf);
  auto cpu_buf = Kokkos::create_mirror_view_and_copy(
		  Kokkos::HostSpace{}, kokkos_buf);
  gpu_to_storage(writef, gpumem_buf);
  cpu_to_storage(writef_cpu, (void *) cpu_buf.data());

  // Compare file signatures
  unsigned char iDigest[SHA256_DIGEST_LENGTH];
  unsigned char oDigest[SHA256_DIGEST_LENGTH], coDigest[SHA256_DIGEST_LENGTH];
  SHASUM256(readf, iDigest, size);
  DumpSHASUM(iDigest);

  SHASUM256(writef, oDigest, size);
  DumpSHASUM(oDigest);

  SHASUM256(writef_cpu, coDigest, size);
  DumpSHASUM(coDigest);

  if ((memcmp(iDigest, oDigest, SHA256_DIGEST_LENGTH) != 0) ||
      (memcmp(iDigest, coDigest, SHA256_DIGEST_LENGTH) != 0)) {
	std::cerr << "SHA SUM Mismatch" << std::endl;
	ret = -1;
  } else {
	std::cout << "SHA SUM Match" << std::endl;
	ret = 0;
  }
  return ret;
}
