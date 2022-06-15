#include <Kokkos_Core.hpp>
#include <fstream>
#include <iostream>
#include <unistd.h>
#include <fcntl.h>
#include <iterator>
#include <chrono>
#include <random>

// needed by cuFile
#include "cufile.h"

// needed by file checksums
#include "cufile_sample_utils.h"

using namespace std;

// The minimum memory unit is 1MB
#define MAX_BUF_SIZE (1024 * 1024UL)

// Write data from GPU memory to a file
int gpu_to_storage(const char *file_name, void *gpumem_buf, int mb){
  const size_t size = MAX_BUF_SIZE * mb;
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
  auto start = std::chrono::steady_clock::now();
  ret = cuFileWrite(fh, gpumem_buf, size, 0, 0);
  if (ret < 0) {
      std::cerr << "write failed : "
          << cuFileGetErrorString(errno) << std::endl;
      return -1;
   }

  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;
  std::cout << "GDS write time: " << elapsed_seconds.count()
	    << " size " << size << " rate "
	    << size / ((MAX_BUF_SIZE * 1024) *  elapsed_seconds.count()) 
	    << " GB/s" << std::endl;
	  
  return 0;
}

// Read data from NVME directly to the GPU memory space 
int storage_to_gpu(const char *file_name, void * gpumem_buf, int mb)
{
  const size_t size = MAX_BUF_SIZE * mb;
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
  auto start = std::chrono::steady_clock::now(); 
  ret = cuFileRead(fh, gpumem_buf, size, 0, 0);
  if (ret < 0) {
    std::cerr << "read failed : "
	<< cuFileGetErrorString(errno) << std::endl;
    cuFileHandleDeregister(fh);
    close(fd);
    return -1;
  }
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;
  std::cout << "GDS read time: " << elapsed_seconds.count()
	    << " size " << size << " rate "
	    << size / ((MAX_BUF_SIZE * 1024) *  elapsed_seconds.count()) 
	    << " GB/s" << std::endl;
  cuFileHandleDeregister(fh);
  close (fd);
  return 0;
}

// Read data from storage memory CPU after which move it to GPU memory
Kokkos::View<char*> storage_to_cpu(const char *file_name, int mb)
{
  const size_t size = MAX_BUF_SIZE * mb;
  Kokkos::View<char*, Kokkos::HostSpace> cpumem_buf("cpu_buffer", size);

  int fd = open(file_name, O_RDONLY);
  auto start = std::chrono::steady_clock::now(); 
  size_t ret = read(fd, cpumem_buf.data(), size);
  auto kokkos_buf = Kokkos::create_mirror_view_and_copy(
		  Kokkos::DefaultExecutionSpace::memory_space{},
		  cpumem_buf);
  //Kokkos::DefaultExecutionSpace{}.fence();
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;
  std::cout << "CopyCPU read time: " << elapsed_seconds.count()
	    << " size " << size << " rate "
	    << size / ((MAX_BUF_SIZE * 1024) *  elapsed_seconds.count()) 
	    << " GB/s" << std::endl;

  return kokkos_buf;
}

// Write data from CPU memory to a file
int cpu_to_storage(const char *file_name,
		   Kokkos::View<char*> kokkos_buf, int mb)
{
  const size_t size = MAX_BUF_SIZE * mb;
  int fd = open(file_name, O_CREAT | O_RDWR , 0664);
  auto start = std::chrono::steady_clock::now(); 

  auto cpu_buf = Kokkos::create_mirror_view_and_copy(
		  Kokkos::HostSpace{}, kokkos_buf);

  size_t ret = write(fd, (void *) cpu_buf.data(), size);
  if (ret < size)
  {
	std::cout << "ERROR writting through the CPU" << std::endl;
	return 1;
  }
  fsync(fd);
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;
  std::cout << "CopyCPU write time: " << elapsed_seconds.count()
	    << " size " << size << " rate "
	    << size / ((MAX_BUF_SIZE * 1024) *  elapsed_seconds.count()) 
	    << " GB/s" << std::endl;

  return 0;
}

// Create a file of a given size
int create_read_file(const char* fname, const size_t size)
{
    void *hostPtr;
    Prng prng(255);
    int ret;
    // Create a Test file using standard Posix File IO calls
    int fd = open(fname, O_RDWR | O_CREAT, 0644);
    if (fd < 0) {
            std::cerr << "test file open error : " << fname << " "
                    << std::strerror(errno) << std::endl;
            return -1;
    }

    hostPtr = malloc(size);
    if (!hostPtr) {
            std::cerr << "buffer allocation failure : "
                    << std::strerror(errno) << std::endl;
            close(fd);
            return -1;
    }

    memset(hostPtr, prng.next_random_offset(), size);
    ret = write(fd, hostPtr, size);
    if (ret < 0) {
            std::cerr << "write failure : " << std::strerror(errno)
                            << std::endl;
            close(fd);
            free(hostPtr);
            return -1;
    }

    free(hostPtr);
    close(fd);
    return 0;
}

// check if two GPU buffers have the same content
void match_buffers(Kokkos::View<char*> buf1, Kokkos::View<char*> buf2,
		   size_t size)
{
    int sum = 0;
    Kokkos::parallel_reduce("checkCorrectness", size,
		    	    KOKKOS_LAMBDA(int i, int &error){
        if (buf1(i) != buf2(i)) ++error;
    }, sum);
    if (sum == 0)
	std::cout << "Buffers match" << std::endl;
    else
	std::cout << "Buffers do not match" << std::endl;
}

// Check if the read and write files have the same content
void match_files(const char *readf, const char *writef,
		 const char *writef_cpu, size_t size)
{
  unsigned char iDigest[SHA256_DIGEST_LENGTH];
  unsigned char oDigest[SHA256_DIGEST_LENGTH];
  unsigned char coDigest[SHA256_DIGEST_LENGTH];
  SHASUM256(readf, iDigest, size);
  SHASUM256(writef, oDigest, size);
  SHASUM256(writef_cpu, coDigest, size);

  int ret = 0;
  if (memcmp(iDigest, coDigest, SHA256_DIGEST_LENGTH) != 0)
  {
	std::cerr << "File Mismatch for the CPU write" << std::endl;
	ret = -1;
  }
  if (memcmp(iDigest, oDigest, SHA256_DIGEST_LENGTH) != 0)
  {
	std::cerr << "File Mismatch for GDS" << std::endl;
	ret = -1;
  }
  if(ret==0)
	std::cout << "Files Match" << std::endl;
}

int main(int argc, char*argv[])
{
  if (argc < 2)
  {
      std::cout << "Usage: " << argv[0] << " file_size(MB)"
                << std::endl;
      return -1;
  }
  int mb = atoi(argv[1]);
  const char *readf = "/mnt/nvme/read.dat";
  const char *writef = "/mnt/nvme/write.dat";
  const char *writef_cpu = "/mnt/nvme/write_cpu.dat";

  // Initialize Kokkos to run on GPU with id 1
  int ret = 0, device_id = 1;
  Kokkos::InitArguments args;
  args.device_id = device_id;
  Kokkos::ScopeGuard gds(args);

  const size_t size = MAX_BUF_SIZE * mb;
  // create the file used for performance measurements
  create_read_file(readf, size);

  Kokkos::View<char*> kokkos_buf( "gpu_buffer", size );
  void *gpumem_buf = (void *) kokkos_buf.data();

  // measure READ performance from storage
  // directy to GPU mem space GDS
  storage_to_gpu(readf, gpumem_buf, mb);

  // measure WRITE performance
  // from GPU mem direct to storage GDS
  gpu_to_storage(writef, gpumem_buf, mb);
  // copy buffer to CPU then POSIX
  cpu_to_storage(writef_cpu, kokkos_buf, mb);

  // Compare file signatures
  match_files(readf, writef, writef_cpu, size);

  // measure READ performance from storage
  // POSIX then copy buffer to GPU mem space
  auto gpu_buf = storage_to_cpu(writef, mb);
  
  // compare buffer information on the GPU
  match_buffers(kokkos_buf, gpu_buf, size);
  return ret;
}
