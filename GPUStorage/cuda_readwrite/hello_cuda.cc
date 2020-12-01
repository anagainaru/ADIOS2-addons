#include <iostream>
#include <unistd.h>
#include <fcntl.h>

// needed by cuFile
#include "cufile.h"

// needed by check_cudaruntimecall
#include "cufile_sample_utils.h"

using namespace std;

#define MAX_BUF_SIZE (31 * 1024 * 1024UL)

int gpu_to_storage(const char *file_name, void *gpumem_buf){
  const size_t size = MAX_BUF_SIZE;
  int fd = open(file_name, O_CREAT | O_RDWR | O_DIRECT, 0664);
if (fd < 0) {
  std::cerr << "write file open error : " << std::strerror(errno)
	  << std::endl;
  cudaFree(gpumem_buf);
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
  	cudaFree(gpumem_buf);
	return -1;
  }
      
  ret = cuFileWrite(fh, gpumem_buf, size, 0, 0);
  if (ret < 0) {
      std::cerr << "write failed : "
          << cuFileGetErrorString(errno) << std::endl;
  	cudaFree(gpumem_buf);
      return -1;
   }

  int idx;
  check_cudaruntimecall(cudaGetDevice(&idx));
  std::cout << "Writing memory of size :"
		<< size << " gpu id: " << idx << std::endl;
  return 0;
}

void *storage_to_gpu(const char *file_name)
{
  int device_id = 3;
  const size_t size = MAX_BUF_SIZE;
  check_cudaruntimecall(cudaSetDevice(device_id));
  int fd = open(file_name, O_RDONLY | O_DIRECT);
  if (fd < 0) {
    std::cerr << "read file open error : " << file_name << " "
        << std::strerror(errno) << std::endl;
    return NULL;
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
    return NULL;
  }

  void *gpumem_buf;

  cudaMalloc(&gpumem_buf, size);
  cudaMemset(gpumem_buf, 0, size);
  ret = cuFileRead(fh, gpumem_buf, size, 0, 0);
  if (ret < 0) {
    std::cerr << "read failed : "
	<< cuFileGetErrorString(errno) << std::endl;
    cuFileHandleDeregister(fh);
    close(fd);
    cudaFree(gpumem_buf);
    return NULL;
  }

  int idx;
  check_cudaruntimecall(cudaGetDevice(&idx));
  std::cout << "Allocating and reading memory of size :"
		<< size << " gpu id: " << idx << std::endl;
  cuFileHandleDeregister(fh);
  close (fd);
  return gpumem_buf;
}

int main(int argc, char*argv[])
{
  const char *readf = "/mnt/nvme/read.dat";
  const char *writef = "/mnt/nvme/write.dat";
  void *devPtr = storage_to_gpu(readf);
  int ret = gpu_to_storage(writef, devPtr);
  return ret;
}
