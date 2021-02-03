/*
 * Distributed under the OSI-approved Apache License, Version 2.0.  See
 * accompanying file Copyright.txt for details.
 *
 * File I/O using the GPU Direct library in CUDA (cuFile)
 *
 *  Created on: Jan 25, 2021
 *      Author: Ana Gainaru gainarua@ornl.gov
 */
#include "GPUdirect.h"

#include <cstdio>      // remove
#include <cstring>     // strerror
#include <errno.h>     // errno
#include <fcntl.h>     // open
#include <stddef.h>    // write output
#include <sys/stat.h>  // open, fstat
#include <sys/types.h> // open
#include <unistd.h>    // write, close
#include <iostream>

/// \cond EXCLUDE_FROM_DOXYGEN
#include <ios> //std::ios_base::failure
/// \endcond

namespace adios2
{
namespace transport
{

GPUdirect::GPUdirect(helper::Comm const &comm)
: Transport("File", "GPU", comm)
{
   // TODO: move from here to IO parameters
   const size_t device_id = 1;
   cudaSetDevice(device_id);
   cudaGetDevice(&m_RankGPU);
}

GPUdirect::~GPUdirect()
{
    close(m_FileDescriptor);
}

void GPUdirect::Open(const std::string &name, const Mode openMode,
                     const bool async)
{
    m_IsOpen = false;
    // when delete this, delete include iostream as well
    std::string fname = name + "." + std::to_string(m_RankGPU);

    CUfileError_t status;
    CUfileDescr_t desc;
    memset((void *)&desc, 0, sizeof(CUfileDescr_t));
    m_FileDescriptor = open(fname.c_str(), O_CREAT | O_RDWR | O_DIRECT, 0664);

    desc.handle.fd = m_FileDescriptor;
    desc.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    status = cuFileHandleRegister(&m_GPUFileHandler, &desc);
    if (status.err != CU_FILE_SUCCESS) {
	std::cout << "WARNING! Could not register file " << m_Name
		  << "in call to GDS open. GPU buffers will be ignored" << std::endl;
	return;
    }

    m_IsOpen = true;
}

void GPUdirect::Write(const char *buffer, size_t size, size_t start)
{
    auto lf_Write = [&](const char *buffer, size_t size,
		    size_t fileOffset, size_t bufferOffset){
	errno = 0;
	int ret = cuFileWrite(m_GPUFileHandler, buffer,
			      size, fileOffset, bufferOffset);
	m_Errno = errno;
	if (ret < 0){
	    throw std::ios_base::failure(
		    "ERROR: couldn't write to file " + m_Name +
		    ", in call to GDS Write" + SysErrMsg());
	}
    };

    if (not m_IsOpen)
    {
	std::cout << "WARNING! Skipping writing buffer from the GPU"
		  << " memory space" << std::endl;
	return;
    }

    size_t fileOffset = m_FileOffset;
    if (start != MaxSizeT)
	fileOffset = start;

    if (size > DefaultMaxFileBatchSize)
    {
	const size_t batches = size / DefaultMaxFileBatchSize;
        const size_t remainder = size % DefaultMaxFileBatchSize;

	size_t position = 0;
        for (size_t b = 0; b < batches; ++b)
        {
            lf_Write(buffer, DefaultMaxFileBatchSize, fileOffset, position);
            position += DefaultMaxFileBatchSize;
	    fileOffset += DefaultMaxFileBatchSize;
        }
        lf_Write(buffer, remainder, fileOffset, position);
    }
    else
    {
        lf_Write(buffer, size, fileOffset, 0);
    }

    m_FileOffset += size;

}

void GPUdirect::Read(char *buffer, size_t size, size_t start)
{
    if (not m_IsOpen)
	return;
    std::cout << "GPU read " << size << " bytes" << std::endl;

    m_Errno = cuFileRead(m_GPUFileHandler, (void *) buffer,
		         size, 0, 0);
    if (m_Errno < 0)
        throw std::ios_base::failure(
	    "ERROR: couldn't read from file " + m_Name +
	    ", in call to GDS IO read" + SysErrMsg());
}

size_t GPUdirect::GetSize()
{
    struct stat fileStat;
    errno = 0;
    if (fstat(m_FileDescriptor, &fileStat) == -1)
    {
        m_Errno = errno;
        throw std::ios_base::failure("ERROR: couldn't get size of file " +
                                     m_Name + SysErrMsg());
    }
    m_Errno = errno;
    return static_cast<size_t>(fileStat.st_size);
}

void GPUdirect::Flush() {}

void GPUdirect::Close()
{
    if (not m_IsOpen)
	return;
    
    errno = 0;
    cuFileHandleDeregister(m_GPUFileHandler);
    const int status = close(m_FileDescriptor);
    m_Errno = errno;
    if (status == -1)
    {
        throw std::ios_base::failure("ERROR: couldn't close file " + m_Name +
                                     ", in call to GDS IO close" +
                                     SysErrMsg());
    }

    m_IsOpen = false;
    m_FileOffset = 0;
}

void GPUdirect::Delete()
{
    if (m_IsOpen)
    {
        Close();
    }
    std::remove(m_Name.c_str());
}

void GPUdirect::CheckFile(const std::string hint) const
{
    if (m_FileDescriptor == -1)
    {
        throw std::ios_base::failure("ERROR: " + hint + SysErrMsg());
    }
}

void GPUdirect::SeekToEnd()
{
    m_FileOffset = GetSize();
}

void GPUdirect::SeekToBegin()
{
    m_FileOffset = 0;
}

std::string GPUdirect::SysErrMsg() const
{
    return std::string(": errno = " + std::to_string(m_Errno) + ": " +
                       strerror(m_Errno));
}

} // end namespace transport
} // end namespace adios2
