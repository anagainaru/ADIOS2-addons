KOKKOS_PATH = ${HOME}/kokkos/kokkos
KOKKOS_DEVICES = "OpenMP"
EXE_NAME = "hello"

SRC = $(wildcard *.cpp)

default: build
	echo "Start Build"

# Common includes and paths for CUDA
CUDA_PATH = /usr/local/cuda
CUFILE_PATH = /usr/local/cuda-11.1/targets/x86_64-linux/lib/
CXXFLAGS = -Wall
CXXFLAGS += -I$(CUFILE_PATH) -I..
###########################
CUDART_STATIC = -Bstatic -lcudart_static -lrt -lpthread -ldl
CUFILE_LIB = -L$(CUFILE_PATH) -lcufile
LDFLAGS =  $(CUFILE_LIB) $(CUDA_LIB)  $(CUDART_STATIC) -lcrypto -lssl
INSTALL_GDSSAMPLES_PREFIX = /usr/local/gds/samples
###########################

ifneq (,$(findstring Cuda,$(KOKKOS_DEVICES)))
CXX = ${KOKKOS_PATH}/bin/nvcc_wrapper
EXE = ${EXE_NAME}.cuda
KOKKOS_ARCH = "Volta70"
KOKKOS_CUDA_OPTIONS = "enable_lambda"
else
CXX = g++
EXE = ${EXE_NAME}.host
KOKKOS_ARCH = "BDW"
CUDA_LIB = -L$(CUDA_PATH)/lib64/ -lcuda
CXXFLAGS += -I$(CUDA_PATH)/include/ 
endif

CXXFLAGS += -O3
LINK = ${CXX}
LINKFLAGS = $(LDFLAGS)

DEPFLAGS = -M

OBJ = $(SRC:.cpp=.o)
LIB = $(LDFLAGS)

include $(KOKKOS_PATH)/Makefile.kokkos

build: $(EXE)

$(EXE): $(OBJ) $(KOKKOS_LINK_DEPENDS)
	$(LINK) $(KOKKOS_LDFLAGS) $(LINKFLAGS) $(EXTRA_PATH) $(OBJ) $(KOKKOS_LIBS) $(LIB) -o $(EXE)

clean: kokkos-clean
	rm -f *.o *.cuda *.host *.tmp

# Compilation rules

%.o:%.cpp $(KOKKOS_CPP_DEPENDS)
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(EXTRA_INC)  -c $<

test: $(EXE)
	./$(EXE)
