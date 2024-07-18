CXX = icpx
CXXFLAGS = -g -O2 -fsycl

# Define target names
TARGETS = sycl_kernel_1gpu \
          sycl_kernel_2gpu \
          sycl_kernel_1gpu_2tile \
          sycl_kernel_2gpu_2tile

# Source files
SRC_1GPU = sycl_kernel_1gpu.cpp
SRC_2GPU = sycl_kernel_2gpu.cpp
SRC_1GPU_2TILE = sycl_kernel_1gpu_2tile.cpp
SRC_2GPU_2TILE = sycl_kernel_2gpu_2tile.cpp

.PHONY: all clean

all: $(TARGETS)

sycl_kernel_1gpu: $(SRC_1GPU)
	$(CXX) $(CXXFLAGS) -o $@ $<

sycl_kernel_2gpu: $(SRC_2GPU)
	$(CXX) $(CXXFLAGS) -o $@ $<

sycl_kernel_1gpu_2tile: $(SRC_1GPU_2TILE)
	$(CXX) $(CXXFLAGS) -o $@ $<

sycl_kernel_2gpu_2tile: $(SRC_2GPU_2TILE)
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)