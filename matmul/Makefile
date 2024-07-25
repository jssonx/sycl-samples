CXX = icpx
CXXFLAGS = -g -O2 -fsycl -std=c++17

TARGETS = matmul_xgpu \
		  matmul_1gpu_2sub

SRC_MATMUL_1GPU_2SUB = matmul_1gpu_2sub.cpp
SRC_MATMUL_XGPU = matmul_xgpu.cpp

.PHONY: all clean run

all: $(TARGETS)

matmul_xgpu: $(SRC_MATMUL_XGPU)
	$(CXX) $(CXXFLAGS) -o $@ $<

matmul_1gpu_2sub: $(SRC_MATMUL_1GPU_2SUB)
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)