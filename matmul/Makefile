CXX = icpx
CXXFLAGS = -g -O2 -fsycl -std=c++17 -pthread

TARGETS = matmul_xgpu \
		  matmul_xgpu_t \
		  matmul_1gpu_2sub

SRC_MATMUL_XGPU = matmul_xgpu.cpp
SRC_MATMUL_XGPU_T = matmul_xgpu_t.cpp
SRC_MATMUL_1GPU_2SUB = matmul_1gpu_2sub.cpp

.PHONY: all clean run

all: $(TARGETS)

matmul_xgpu_t: $(SRC_MATMUL_XGPU_T)
	$(CXX) $(CXXFLAGS) -o $@ $<

matmul_xgpu: $(SRC_MATMUL_XGPU)
	$(CXX) $(CXXFLAGS) -o $@ $<

matmul_1gpu_2sub: $(SRC_MATMUL_1GPU_2SUB)
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)