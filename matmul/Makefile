CXX = icpx
CXXFLAGS = -g -O2 -fsycl -std=c++17

TARGETS = matmul_1gpu \
		  matmul_2gpu

SRC_MATMUL_1GPU = matmul_1gpu.cpp
SRC_MATMUL_2GPU = matmul_2gpu.cpp

.PHONY: all clean run

all: $(TARGETS)

matmul_1gpu: $(SRC_MATMUL_1GPU)
	$(CXX) $(CXXFLAGS) -o $@ $<

matmul_2gpu: $(SRC_MATMUL_2GPU)
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)