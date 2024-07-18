CXX = icpx
SYCL_CXXFLAGS = -std=c++17 -fsycl -g -O2 -o 
SYCL_LDFLAGS = 
SYCL_EXE_NAME = matrix_mul_dpc
SYCL_SOURCES = matrix_mul_sycl.cpp

all: build_sycl

build_sycl:
	$(CXX) $(SYCL_CXXFLAGS) $(SYCL_EXE_NAME) $(SYCL_SOURCES) $(SYCL_LDFLAGS)

run: run_sycl

run_sycl:
	./$(SYCL_EXE_NAME)

clean: 
	rm -rf $(SYCL_EXE_NAME)