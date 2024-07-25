SRCS	= ./single.cc ./syclgpu.cc ./syclgpu2.cc ./minitest.cc
TARGET	= single.sycloffload.icpx.intelgpu${TARGET_SUFFIX}

default all: $(TARGET)

CXX = icpx
OMPFLAGS = -g -O2 -fsycl -fopenmp -lm -qopenmp -fopenmp-targets=spir64

$(TARGET): ${SRCS}
	$(CXX) $(OMPFLAGS) -o $(TARGET) ${SRCS}
	@echo ""

