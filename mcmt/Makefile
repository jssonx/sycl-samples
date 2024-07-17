CXX = icpx
OMPFLAGS = -g -O2 -fsycl

TARGET_2_TILE = sycloffload.icpx.intelgpu.2tile
TARGET_NULL_TILE = sycloffload.icpx.intelgpu.nulltile
TARGET_2_GPU_NULL_TILE = sycloffload.icpx.intelgpu.2gpu.nulltile
TARGET_2_GPU_2_TILE = sycloffload.icpx.intelgpu.2gpu.2tile

.PHONY: all clean

all: $(TARGET_2_TILE) $(TARGET_NULL_TILE) $(TARGET_2_GPU_NULL_TILE) $(TARGET_2_GPU_2_TILE)

$(TARGET_2_TILE): main-gpu-1-tile-2.cpp
	$(CXX) $(OMPFLAGS) -o $(TARGET_2_TILE) main-gpu-1-tile-2.cpp

$(TARGET_NULL_TILE): main-gpu-1-tile-null.cpp
	$(CXX) $(OMPFLAGS) -o $(TARGET_NULL_TILE) main-gpu-1-tile-null.cpp

$(TARGET_2_GPU_NULL_TILE): main-gpu-2-tile-null.cpp
	$(CXX) $(OMPFLAGS) -o $(TARGET_2_GPU_NULL_TILE) main-gpu-2-tile-null.cpp

$(TARGET_2_GPU_2_TILE): main-gpu-2-tile-2.cpp
	$(CXX) $(OMPFLAGS) -o $(TARGET_2_GPU_2_TILE) main-gpu-2-tile-2.cpp

clean:
	rm -f $(TARGET_2_TILE) $(TARGET_NULL_TILE) $(TARGET_2_GPU_NULL_TILE) $(TARGET_2_GPU_2_TILE)
