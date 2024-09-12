# Common source files
COMMON_SRCS = ./func.cc ./common.cc

# OpenMP specific files
OMP_SRCS = ./main.cc $(COMMON_SRCS)
OMP_TARGET = omp.sycloffload.icpx.intelgpu${TARGET_SUFFIX}
OMP_CXX = icpx
OMP_FLAGS = -g -O2 -fsycl -fopenmp -lm -qopenmp -fopenmp-targets=spir64

# MPI specific files
MPI_SRCS = ./main_mpi.cc $(COMMON_SRCS)
MPI_TARGET = mpi.sycloffload.icpx.intelgpu${TARGET_SUFFIX}
MPI_CXX = icpx
MPI_FLAGS = -g -O2 -fsycl -lm
MPI_LDFLAGS = -lmpi

# Default target
default: $(OMP_TARGET)

# All targets
all: $(OMP_TARGET) $(MPI_TARGET)

# OpenMP build
$(OMP_TARGET): ${OMP_SRCS}
	$(OMP_CXX) $(OMP_FLAGS) -o $(OMP_TARGET) ${OMP_SRCS}
	@echo "Built OpenMP target: $(OMP_TARGET)"

# MPI build
$(MPI_TARGET): ${MPI_SRCS}
	$(MPI_CXX) $(MPI_FLAGS) -o $(MPI_TARGET) ${MPI_SRCS} $(MPI_LDFLAGS)
	@echo "Built MPI target: $(MPI_TARGET)"

# Clean
clean:
	rm -f $(OMP_TARGET) $(MPI_TARGET)

.PHONY: default all clean