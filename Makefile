CC = nvcc -ccbin g++-13
CCFLAGS = -O3 --compiler-options="-g3 -Wall -Wextra -Werror -Wno-unused-function" \
		  -gencode arch=compute_75,code=sm_75 \
		  -gencode arch=compute_86,code=sm_86 \
		  -gencode arch=compute_89,code=sm_89
# add debugging information, at the cost of speed
# CCFLAGS += -g -G
# get verbose output from the PTX optimizing assembler
# CCFLAGS += --ptxas-options=-v

EXECS = saxpy sgemm conv1d query_config
# rebuild when compiler flags or shared headers change
DEPS = Makefile cuda_error.cuh

.PHONY: all clean check

all: $(EXECS)

saxpy: vector_addition/saxpy.cu $(DEPS)
	$(CC) $(CCFLAGS) $< -o $@

sgemm: matrix_multiply/sgemm.cu matrix_multiply/kernels.cuh $(DEPS)
	$(CC) $(CCFLAGS) $< -o $@ -lcublas

conv1d: depth_conv1d/conv1d_cuda.cu depth_conv1d/conv1d_kernel.cuh $(DEPS)
	$(CC) $(CCFLAGS) $< -o $@

query_config: query_config.cu $(DEPS)
	$(CC) $(CCFLAGS) $< -o $@
	cp $@ ../stencil

# See here for more: https://stackoverflow.com/a/14061796.
ifeq (check,$(firstword $(MAKECMDGOALS))) # if the first argument is "check",
  # use the rest as arguments for "check", and turn into do-nothing targets
  CHECK_ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
  $(eval $(CHECK_ARGS):;@:)
endif

check: all
	@echo "uv run python3 run_tests.py $(CHECK_ARGS)"
	@bash -c "uv run python3 run_tests.py $(CHECK_ARGS)"

clean:
	rm -f $(EXECS)
