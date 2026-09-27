import os, sys, shutil
import errno
import subprocess
import itertools
import time
import csv
import re
import math
import functools
from io import StringIO
from collections.abc import Iterable, Callable

from pynvml import nvmlInit, nvmlDeviceGetCount

from perf import saxpy_flops, saxpy_memory_accesses, saxpy_transferred
from perf import sgemm_flops, sgemm_memory_accesses

RED = "\x1B[0;31m"
GREEN = "\x1B[0;32m"
YELLOW = "\x1B[0;33m"
BLUE = "\x1B[0;34m"
BOLD = "\x1B[1m"
UNDERLINE = "\x1B[4m"
CLEAR = "\x1B[0m"
UP = "\x1B[%dA"

CUBLAS_STDOUT = "cublas_stdout"
KERNEL_STDOUT = "kernel_stdout"
NUM_SGEMM = 5 # number of SGEMM CUDA kernels (excluding sequential)
ANSI = True # whether we printing ANSI color, to a terminal (must be manually overridden)

# -----------------------------------------------------------------------------------
# Helper utility functions

def is_cuda_available() -> bool:
    """
    Determines if CUDA is available on the given machine.
    
    Returns:
        bool: whether or not CUDA is available on the given machine
    """
    try:
        nvmlInit()
        return nvmlDeviceGetCount() > 0
    except:
        return False

def get_profiler() -> tuple[bool, bool]:
    """
    Determines which profiler to use by checking if nvprof or nsys is available.
    
    Returns:
        tuple[bool, bool]: (use_nvprof, profiler_available)
    """
    for profiler in ["nvprof", "nsys"]:
        try:
            result = subprocess.run(
                ["which", profiler],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            if result.returncode == 0:
                return profiler == "nvprof", True
        except Exception:
            continue
    return False, False

def get_compute_cap() -> str:
    """
    Get the compute capability of the GPU node we are running on.
    
    Returns:
        str: the GPU's compute capability
    """
    return subprocess.getoutput('nvidia-smi --id=$CUDA_VISIBLE_DEVICES '
                                '--query-gpu=compute_cap --format=csv,noheader')

def silent_remove(filename: str):
    """
    Silently removes a file that may or may not exist.
    
    Args:
        filename (str): the path of the file to delete
    """
    try:
        os.remove(filename)
    except OSError as e:
        if e.errno != errno.ENOENT:
            raise

def print_and_count(print_str: str) -> int:
    """
    Prints a string to the terminal and determines how many lines it's printing
    takes up in the terminal window
    
    Args:
        print_str (str): the string to print
    Returns:
        int: the number of lines the string takes up in the terminal window
    """
    width = shutil.get_terminal_size((80, 20)).columns
    
    count = 0
    for line in print_str.splitlines():
        if not line:
            count += 1
        else:
            count += math.ceil(len(line) / width)
    print(print_str)
    return count

# -----------------------------------------------------------------------------------
# Helper tester functions

def profile_cmd(cmd_args: list[str], use_nvprof: bool, op_keywords: list[str]) -> dict[str, float] | None:
    """
    Profiles a given CUDA executable for the runtime of specified kernel(s) and other
    CUDA operations.
    
    Args:
        cmd_args (list[str]): the command-line arguments for the command to be profiled
        use_nvprof (bool): whether to use `nvprof` or `nsys` for profiling
        op_keywords (list[str]): a list of regex patterns to match the name of an operation
        whose runtime is to be profiled
    Returns:
        dict[str, float] | None: on success, a mapping from each regex pattern to 
        the runtime in ms of the first matched operation (additionally containing 
        a mapping from `KERNEL_STDOUT` and `CUBLAS_STDOUT` to the executable's 
        printed kernel and cuBLAS runtime); on failure, None
    """
    keyword_to_runtime = {}
    if use_nvprof:
        OUT_CSV = "/tmp/runtime.csv"
        # use `nvprof` to profile, output info to CSV
        try:
            res = subprocess.run(
                ["nvprof", "--normalized-time-unit", "ms", "--csv", "--force-overwrite", "--log-file", OUT_CSV] + cmd_args,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
        except Exception as e:
            print(f"ERROR: could not run `nvprof`: {e}")
            return None
        if not os.path.isfile(OUT_CSV): # correctness failed
            print("ERROR: `nvprof` did not output a log file")
            return None
        with open(OUT_CSV, errors="ignore") as f:
            output = f.read()
        if output.startswith("======== Error: Application returned non-zero code"):
            print(f"{RED if ANSI else ''}Correctness test failed!{CLEAR if ANSI else ''}")
            silent_remove(OUT_CSV)
            return None
        
        # parse output CSV information
        start_line = re.search(r'==\d+== Profiling result:', output)
        if not start_line:
            print("ERROR: Unable to parse profiler output.")
            print(output)
            silent_remove(OUT_CSV)
            return None
        reader = csv.DictReader(StringIO(output[start_line.end():].strip()))
        for row in reader:
            for keyword in op_keywords:
                if re.search(keyword, row['Name']):
                    keyword_to_runtime[keyword] = float(row['Avg'])
                    break
        silent_remove(OUT_CSV)
    else:
        PROFILE_OUT = "/tmp/profile.nsys-rep"
        SQLITE_OUT = "/tmp/profile.sqlite"
        OUT_PREFIX = "/tmp/out"
        # use `nsys profile` to profile
        try:
            res = subprocess.run(
                ["nsys", "profile", "--force-overwrite", "true", "--output", PROFILE_OUT] + cmd_args,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
        except Exception as e:
            print(f"ERROR: could not run `nsys profile`: {e}")
            return None
        if res.returncode != 0:
            print(f"{RED if ANSI else ''}Correctness test failed!{CLEAR if ANSI else ''}")
            silent_remove(PROFILE_OUT)
            return None
        if not os.path.isfile(PROFILE_OUT):
            print("ERROR: `nsys` profiler did not output a log file")
            silent_remove(PROFILE_OUT)
            return None

        # parse output of `nsys profile` into CSV using `nsys stats`
        try:
            subprocess.run(
                ["nsys", "stats", "--format", "csv", "--report", "cuda_gpu_kern_sum,cuda_gpu_mem_time_sum",
                "--force-export", "--sqlite", SQLITE_OUT, "--force-overwrite", 
                "--output", OUT_PREFIX, PROFILE_OUT],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except Exception as e:
            print(f"ERROR: could not run `nsys stats`: {e}")
            return None
        
        # open and parse CSV outputs
        KERN_OUT = OUT_PREFIX + "_cuda_gpu_kern_sum.csv"
        MEM_OUT = OUT_PREFIX + "_cuda_gpu_mem_time_sum.csv"
        OUTPUT_FILES = [PROFILE_OUT, SQLITE_OUT, KERN_OUT, MEM_OUT]
        if not os.path.isfile(KERN_OUT):
            print("ERROR: `nsys` stats analyzer did not output a kernel analysis log file")
            for filename in OUTPUT_FILES:
                silent_remove(filename)
            return None
        if not os.path.isfile(MEM_OUT):
            print("ERROR: `nsys` stats analyzer did not output a memory transfer analysis log file")
            for filename in OUTPUT_FILES:
                silent_remove(filename)
            return None
        with open(KERN_OUT, errors="ignore") as kern_f, open(MEM_OUT, errors="ignore") as mem_f:
            kernel_reader = csv.DictReader(kern_f)
            mem_reader = csv.DictReader(mem_f)
            for row in kernel_reader:
                for keyword in op_keywords:
                    if re.search(keyword, row['Name']):
                        keyword_to_runtime[keyword] = float(row['Avg (ns)']) * 1e-6
                        break
            for row in mem_reader:
                for keyword in op_keywords:
                    if re.search(keyword, row['Operation']):
                        keyword_to_runtime[keyword] = float(row['Avg (ns)']) * 1e-6
                        break

        for filename in OUTPUT_FILES:
            silent_remove(filename)
    
    # Scrape runtime from stdout
    cublas_match = re.search(r"cuBLAS ran in ([0-9.]+)ms", res.stdout)
    if cublas_match:
        keyword_to_runtime[CUBLAS_STDOUT] = float(cublas_match.group(1))
    kernel_match = re.search(r"Kernel \d ran in ([0-9.]+)ms", res.stdout)
    if kernel_match:
        keyword_to_runtime[KERNEL_STDOUT] = float(kernel_match.group(1))
    
    return keyword_to_runtime


CURR_TEST_NUM = 1

def run_test(test: list[str] | Callable[[], bool], test_name: str) -> bool:
    """
    Runs a given correctness test executable, determining if it passed or not 
    based on its exit code.
    
    Args:
        test (list[str]): either a test command and its arguments to run, or a 
        test function to call that returns whether or not the test passed
        test_name (str): the display name for the test
    Returns:
        bool: whether the test passed or not
    """
    is_cmd = isinstance(test, list)
    global CURR_TEST_NUM
    if ANSI:
        print(f"{CURR_TEST_NUM}. [ ...... ] {YELLOW}{test_name}{CLEAR}")
    
    num_lines = 1
    after_str = ""
    if is_cmd:
        cmd = f"> {' '.join(test)}"
        if ANSI:
            num_lines += print_and_count(cmd)
        else:
            after_str += f"{cmd}\n"
        try:
            passed = subprocess.run(
                test, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            ).returncode == 0
        except Exception as e:
            error_str = f"ERROR: running test encountered exception: {e}"
            if ANSI:
                num_lines += print_and_count(error_str)
            else:
                after_str += f"{error_str}\n"
            passed = False
    else:
        try:
            passed = test()
        except Exception as e:
            error_str = f"ERROR: running test encountered exception: {e}"
            if ANSI:
                num_lines += print_and_count(error_str)
            else:
                after_str += f"{error_str}\n"
            passed = False
    
    if ANSI:
        print(f"{UP % num_lines}{CURR_TEST_NUM}. [ {GREEN if passed else RED}"
              f"{'PASSED' if passed else 'FAILED'}{CLEAR} ]", 
              end=('\n' * num_lines))
    else:
        print(f"{CURR_TEST_NUM}. [ {'PASSED' if passed else 'FAILED'} ] {test_name}")
        print(after_str, end='')
    CURR_TEST_NUM += 1
    return passed

# -----------------------------------------------------------------------------------
# Test suites

def print_title(title: str):
    """
    Prints the title of a test suite.
    
    Args:
        title (str): the test suite's title
    """
    if ANSI:
        print(f"\n{BLUE}==={CLEAR} {YELLOW}{BOLD}{title}{CLEAR} {BLUE}==={CLEAR}")
    else:
        print(f"\n=== {title} ===")

def saxpy_test_suite() -> tuple[int, int]:
    """
    Runs the SAXPY CUDA kernel correctness test suite.
    
    Returns:
        tuple[int, int]: the tuple of tests that passed and total tests ran
    """
    print_title("SAXPY CORRECTNESS TESTS")
    total, correct = 0, 0
    for n in [1024, 4096, 1048576, 1, 1225, 10000000]:
        total += 1
        if run_test(["./saxpy", "-n", str(n)], f"SAXPY correctness with N={n}"):
            correct += 1
    return correct, total

def saxpy_perf_suite(use_nvprof: bool):
    """
    Runs the SAXPY performance test suite
    
    Args:
        use_nvprof (bool): whether to use nvprof or nsys for profiling
    """
    print_title("SAXPY PERFORMANCE")
    
    N = 10000000
    cmd = ["./saxpy", "-n", str(N)]
    print("> ", " ".join(cmd))
    profile_res = profile_cmd(cmd, use_nvprof, ["saxpy", "HtoD", "DtoH"])
    print(profile_res)
    if not profile_res: # profiling or correctness failed, error printed
        return

    kernel, dtoh, htod = profile_res['saxpy'], profile_res['DtoH'], profile_res['HtoD']
    print(f"Kernel ran in {kernel:.3f}ms")
    print(f"Effective compute bandwidth: {saxpy_flops(N) / kernel * 1e-6:.3f} GFLOPs/s")
    print(f"Effective memory bandwidth: {saxpy_memory_accesses(N) / kernel * 1e-6:.3f} GB/s")
    print(f"Effective arithmetic intensity: {saxpy_flops(N) / saxpy_memory_accesses(N):.3f} FLOPs/B")
    print(f"Effective host-to-device memory bandwidth: {saxpy_transferred(N) / htod * 1e-6:.3f} GB/s")
    print(f"Effective device-to-host memory bandwidth: {saxpy_transferred(N) / dtoh * 1e-6:.3f} GB/s")


def sgemm_test_suite(sgemm_nums: Iterable[int], 
                     mkns_sets: list[list[tuple[int, int, int]]]) -> list[list[tuple[int, int]]]:
    """
    Runs the SGEMM CUDA kernel correctness test suite on the given M, K, and N sizes.
    
    Args:
        sgemm_nums (Iterable[int]): the sets of the SGEMM kernels to test
        mkns_sets (list[list[tuple[int, int, int]]]): the sets of M, K, and N values 
        to test correctness with
    Returns:
        list[list[tuple[int, int]]]: a list containing, for each kernel, a list 
        that contains, for each set of values tested, the tuple of tests that 
        passed and total tests ran
    """
    scores: list[list[tuple[int, int]]] = []
    first = True
    for i, sgemm_num in enumerate(sgemm_nums):
        if first:
            print_title("SGEMM CORRECTNESS TESTS")
            first = False
        print(f"{os.linesep if i != 0 else ''}{UNDERLINE if ANSI else ''}"
              f"Tests for Kernel {sgemm_num}{CLEAR if ANSI else ''}:")
        
        kernel_scores: list[tuple[int, int]] = []
        for mkns in mkns_sets:
            total, correct = 0, 0
            for M, K, N in mkns:
                if sgemm_num == 0:
                    M //= 8; K //= 8; N //= 8
                if run_test(["./sgemm", str(sgemm_num), "-M", str(M), "-K", str(K), "-N", str(N)], 
                            f"SGEMM correctness with M={M}, K={K}, N={N}"):
                    correct += 1
                total += 1
            kernel_scores.append((correct, total))
        scores.append(kernel_scores)
    return scores

def sgemm_perf_suite(sgemm_nums: Iterable[int], use_nvprof: bool, 
                     verbose: bool = False) -> list[float | None]:
    """
    Runs the SGEMM CUDA kernel performance test suite.
    
    Args:
        sgemm_nums (Iterable[int]): the SGEMM kernels to test
        use_nvprof (bool): whether to use nvprof or nsys for profiling
        verbose (bool): whether to run tests in verbose mode; defaults to False
    Returns:
        list[float | None]: for each kernel tested, the relative performance
        to cuBLAS or None if the test failed
    """
    first = True
    M = K = N = 1024
    rel_perfs = []
    for sgemm_num in sgemm_nums:
        if first:
            print_title("SGEMM PERFORMANCE")
            first = False
        cmd = ["./sgemm", str(sgemm_num), "-M", str(M), "-K", str(K), "-N", str(N)]
        print("> ", " ".join(cmd))
        kern_iden = rf"sgemm{SGEMM_SUFFIXES[sgemm_num]}" if sgemm_num != 0 else KERNEL_STDOUT
        cublas_iden = rf"sgemm(?!{SGEMM_SUFFIXES[sgemm_num]})" if sgemm_num != 0 else CUBLAS_STDOUT
        # profile kernel using profiler, due to non-negligible overhead of kernel launch time
        # that gets included within in-driver runtime measurements
        profile_res = profile_cmd(cmd, use_nvprof, [kern_iden, cublas_iden])
        
        if profile_res: # profiling or correctness didn't fail
            kernel, cublas = profile_res[kern_iden], profile_res[cublas_iden]
            print(f"Kernel {sgemm_num} ran in {kernel:.3f}ms, "
                f"{BOLD if ANSI else ''}{UNDERLINE if ANSI else ''}"
                f"{kernel/cublas:.2f}x cuBLAS's runtime{CLEAR if ANSI else ''}")
            if verbose:
                print(f"Effective compute bandwidth: {sgemm_flops(M, K, N) / kernel * 1e-6:.3f} GFLOPs/s")
                print(f"Effective memory bandwidth: {sgemm_memory_accesses(M, K, N) / kernel * 1e-6:.3f} GB/s")
                print(f"Effective arithmetic intensity: {sgemm_flops(M, K, N) / sgemm_memory_accesses(M, K, N):.3f} FLOPs/B")
            rel_perfs.append(kernel/cublas)
        else:
            rel_perfs.append(None)
    return rel_perfs


def conv1d_test_suite(bldks: Iterable[tuple[int, int, int, int]]) -> tuple[int, int]:
    """
    Runs the 1D depthwise convolution CUDA kernel correctness test suite on 
    the given dimensions sizes.
    
    Args:
        bldks (Iterable[tuple[int, int, int, int]]): the values of the batch,
        length, depth, and filter length dimensions respectively to test
    Returns:
        tuple[int, int]: the tuple of tests that passed and total tests ran
    """
    print_title("CONV1D CORRECTNESS TESTS")
    REPEATS=1
    total, correct = 0, 0
    for B, L, D, K in bldks:
        if run_test(["./conv1d", "-B", str(B), "-L", str(L), "-D", str(D), "-K", str(K), "-R", str(REPEATS)], 
                    f"1D convolution correctness with B={B}, L={L}, D={D}, K={K}"):
            correct += 1
        total += 1
    return correct, total

def conv1d_perf_suite(bldks: Iterable[tuple[int, int, int, int]]) -> list[float | None]:
    """
    Runs the 1D depthwise convolution kernel performance test suite on the given
    dimension sizes.
    
    Args:
        bldks (Iterable[tuple[int, int, int, int]]): the values of the batch,
        length, depth, and filter length dimensions respectively to test
    Returns:
        list[float | None]: for each test, the relative performance to PyTorch
        or None if the test failed
    """
    print("\nImporting PyTorch...")
    import torch
    from torch.nn import Conv1d
    from prettytable import PrettyTable

    torch.manual_seed(1390)
    torch.set_default_device('cuda')
    torch.set_default_dtype(torch.float32)
    nbytes = 4

    results = PrettyTable()
    results.field_names = ["B", "L", "D", "K", "torch time (ms)", "cuda time (ms)", "speedup", "Effective bandwidth (GB/s)", "TFLOPS"]
    results.float_format = "0.3"

    REPEATS = 20
    rel_perfs = []

    def run_conv1d_test(B: int, L: int, D: int, K: int) -> bool:
        # --- Time PyTorch ---
        u = torch.randn([B, D, L])
        conv1d_torch = Conv1d(

            in_channels=D, out_channels=D,
            kernel_size=K, groups=D, padding=K//2
        )

        # Warm-up
        conv1d_torch(u)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(REPEATS):
            conv1d_torch(u)
        end.record()
        torch.cuda.synchronize()
        torch_time = start.elapsed_time(end) / REPEATS  # ms

        # --- Time CUDA binary ---
        try:
            res = subprocess.run(
                ["./conv1d", "-B", str(B), "-L", str(L), "-D", str(D), "-K", str(K), "-R", str(REPEATS)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
                )
            if res.returncode != 0:
                print(f"Error: conv1d binary failed with return code {res.returncode}")
                rel_perfs.append(None)
                return False
            match = re.search(r"Kernel ran in ([0-9.]+)ms", res.stdout)
            if not match:
                print(f"Error: could not parse kernel time from output:\n{res.stdout}")
                rel_perfs.append(None)
                return False
            cuda_time = float(match.group(1))
        except Exception as e:
            print(f"Error: conv1d binary invocation failed: {e}")
            rel_perfs.append(None)
            return False

        speedup = torch_time / cuda_time
        effective_bandwidth = (B * L * D * 2 + K * D) * nbytes / (cuda_time * 1e-3) / (2 ** 30)
        tera_flops = (B * L * D * 2 * K) / (cuda_time * 1e-3) / (2 ** 40)
        results.add_row([B, L, D, K, torch_time, cuda_time, speedup, effective_bandwidth, tera_flops])

        rel_perfs.append(speedup)
        return True

    print_title("CONV1D PERFORMANCE TESTS")
    all_passed = True
    for B, L, D, K in bldks:
        if not run_test(functools.partial(run_conv1d_test, B, L, D, K), 
        f"1D convolution performance with B={B}, L={L}, D={D}, K={K}"):
            all_passed = False
    if all_passed:
        print("\n1D Convolution Performance:")
        print(results)
    return rel_perfs

NO_COLOR = "--no-color"
OPTIONS = {"all", "saxpy_correct", "saxpy_perf", "sgemm_all", "sgemm_perf", "sgemm0", "sgemm1", 
           "sgemm2", "sgemm3", "sgemm4", "sgemm5", "nondivisible", "conv1d_correct", "conv1d_perf",
           NO_COLOR}
SGEMM_SUFFIXES = {
    1: "_naive", 
    2: "_global_coalescing",
    3: "_shared_mem_cache",
    4: "_1D_thread_tiling",
    5: "_2D_thread_tiling",
}

MKNs: list[tuple[int, int, int]] = [
    (512, 512, 512), (1024, 1024, 1024), (1024, 2048, 4096), (1024, 4096, 2048), 
    (2048, 1024, 4096), (2048, 4096, 1024), (4096, 1024, 2048), (4096, 2048, 1024)
]
NONDIV_MKNs: list[tuple[int, int, int]] = [
    (1025, 1025, 1025), (2047, 2047, 2047), (1000, 2025, 4071), (1000, 4071, 2025), 
    (2025, 1000, 4071), (2025, 4071, 1000), (4071, 1000, 2025), (4071, 2025, 1000)
]

BLDK_CORRECT: Iterable[tuple[int, int, int, int]] = itertools.product(
    [3], [1024, 2000, 4096], [768, 1000, 2048], [3, 5]
)
BLDK_PERF: Iterable[tuple[int, int, int, int]] = itertools.product(
    [1], [4096, 8192], [2048, 8192], [3]
)

WARMUP_PTS = 0.5
SGEMM_PTS = [0.25, 0.5, 0.5, 1.0, 1.25, 1.25]
SGEMM_NONDIV_PTS: list[float] = [0, 0, 0, 0.25, 0.25, 0.25]
SGEMM_PERF_FASTER_PTS = [0, 0, 1, 2, 2, 2]
SGEMM_PERF_RANGE: list[tuple[float, float]] = [(0, math.inf), (0, math.inf), (8, 15), (5, 8), (3, 5), (1, 3)]
SGEMM_PERF_IN_RANGE_PTS = [0, 0, 3, 5, 5, 5]
CONV_PTS, CONV_ALL_PTS = 1, 2
CONV_PERF_PTS, CONV_PERF_CUTOFF = 4, 1.2

if __name__ == "__main__":
    args = set(sys.argv[1:])
    if NO_COLOR in args:
        ANSI = False
        args.remove(NO_COLOR)
    
    if len(args) == 0:
        args = {"all"}
        
    if not is_cuda_available():
        print("ERROR: Could not find an installation of CUDA. Please ensure that you are running on a Hydra GPU node using the `interact` script.")
        sys.exit(1)
    use_nvprof, profiler_available = get_profiler()
    if not profiler_available:
        print("ERROR: neither nvprof nor nsys found on this system.")
        sys.exit(1)
    
    invalid = args.difference(OPTIONS)
    if invalid:
        if len(invalid) == 1:
            print(f"ERROR: Received invalid test option: {invalid.pop()}")
        else:
            print(f"ERROR: Received invalid test options: {list(invalid)}")
        sys.exit(1)
    
    if "saxpy_correct" in args or "all" in args:
        saxpy_scores = saxpy_test_suite()
    
    if "saxpy_perf" in args or "all" in args:
        saxpy_perf_suite(use_nvprof)
    
    # sgemm => run all tests for all kernels, no perf
    # sgemm# => run divisible tests for just those kernels, with perf displayed verbosely
    # sgemm_perf => run just perf tests for all, non-verbosely (overrides sgemm#'s perf display)
    # nondivisible => run nondivisible tests for all kernels, no perf
    spec_sgemms = sorted([int(arg[-1]) for arg in args if re.fullmatch(r'sgemm\d', arg)])
    sgemms = []
    if "all" in args or "sgemm_all" in args:
        sgemms = range(NUM_SGEMM+1)
        mkns = [MKNs, NONDIV_MKNs]
    else:
        mkns = []
        if len(spec_sgemms) > 0:
            sgemms = spec_sgemms
            mkns.append(MKNs)
        if "nondivisible" in args:
            sgemms = range(NUM_SGEMM+1)
            mkns.append(NONDIV_MKNs)
    sgemm_scores = sgemm_test_suite(sgemms, mkns)
    
    run_all_perf = "all" in args or "sgemm_perf" in args
    perf_sgemms = range(NUM_SGEMM+1) if run_all_perf else spec_sgemms
    sgemm_perfs = sgemm_perf_suite(perf_sgemms, use_nvprof, not run_all_perf)
    
    if "all" in args or "conv1d_correct" in args:
        conv1d_score = conv1d_test_suite(BLDK_CORRECT)
    if "all" in args or "conv1d_perf" in args:
        conv1d_perfs = conv1d_perf_suite(BLDK_PERF)
    
    if "all" in args:
        warmup_pts = saxpy_scores[0] * WARMUP_PTS
        warmup_total = saxpy_scores[1] * WARMUP_PTS
        print(f"\nWarm-up: {saxpy_scores[0]}/{saxpy_scores[1]} tests passed"
              f" ({warmup_pts}/{warmup_total})")
        
        part1_pts, part1_total = 0, 0
        part1_perf_pts, part1_perf_total = 0, 0
        for i in range(NUM_SGEMM+1):
            print(f"Kernel {i}: {sgemm_scores[i][0][0]}/{sgemm_scores[i][0][1]}"
                  f" tests passed ({sgemm_scores[i][0][0] * SGEMM_PTS[i]}/"
                  f"{sgemm_scores[i][0][1] * SGEMM_PTS[i]}), ", end="")
            part1_pts += sgemm_scores[i][0][0] * SGEMM_PTS[i]
            part1_total += sgemm_scores[i][0][1] * SGEMM_PTS[i]
            
            perf_total = SGEMM_PERF_FASTER_PTS[i] + SGEMM_PERF_IN_RANGE_PTS[i]
            if sgemm_perfs[i] is not None:
                perf_pts = 0
                if i == 0 or sgemm_perfs[i-1] is None or sgemm_perfs[i] < sgemm_perfs[i-1]:
                    perf_pts += SGEMM_PERF_FASTER_PTS[i]
                if sgemm_perfs[i] <= SGEMM_PERF_RANGE[i][1]: 
                    # less than high of expected range, get full points
                    perf_pts += SGEMM_PERF_IN_RANGE_PTS[i]
                else:
                    # otherwise, linear scale of total points, up to size of range above max expected
                    range_size = SGEMM_PERF_RANGE[i][1] - SGEMM_PERF_RANGE[i][0]
                    percent_above = (sgemm_perfs[i] - SGEMM_PERF_RANGE[i][1]) / range_size
                    perf_pts += round(max(0, 1 - percent_above) * SGEMM_PERF_IN_RANGE_PTS[i], 1)
                print(f"{sgemm_perfs[i]:.2f}x slower than cuBLAS ({perf_pts}/{perf_total})")
                part1_perf_pts += perf_pts
            else:
                print(f"performance test failed (0/{perf_total})")
            part1_perf_total += perf_total
        
        nondiv_passed = sum(score[1][0] for score in sgemm_scores)
        nondiv_tests = sum(score[1][1] for score in sgemm_scores)
        nondiv_pts = sum(score[1][0] * pts for score, pts in zip(sgemm_scores, SGEMM_NONDIV_PTS))
        nondiv_total = sum(score[1][1] * pts for score, pts in zip(sgemm_scores, SGEMM_NONDIV_PTS))
        print(f"Nondivisible matrix sizes: {nondiv_passed}/{nondiv_tests} tests "
              f"passed ({nondiv_pts}/{nondiv_total})")
        part1_pts += nondiv_pts
        part1_total += nondiv_total
        print(f"Part 1 correctness: {part1_pts}/{part1_total} pts")
        print(f"Part 1 performance: {round(part1_perf_pts, 1)}/{part1_perf_total} pts")
        print(f"Warm-up and Part 1: {round(part1_pts + part1_perf_pts, 1)}/{part1_total + part1_perf_total} pts\n")
        
        conv_pts = conv1d_score[0] * CONV_PTS + CONV_ALL_PTS * (conv1d_score[0] == conv1d_score[1])
        conv_total = conv1d_score[1] * CONV_PTS + CONV_ALL_PTS
        print(f"1D convolution: {conv1d_score[0]}/{conv1d_score[1]} tests passed"
              f" ({conv_pts}/{conv_total})")
        
        conv_perf_pts = sum((max(min(CONV_PERF_CUTOFF, perf), 1) - 1) / (CONV_PERF_CUTOFF - 1) * CONV_PERF_PTS 
                            for perf in conv1d_perfs if perf is not None)
        conv_perf_total = len(conv1d_perfs) * CONV_PERF_PTS
        conv1d_perfs_print = ["Failed" if perf is None else f"{perf:.2f}" for perf in conv1d_perfs]
        print(f"1D convolution performance gaps with PyTorch: [{', '.join(conv1d_perfs_print)}]"
              f" ({round(conv_perf_pts, 1)}/{conv_perf_total})")
        print(f"Part 2: {round(conv_pts + conv_perf_pts, 1)}/{conv_total + conv_perf_total} pts")
        
