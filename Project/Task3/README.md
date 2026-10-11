# Task 3 – Password Cracking (CUDA)

Reads a file of encrypted passwords and recovers each original password on the
GPU. A raw password is 2 lowercase letters followed by 2 digits
(26·26·10·10 = 67,600 possibilities). The lecturer's `cudaCrypt` turns each
raw password into a 10-character encrypted string; this program reverses it by
brute force, one GPU thread per password.

## Files
- `crack.cu` – the CUDA program
- `passgen.c` – test-data generator (adapted from the lecturer's
  `PasswordGeneratorToText2.c`)
- `passwords.txt` – encrypted passwords (the program input)
- `passwords_plain.txt` – the matching raw passwords (only to check results)

## Build & run — Google Colab (GPU runtime)
This task needs `nvcc`, which is not installed locally. Use Colab exactly as
in HPC-8/9/11:

Cell 1:
```
!nvcc --version
```
Cell 2 (upload `crack.cu` and `passwords.txt`, or recreate `passwords.txt`
with `passgen`):
```
!nvcc -o crack crack.cu
!./crack passwords.txt
```
The results are written to `decrypted.txt`, one `encrypted -> decrypted` line
per password.

## Generating test data (local gcc)
```
gcc -O2 -o passgen passgen.c
./passgen 10000        # writes passwords.txt and passwords_plain.txt
```

## How it works
1. **Host read** (two passes, as in `OMPrime1.c`): count the lines with a
   `getc` loop, `malloc` a host buffer, then `fscanf("%10s")` each line.
2. **GPU memory** sized to the file: `cudaMalloc` `d_enc` (`N*11`) and `d_dec`
   (`N*5`), then `cudaMemcpy` host→device. Every CUDA call is checked with the
   `CUDA_CHECK` macro, which prints `cudaGetErrorString` and exits on failure.
3. **`__device__ cudaCrypt`** is the lecturer's function, changed only to
   write into a caller-supplied buffer instead of a `static` one (a `static`
   buffer would be shared by every GPU thread).
4. **Kernel**: one thread per encrypted password. The thread loops over all
   67,600 candidates, encrypts each, compares all 10 characters, and on a
   match writes the 4-character raw password to `d_dec`.
5. **Launch**: `threads = 1000` (or `N` if smaller),
   `blocks = (N + threads - 1) / threads` (the HPC-9 formula).
6. Copy results back, write `decrypted.txt`, then `cudaFree` and `free`
   everything.

## Checks performed
The kernel logic was mirrored on the CPU and run against `passgen` output for
N = 1,000: every recovered password matched `passwords_plain.txt` exactly.
(If two raw passwords ever encrypt to the same string, the kernel returns the
first match; the plaintext comparison would reveal it.)
