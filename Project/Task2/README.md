# Task 2 – Matrix Operations (OpenMP)

Reads pairs of matrices from a file and computes, for each pair: addition,
subtraction, element-wise multiplication, element-wise division (NaN on
division by zero), the transpose of each matrix, and the matrix product
A×B. The work is parallelised with OpenMP.

## Files
- `matrices.c` – the program
- `MatData.txt` – the input data (50 matrices = 25 pairs)
- `results.txt` – generated output

## Build
Local MinGW gcc (or any gcc with OpenMP):
```
gcc -O2 -fopenmp -o matrices matrices.c -lm
```

## Run
```
./matrices <input_file> <num_threads>
```
Example:
```
./matrices MatData.txt 4
```
The results are written to `results.txt`.

## Input format
Each matrix is a `rows,cols` header line, then `rows` lines of
comma-separated values (the `MatData.txt` / `MatAdd.c` format). Matrices are
read in pairs (A, B).

## How it works
- Matrices are `double **` with a per-row `malloc`; `freeMatrix` frees every
  row then the row array.
- Parsing uses `fscanf("%d,%d")` for the header and `fscanf("%lf,")` /
  `fscanf("%lf")` for the values, with every return value checked. A bad
  header, a short row or a non-numeric token is reported to `stderr` and the
  program stops cleanly.
- An operation that the shapes do not allow prints a one-line message and the
  rest of the operations still run.
- The thread count is **capped** at the largest relevant dimension before
  `omp_set_num_threads`. Element-wise operations and transposes use
  `#pragma omp parallel for collapse(2)`; the multiply parallelises the row
  loop with `private(j, k, tot)`. Each operation is timed with
  `omp_get_wtime`.

## Checks performed
- A small hand-made file gives the exact expected values (including NaN for a
  zero divisor and a correct A×B).
- `results.txt` is identical for 1 and 8 threads.
- Error files (missing file, bad header, short row, odd number of matrices)
  are all handled.
