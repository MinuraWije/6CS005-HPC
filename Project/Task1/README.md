# Task 1 – Word Occurrence Counting (Pthreads)

Counts how often each word appears in a text file, using a number of POSIX
threads given on the command line.

## Files
- `wordcount.c` – the program
- `WordOccurrenceDataset.txt` – the input data (120,000 words)
- `result.txt` – generated output

## Build
Local MinGW gcc (or any gcc with POSIX threads):
```
gcc -O2 -o wordcount wordcount.c -lpthread
```

## Run
```
./wordcount <input_file> <num_threads>
```
Example:
```
./wordcount WordOccurrenceDataset.txt 4
```
The results are written to `result.txt`.

## How it works
1. The file is read in two passes (as in the Week 3 `OMPrime1.c`): the first
   pass counts the words with a `getc` loop, the second reads them into a
   `malloc`'d array. Each word is folded to lower case.
2. The words are sliced between the threads **by index**, using the exact
   scheme from `week1.c`: each thread gets `total / threads` words, and the
   first `total % threads` threads take one extra. No word is ever split.
3. Each thread counts its own slice into a **local** table (linear search),
   then merges that table into one **shared** table inside a
   `pthread_mutex` critical section. Locking once per thread keeps the
   critical section small.
4. Each thread also returns how many words it processed via `pthread_exit`;
   `main` sums these as a check against the total.
5. The shared table is sorted (count descending, ties alphabetical) with a
   nested-loop sort and written to `result.txt`.

## Checks performed
- Output is identical for 1, 4 and 8 threads (only the header's thread count
  differs).
- 94 unique words, counts summing to 120,000, cross-checked with
  `sort | uniq -c`.
- Edge cases: more threads than words, a missing file, missing arguments and
  a thread count of 0 all handled.
