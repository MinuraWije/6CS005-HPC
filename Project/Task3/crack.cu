#include <stdio.h>
#include <stdlib.h>

// Task 3 - password cracking with CUDA.
// Usage: ./crack <encrypted_file>   Output: decrypted.txt
// Each input line is a 10-char password from the lecturer's cudaCrypt (2 lowercase
// letters + 2 digits = 26*26*10*10 = 67600 candidates). One GPU thread per line
// brute-forces every candidate and writes back the 4-char original on a match.

#define RAW_LEN  4
#define DEC_LEN  5      // raw + '\0'
#define ENC_LEN 10
#define ENC_BUF 11      // encrypted + '\0'

#define CUDA_CHECK(call)                                                     \
    do {                                                                    \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));         \
            exit(1);                                                       \
        }                                                                 \
    } while (0)

// Device version of the lecturer's cudaCrypt; writes into a caller buffer so
// each thread has its own (the original used a shared static buffer).
__device__ void cudaCrypt(const char *rawPassword, char *newPassword) {
    newPassword[0] = rawPassword[0] + 2;
    newPassword[1] = rawPassword[0] - 2;
    newPassword[2] = rawPassword[0] + 1;
    newPassword[3] = rawPassword[1] + 3;
    newPassword[4] = rawPassword[1] - 3;
    newPassword[5] = rawPassword[1] - 1;
    newPassword[6] = rawPassword[2] + 2;
    newPassword[7] = rawPassword[2] - 2;
    newPassword[8] = rawPassword[3] + 4;
    newPassword[9] = rawPassword[3] - 4;
    newPassword[10] = '\0';

    for (int i = 0; i < 10; i++) {
        if (i >= 0 && i < 6) {
            if (newPassword[i] > 122) {
                newPassword[i] = (newPassword[i] - 122) + 97;
            } else if (newPassword[i] < 97) {
                newPassword[i] = (97 - newPassword[i]) + 97;
            }
        } else {
            if (newPassword[i] > 57) {
                newPassword[i] = (newPassword[i] - 57) + 48;
            } else if (newPassword[i] < 48) {
                newPassword[i] = (48 - newPassword[i]) + 48;
            }
        }
    }
}

__global__ void crackKernel(const char *d_enc, char *d_dec, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const char *target = &d_enc[idx * ENC_BUF];

    char candidate[RAW_LEN];
    char attempt[ENC_BUF];

    for (char a = 'a'; a <= 'z'; a++) {
        for (char b = 'a'; b <= 'z'; b++) {
            for (char c = '0'; c <= '9'; c++) {
                for (char d = '0'; d <= '9'; d++) {
                    candidate[0] = a;
                    candidate[1] = b;
                    candidate[2] = c;
                    candidate[3] = d;

                    cudaCrypt(candidate, attempt);

                    int match = 1;
                    for (int k = 0; k < ENC_LEN; k++) {
                        if (attempt[k] != target[k]) {
                            match = 0;
                            break;
                        }
                    }
                    if (match) {
                        d_dec[idx * DEC_LEN + 0] = a;
                        d_dec[idx * DEC_LEN + 1] = b;
                        d_dec[idx * DEC_LEN + 2] = c;
                        d_dec[idx * DEC_LEN + 3] = d;
                        d_dec[idx * DEC_LEN + 4] = '\0';
                        return;
                    }
                }
            }
        }
    }
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <encrypted_file>\n", argv[0]);
        return 1;
    }
    char *filename = argv[1];

    // Pass 1: count the lines.
    FILE *fp = fopen(filename, "r");
    if (fp == NULL) {
        fprintf(stderr, "Could not open file: %s\n", filename);
        return 1;
    }
    int n = 0;
    int ch;
    for (ch = getc(fp); ch != EOF; ch = getc(fp)) {
        if (ch == '\n') n++;
    }
    fclose(fp);
    if (n == 0) {
        fprintf(stderr, "File has no passwords\n");
        return 1;
    }

    // Pass 2: read the encrypted passwords into a host buffer.
    char *h_enc = (char *)malloc((size_t)n * ENC_BUF);
    char *h_dec = (char *)malloc((size_t)n * DEC_LEN);
    if (h_enc == NULL || h_dec == NULL) {
        fprintf(stderr, "Host allocation failed\n");
        free(h_enc); free(h_dec);
        return 1;
    }
    fp = fopen(filename, "r");
    if (fp == NULL) {
        fprintf(stderr, "Could not re-open file: %s\n", filename);
        free(h_enc); free(h_dec);
        return 1;
    }
    for (int i = 0; i < n; i++) {
        if (fscanf(fp, "%10s", &h_enc[i * ENC_BUF]) != 1) {
            fprintf(stderr, "Unexpected end of file at line %d\n", i);
            fclose(fp);
            free(h_enc); free(h_dec);
            return 1;
        }
    }
    fclose(fp);

    // GPU memory sized to the file.
    char *d_enc = NULL;
    char *d_dec = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_enc, (size_t)n * ENC_BUF));
    CUDA_CHECK(cudaMalloc((void **)&d_dec, (size_t)n * DEC_LEN));
    CUDA_CHECK(cudaMemcpy(d_enc, h_enc, (size_t)n * ENC_BUF,
                          cudaMemcpyHostToDevice));

    // Split blocks/threads by password count so each thread cracks one password.
    int threads = 1000;
    if (threads > n) threads = n;
    int blocks = (n + threads - 1) / threads;
    printf("Cracking %d passwords with %d blocks x %d threads\n",
           n, blocks, threads);

    crackKernel<<<blocks, threads>>>(d_enc, d_dec, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_dec, d_dec, (size_t)n * DEC_LEN,
                          cudaMemcpyDeviceToHost));

    FILE *out = fopen("decrypted.txt", "w");
    if (out == NULL) {
        fprintf(stderr, "Could not open decrypted.txt for writing\n");
        cudaFree(d_enc); cudaFree(d_dec);
        free(h_enc); free(h_dec);
        return 1;
    }
    for (int i = 0; i < n; i++) {
        fprintf(out, "%s -> %s\n", &h_enc[i * ENC_BUF], &h_dec[i * DEC_LEN]);
    }
    fclose(out);
    printf("Wrote decrypted.txt\n");

    CUDA_CHECK(cudaFree(d_enc));
    CUDA_CHECK(cudaFree(d_dec));
    free(h_enc);
    free(h_dec);
    return 0;
}
