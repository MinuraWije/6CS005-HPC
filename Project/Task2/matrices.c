#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

// Task 2 - matrix operations with OpenMP.
// Usage: ./matrices <input_file> <num_threads>   Output: results.txt
// Matrices are read as "rows,cols" headers + comma-separated rows and processed
// in pairs: A+B, A-B, A.*B, A./B, transpose(A), transpose(B), A*B.

struct Matrix {
    int rows;
    int cols;
    double **data;
};

double **allocMatrix(int rows, int cols) {
    double **m = malloc(rows * sizeof(double *));
    if (m == NULL) return NULL;
    for (int r = 0; r < rows; r++) {
        m[r] = malloc(cols * sizeof(double));
        if (m[r] == NULL) {
            for (int k = 0; k < r; k++) free(m[k]);
            free(m);
            return NULL;
        }
    }
    return m;
}

void freeMatrix(double **m, int rows) {
    if (m == NULL) return;
    for (int r = 0; r < rows; r++) {
        free(m[r]);
    }
    free(m);
}

// Returns 1 on success, 0 at clean end of file, -1 on a parse error.
int readMatrix(FILE *fp, struct Matrix *out) {
    int rows, cols;
    int got = fscanf(fp, "%d,%d", &rows, &cols);
    if (got == EOF) {
        return 0;
    }
    if (got != 2 || rows <= 0 || cols <= 0) {
        fprintf(stderr, "Invalid matrix header (got %d fields, rows=%d cols=%d)\n",
                got, rows, cols);
        return -1;
    }

    double **m = allocMatrix(rows, cols);
    if (m == NULL) {
        fprintf(stderr, "Could not allocate a %dx%d matrix\n", rows, cols);
        return -1;
    }

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols - 1; c++) {
            if (fscanf(fp, "%lf,", &m[r][c]) != 1) {
                fprintf(stderr, "Matrix %dx%d: bad or missing value at row %d col %d\n",
                        rows, cols, r, c);
                freeMatrix(m, rows);
                return -1;
            }
        }
        if (fscanf(fp, "%lf", &m[r][cols - 1]) != 1) {
            fprintf(stderr, "Matrix %dx%d: bad or missing value at row %d col %d\n",
                    rows, cols, r, cols - 1);
            freeMatrix(m, rows);
            return -1;
        }
    }

    out->rows = rows;
    out->cols = cols;
    out->data = m;
    return 1;
}

// "Label - rows,cols" header followed by comma-separated rows.
void writeMatrix(FILE *out, const char *label, double **m, int rows, int cols) {
    fprintf(out, "%s - %d,%d\n", label, rows, cols);
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            fprintf(out, "%f", m[r][c]);
            if (c < cols - 1) fprintf(out, ",");
        }
        fprintf(out, "\n");
    }
}

// Cap the thread count at the largest dimension involved, then apply it.
int capThreads(int requested, int maxDim) {
    int t = requested;
    if (t > maxDim) t = maxDim;
    if (t < 1) t = 1;
    omp_set_num_threads(t);
    return t;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input_file> <num_threads>\n", argv[0]);
        return 1;
    }

    char *filename = argv[1];
    int requested = atoi(argv[2]);
    if (requested <= 0) {
        fprintf(stderr, "Number of threads must be a positive integer\n");
        return 1;
    }

    FILE *fp = fopen(filename, "r");
    if (fp == NULL) {
        fprintf(stderr, "Could not open file: %s\n", filename);
        return 1;
    }

    FILE *out = fopen("results.txt", "w");
    if (out == NULL) {
        fprintf(stderr, "Could not open results.txt for writing\n");
        fclose(fp);
        return 1;
    }

    int pair = 0;
    while (1) {
        struct Matrix A, B;

        int ra = readMatrix(fp, &A);
        if (ra == 0) break;
        if (ra == -1) {
            fprintf(out, "Stopped: could not read matrix A of a pair.\n");
            break;
        }

        int rb = readMatrix(fp, &B);
        if (rb == 0) {
            fprintf(out, "==================== Pair %d ====================\n", pair + 1);
            fprintf(out, "Only one matrix left (%dx%d); a pair needs two. Skipped.\n",
                    A.rows, A.cols);
            freeMatrix(A.data, A.rows);
            break;
        }
        if (rb == -1) {
            fprintf(out, "Stopped: could not read matrix B of a pair.\n");
            freeMatrix(A.data, A.rows);
            break;
        }

        pair++;
        fprintf(out, "==================== Pair %d ====================\n", pair);
        fprintf(out, "Matrix A: %d,%d   Matrix B: %d,%d\n",
                A.rows, A.cols, B.rows, B.cols);

        int sameShape = (A.rows == B.rows && A.cols == B.cols);

        // Element-wise operations need identical shapes.
        if (sameShape) {
            int t = capThreads(requested, A.rows > A.cols ? A.rows : A.cols);
            fprintf(out, "(element-wise ops using %d threads)\n", t);

            double **add = allocMatrix(A.rows, A.cols);
            double **sub = allocMatrix(A.rows, A.cols);
            double **mul = allocMatrix(A.rows, A.cols);
            double **dvd = allocMatrix(A.rows, A.cols);
            if (!add || !sub || !mul || !dvd) {
                fprintf(stderr, "Allocation failed for element-wise results\n");
                freeMatrix(add, A.rows); freeMatrix(sub, A.rows);
                freeMatrix(mul, A.rows); freeMatrix(dvd, A.rows);
                freeMatrix(A.data, A.rows); freeMatrix(B.data, B.rows);
                fclose(fp); fclose(out);
                return 1;
            }

            double start = omp_get_wtime();
            #pragma omp parallel for collapse(2)
            for (int i = 0; i < A.rows; i++) {
                for (int j = 0; j < A.cols; j++) {
                    add[i][j] = A.data[i][j] + B.data[i][j];
                    sub[i][j] = A.data[i][j] - B.data[i][j];
                    mul[i][j] = A.data[i][j] * B.data[i][j];
                    if (B.data[i][j] == 0.0) {
                        dvd[i][j] = NAN;        // NaN on divide-by-zero
                    } else {
                        dvd[i][j] = A.data[i][j] / B.data[i][j];
                    }
                }
            }
            double elapsed = omp_get_wtime() - start;

            writeMatrix(out, "Addition", add, A.rows, A.cols);
            writeMatrix(out, "Subtraction", sub, A.rows, A.cols);
            writeMatrix(out, "Element-wise multiplication", mul, A.rows, A.cols);
            writeMatrix(out, "Element-wise division", dvd, A.rows, A.cols);
            printf("Pair %d: element-wise ops took %f s\n", pair, elapsed);

            freeMatrix(add, A.rows); freeMatrix(sub, A.rows);
            freeMatrix(mul, A.rows); freeMatrix(dvd, A.rows);
        } else {
            fprintf(out, "Addition cannot be done - A is %d,%d and B is %d,%d\n",
                    A.rows, A.cols, B.rows, B.cols);
            fprintf(out, "Subtraction cannot be done - different size\n");
            fprintf(out, "Element-wise multiplication cannot be done - different size\n");
            fprintf(out, "Element-wise division cannot be done - different size\n");
        }

        // Transposes are always valid.
        {
            int t = capThreads(requested, A.rows > A.cols ? A.rows : A.cols);
            double **ta = allocMatrix(A.cols, A.rows);
            if (ta != NULL) {
                #pragma omp parallel for collapse(2)
                for (int i = 0; i < A.rows; i++) {
                    for (int j = 0; j < A.cols; j++) {
                        ta[j][i] = A.data[i][j];
                    }
                }
                writeMatrix(out, "Transpose of A", ta, A.cols, A.rows);
                freeMatrix(ta, A.cols);
            }
            (void)t;
        }
        {
            int t = capThreads(requested, B.rows > B.cols ? B.rows : B.cols);
            double **tb = allocMatrix(B.cols, B.rows);
            if (tb != NULL) {
                #pragma omp parallel for collapse(2)
                for (int i = 0; i < B.rows; i++) {
                    for (int j = 0; j < B.cols; j++) {
                        tb[j][i] = B.data[i][j];
                    }
                }
                writeMatrix(out, "Transpose of B", tb, B.cols, B.rows);
                freeMatrix(tb, B.cols);
            }
            (void)t;
        }

        // A x B needs A.cols == B.rows.
        if (A.cols == B.rows) {
            int t = capThreads(requested, A.rows);
            fprintf(out, "(matrix multiplication using %d threads)\n", t);

            double **prod = allocMatrix(A.rows, B.cols);
            if (prod != NULL) {
                int i, j, k;
                double tot;
                double start = omp_get_wtime();
                #pragma omp parallel for private(j, k, tot)
                for (i = 0; i < A.rows; i++) {
                    for (j = 0; j < B.cols; j++) {
                        tot = 0.0;
                        for (k = 0; k < A.cols; k++) {
                            tot += A.data[i][k] * B.data[k][j];
                        }
                        prod[i][j] = tot;
                    }
                }
                double elapsed = omp_get_wtime() - start;
                writeMatrix(out, "Matrix multiplication", prod, A.rows, B.cols);
                printf("Pair %d: matrix multiply took %f s\n", pair, elapsed);
                freeMatrix(prod, A.rows);
            }
        } else {
            fprintf(out, "Matrix multiplication cannot be done - "
                         "A columns (%d) do not match B rows (%d)\n",
                    A.cols, B.rows);
        }

        fprintf(out, "\n");
        freeMatrix(A.data, A.rows);
        freeMatrix(B.data, B.rows);
    }

    printf("Processed %d pair(s). Results in results.txt\n", pair);

    fclose(fp);
    fclose(out);
    return 0;
}
