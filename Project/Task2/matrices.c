#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

// Read one line without the newline into a malloc'd string (caller frees), or
// NULL at end of file. A trailing '\r' is stripped so CRLF files also parse.
char *readLine(FILE *fp) {
    size_t cap = 128, len = 0;
    char *buf = malloc(cap);
    if (buf == NULL) return NULL;
    int c;
    while ((c = fgetc(fp)) != EOF && c != '\n') {
        if (len + 1 >= cap) {
            cap *= 2;
            char *tmp = realloc(buf, cap);
            if (tmp == NULL) { free(buf); return NULL; }
            buf = tmp;
        }
        buf[len++] = (char)c;
    }
    if (c == EOF && len == 0) { free(buf); return NULL; }
    if (len > 0 && buf[len - 1] == '\r') len--;
    buf[len] = '\0';
    return buf;
}

int isBlank(const char *s) {
    for (; *s; s++) {
        if (*s != ' ' && *s != '\t' && *s != '\r') return 0;
    }
    return 1;
}

// Returns 1 on success, 0 at clean end of file, -1 on a parse error. Parsing is
// line-based so a wrong row count or non-numeric token is caught, not absorbed
// into the next row.
int readMatrix(FILE *fp, struct Matrix *out) {
    // Skip blank lines, then read the "rows,cols" header.
    char *line;
    do {
        line = readLine(fp);
        if (line == NULL) return 0;
        if (isBlank(line)) { free(line); line = NULL; }
    } while (line == NULL);

    int rows, cols;
    char extra;
    int got = sscanf(line, "%d,%d %c", &rows, &cols, &extra);
    free(line);
    if (got != 2 || rows <= 0 || cols <= 0) {
        fprintf(stderr, "Invalid matrix header (expected rows,cols)\n");
        return -1;
    }

    double **m = allocMatrix(rows, cols);
    if (m == NULL) {
        fprintf(stderr, "Could not allocate a %dx%d matrix\n", rows, cols);
        return -1;
    }

    for (int r = 0; r < rows; r++) {
        line = readLine(fp);
        if (line == NULL || isBlank(line)) {
            fprintf(stderr, "Matrix %dx%d: declares %d rows but row %d is missing\n",
                    rows, cols, rows, r);
            free(line);
            freeMatrix(m, rows);
            return -1;
        }

        int c = 0;
        char *tok = strtok(line, ",");
        while (tok != NULL) {
            if (c >= cols) {
                fprintf(stderr, "Matrix %dx%d: row %d has more than %d values\n",
                        rows, cols, r, cols);
                free(line);
                freeMatrix(m, rows);
                return -1;
            }
            char *end;
            double v = strtod(tok, &end);
            while (*end == ' ' || *end == '\t' || *end == '\r') end++;
            if (end == tok || *end != '\0') {
                fprintf(stderr, "Matrix %dx%d: non-numeric value '%s' at row %d col %d\n",
                        rows, cols, tok, r, c);
                free(line);
                freeMatrix(m, rows);
                return -1;
            }
            m[r][c++] = v;
            tok = strtok(NULL, ",");
        }
        free(line);

        if (c != cols) {
            fprintf(stderr, "Matrix %dx%d: row %d has %d values, expected %d\n",
                    rows, cols, r, c, cols);
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
