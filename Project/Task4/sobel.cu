#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lodepng.h"

/*
 * Task 4 - Sobel edge detection across many PNGs with CUDA (6CS005).
 *
 * Usage: ./sobel <image1.png> [image2.png ...]
 * For each input "name.png" it writes "name_sobel.png".
 *
 * This is the HPC-11 week 2 program, extended to:
 *   - loop over every image given on the command line,
 *   - use zero padding at the edges (out-of-image neighbours count as 0)
 *     instead of blacking the border out, as the brief requires,
 *   - build the output file name with sprintf,
 *   - check the CUDA calls for errors and free memory per image,
 *   - skip a bad image and carry on with the rest.
 */

/*
 * CUDA error-checking helper (required by the brief).  Returns 0 on success,
 * 1 on failure, so the caller can clean up and skip this image rather than
 * killing the whole batch.
 */
#define CUDA_TRY(call)                                                        \
    do {                                                                     \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "  CUDA error at %s:%d - %s\n",                 \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            goto cleanup;                                                   \
        }                                                                  \
    } while (0)

/*
 * Sobel kernel with zero padding.
 * RGBA layout: R=pixel[0] G=pixel[1] B=pixel[2] A=pixel[3].
 */
__global__ void sobelEdge(unsigned char *inImg, unsigned char *outImg,
                          unsigned int width, unsigned int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;                         /* spare threads outside the image */
    }

    int index = (y * width + x) * 4;

    int gx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
    int gy[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };

    float Gx = 0.0f;
    float Gy = 0.0f;

    for (int ky = -1; ky <= 1; ky++) {
        for (int kx = -1; kx <= 1; kx++) {
            int pixelX = x + kx;
            int pixelY = y + ky;

            float gray;
            if (pixelX < 0 || pixelX >= width ||
                pixelY < 0 || pixelY >= height) {
                /* Zero padding: a neighbour off the edge counts as 0. */
                gray = 0.0f;
            } else {
                int pixelIndex = (pixelY * width + pixelX) * 4;
                unsigned char r = inImg[pixelIndex];
                unsigned char g = inImg[pixelIndex + 1];
                unsigned char b = inImg[pixelIndex + 2];
                gray = 0.299f * r + 0.587f * g + 0.114f * b;
            }

            Gx += gray * gx[ky + 1][kx + 1];
            Gy += gray * gy[ky + 1][kx + 1];
        }
    }

    float magnitude = sqrtf(Gx * Gx + Gy * Gy);
    if (magnitude > 255.0f) {
        magnitude = 255.0f;
    }
    unsigned char edge = (unsigned char)magnitude;

    outImg[index]     = edge;
    outImg[index + 1] = edge;
    outImg[index + 2] = edge;
    outImg[index + 3] = 255;
}

/* Process one image.  Returns 0 on success, non-zero on failure. */
int processImage(const char *inFileName) {
    unsigned char *c_inImg = NULL;
    unsigned char *c_outImg = NULL;
    unsigned char *d_inImg = NULL;
    unsigned char *d_outImg = NULL;
    unsigned int width = 0, height = 0;
    int rc = 1;

    printf("Processing %s\n", inFileName);

    /* Decode the PNG. */
    unsigned int lpError =
        lodepng_decode32_file(&c_inImg, &width, &height, inFileName);
    if (lpError) {
        fprintf(stderr, "  Error decoding '%s': %s\n",
                inFileName, lodepng_error_text(lpError));
        return 1;                        /* skip this image */
    }
    if (width == 0 || height == 0) {
        fprintf(stderr, "  '%s' has zero size, skipping\n", inFileName);
        free(c_inImg);
        return 1;
    }
    printf("  width %u, height %u\n", width, height);

    unsigned int imgSize = 4 * width * height;

    /* Output buffer on the heap (not the stack, so big images are fine). */
    c_outImg = (unsigned char *)malloc(imgSize);
    if (c_outImg == NULL) {
        fprintf(stderr, "  Could not allocate output buffer\n");
        free(c_inImg);
        return 1;
    }

    /* GPU memory, checked for errors. */
    CUDA_TRY(cudaMalloc((void **)&d_inImg, imgSize));
    CUDA_TRY(cudaMalloc((void **)&d_outImg, imgSize));
    CUDA_TRY(cudaMemcpy(d_inImg, c_inImg, imgSize, cudaMemcpyHostToDevice));

    /* 16x16 blocks, grid big enough to cover the image (HPC-11 wk2). */
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    sobelEdge<<<gridSize, blockSize>>>(d_inImg, d_outImg, width, height);
    CUDA_TRY(cudaGetLastError());
    CUDA_TRY(cudaDeviceSynchronize());

    CUDA_TRY(cudaMemcpy(c_outImg, d_outImg, imgSize, cudaMemcpyDeviceToHost));

    /* Build the output name: "photo.png" -> "photo_sobel.png". */
    char outFileName[512];
    const char *dot = strrchr(inFileName, '.');
    if (dot != NULL) {
        int base = (int)(dot - inFileName);
        sprintf(outFileName, "%.*s_sobel%s", base, inFileName, dot);
    } else {
        sprintf(outFileName, "%s_sobel.png", inFileName);
    }

    lpError = lodepng_encode32_file(outFileName, c_outImg, width, height);
    if (lpError) {
        fprintf(stderr, "  Error encoding '%s': %s\n",
                outFileName, lodepng_error_text(lpError));
    } else {
        printf("  Saved %s\n", outFileName);
        rc = 0;
    }

cleanup:
    /* Free GPU and CPU memory for this image before the next one. */
    if (d_inImg)  cudaFree(d_inImg);
    if (d_outImg) cudaFree(d_outImg);
    free(c_inImg);
    free(c_outImg);
    return rc;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <image1.png> [image2.png ...]\n", argv[0]);
        return 1;
    }

    int ok = 0, failed = 0;
    for (int i = 1; i < argc; i++) {
        if (processImage(argv[i]) == 0) {
            ok++;
        } else {
            failed++;
        }
    }

    printf("Done: %d image(s) processed, %d skipped.\n", ok, failed);
    return 0;
}
