# Task 4 – Sobel Edge Detection across many PNGs (CUDA)

Applies the Sobel edge-detection filter to one or more PNG images on the GPU.
For each input `name.png` it writes `name_sobel.png`.

## Files
- `sobel.cu` – the CUDA program
- `lodepng.h`, `lodepng.cpp` – PNG decode/encode (lvandeve/lodepng)
- `images/` – test images (4×4, 6×4, a solid colour, a split edge, and one
  wider than 1024 px)

## Build & run — Google Colab (GPU runtime)
This task needs `nvcc`, which is not installed locally. Use Colab as in
HPC-11 week 2:

Cell 1:
```
!nvcc --version
```
Cell 2 (if you did not upload them, fetch lodepng as in class):
```
!wget -q https://raw.githubusercontent.com/lvandeve/lodepng/master/lodepng.h
!wget -q https://raw.githubusercontent.com/lvandeve/lodepng/master/lodepng.cpp
```
Cell 3 (upload `sobel.cu` and the images):
```
!nvcc -o sobel sobel.cu lodepng.cpp
!./sobel images/4x4.png images/6x4.png images/half.png images/wide.png
```

## Usage
```
./sobel <image1.png> [image2.png ...]
```
Every argument is an input image; each is processed independently and a bad
image is reported and skipped without stopping the batch.

## How it works
This is the HPC-11 week 2 program, extended to:
- **loop over every image** given on the command line;
- use **zero padding** at the edges — a neighbour that falls outside the image
  counts as 0 — instead of blacking the border out, as the brief requires;
- build the output name with `sprintf` (`photo.png` → `photo_sobel.png`);
- **check every CUDA call** and free GPU and CPU memory per image, so memory
  use stays flat however many images are processed;
- **skip a bad image** and carry on.

Per image: decode with `lodepng_decode32_file`, `cudaMalloc` + `cudaMemcpy`
sized to the image, launch `sobelEdge` with 16×16 blocks over a grid of
`(w+15)/16 × (h+15)/16`, convert each neighbour to grey
(`0.299R + 0.587G + 0.114B`), accumulate `Gx`/`Gy`, write
`G = sqrt(Gx²+Gy²)` clamped to 255, copy back, and encode with
`lodepng_encode32_file`.

## Checks performed
The Sobel maths with zero padding was reproduced with a NumPy reference on the
4×4 and 6×4 images and matched a hand calculation (e.g. the top-left pixel of
4×4 gives Gx = Gy ≈ 162, magnitude ≈ 229). The 16×16 grid handles any size,
including the >1024 px test image, where the Week 9 `<<<height,width>>>`
launch would fail.
