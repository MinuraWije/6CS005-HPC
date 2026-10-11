# 6CS005 HPC – Submission

Arranged to match the required layout: a separate **Project** folder and a
separate **Evidence** folder.

```
submission/
├── Project/
│   ├── Task1/   wordcount.c, WordOccurrenceDataset.txt, README.md
│   ├── Task2/   matrices.c, MatData.txt, README.md
│   ├── Task3/   crack.cu, passgen.c, passwords.txt, passwords_plain.txt, README.md
│   └── Task4/   sobel.cu, lodepng.h, lodepng.cpp, images/, README.md
└── Evidence/
    ├── Notebooks/  Task1.ipynb … Task4.ipynb  (the Colab notebooks)
    ├── Videos/     5 recordings (4 task runs + 1 submission), < 20 MB each
    └── Report/     6CS005_Report.docx  (fill in URLs, screenshots, write-ups)
```

The task folders under `Project/` hold only the source and resource files (matching
the whiteboard: `C/cu/cpp/h` + `txt, png`). The `.ipynb` notebooks live under
`Evidence/Notebooks/`.

## What is done
- All four programs are written and verified (see each task's README).
- Each task has a Colab notebook (`TaskN.ipynb`) that **downloads its data with
  `wget`**, writes the source with `%%writefile`, compiles and runs it — the
  class workflow (HPC-8/9/11). Tasks 1–2 use gcc; Tasks 3–4 use nvcc (GPU).
- `Evidence/Report/6CS005_Report.docx` is a per-task template with the
  marking-criteria tables and placeholders for Colab links, screenshots and
  write-ups.
- `Evidence/Videos/README.txt` lists exactly what each video must show.

## IMPORTANT — make the notebooks runnable by anyone
Each notebook has a `RAW_BASE` variable at the top:
```python
RAW_BASE = "https://raw.githubusercontent.com/USERNAME/REPO/BRANCH/Project"
```
1. Create a **public GitHub repo** and push the `Project/` folder into it.
2. Replace `USERNAME/REPO/BRANCH` with your repo (it must point at the folder
   that contains `Task1 .. Task4`).
3. The `wget` cells then pull the data/images, so **anyone with the Colab link
   can run the notebook end to end** without uploading anything.

## What you still need to do (only you can)
1. **Push `Project/` to a public GitHub repo** and set `RAW_BASE` in each notebook.
2. **Run each notebook in Colab** (Tasks 3–4: Runtime → Change runtime type → GPU).
3. **Share each notebook**: Share → Anyone with the link → **Viewer** → Copy link,
   and paste the link into the report.
4. **Record the 5 videos** per `Evidence/Videos/README.txt` (voice, show your ID,
   show your Colab profile, live run→result; plus the submission upload showing
   the system tray + date/time). Keep each under 20 MB.
5. **Fill in** your name, ID, date and the "Results / discussion" lines in the report.
6. **Verify**: open the shared Colab link in a private window (or ask a friend) to
   confirm it is viewable and runs. Then **zip each task** and submit on Canvas —
   early, not at the last minute.

The red italic text in the report marks every spot that needs your input.
