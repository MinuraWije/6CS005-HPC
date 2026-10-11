#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

// Task 1 - word occurrence counting with Pthreads.
// Usage: ./wordcount <input_file> <num_threads>   Output: result.txt

#define MAX_WORD_LEN 50
#define MAX_UNIQUE  1000

struct wordCount {
    char word[MAX_WORD_LEN];
    int  count;
};

struct threadArgs {
    int start;
    int finish;
    char (*words)[MAX_WORD_LEN];
};

struct wordCount table[MAX_UNIQUE];
int uniqueCount = 0;
pthread_mutex_t mutex;

int findWord(struct wordCount *tbl, int size, const char *word) {
    for (int i = 0; i < size; i++) {
        if (strcmp(tbl[i].word, word) == 0) {
            return i;
        }
    }
    return -1;
}

void *countMethod(void *args) {
    struct threadArgs *ta = (struct threadArgs *)args;
    int start = ta->start;
    int finish = ta->finish;
    char (*words)[MAX_WORD_LEN] = ta->words;

    // Count into a local table first, then merge once under the lock.
    struct wordCount *local = malloc(MAX_UNIQUE * sizeof(struct wordCount));
    if (local == NULL) {
        fprintf(stderr, "Thread could not allocate local table\n");
        pthread_exit(NULL);
    }
    int localCount = 0;

    long processed = 0;
    for (int i = start; i <= finish; i++) {
        int pos = findWord(local, localCount, words[i]);
        if (pos == -1) {
            strcpy(local[localCount].word, words[i]);
            local[localCount].count = 1;
            localCount++;
        } else {
            local[pos].count++;
        }
        processed++;
    }

    pthread_mutex_lock(&mutex);
    for (int i = 0; i < localCount; i++) {
        int pos = findWord(table, uniqueCount, local[i].word);
        if (pos == -1) {
            strcpy(table[uniqueCount].word, local[i].word);
            table[uniqueCount].count = local[i].count;
            uniqueCount++;
        } else {
            table[pos].count += local[i].count;
        }
    }
    pthread_mutex_unlock(&mutex);

    free(local);

    long *answer = malloc(sizeof(long));
    *answer = processed;
    pthread_exit(answer);
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input_file> <num_threads>\n", argv[0]);
        return 1;
    }

    char *filename = argv[1];
    int threadCount = atoi(argv[2]);
    if (threadCount <= 0) {
        fprintf(stderr, "Number of threads must be a positive integer\n");
        return 1;
    }

    // Pass 1: count the words.
    FILE *fp = fopen(filename, "r");
    if (fp == NULL) {
        fprintf(stderr, "Could not open file: %s\n", filename);
        return 1;
    }

    int wordTotal = 0;
    int inWord = 0;
    int c;
    for (c = getc(fp); c != EOF; c = getc(fp)) {
        if (c == ' ' || c == '\n' || c == '\t' || c == '\r') {
            inWord = 0;
        } else if (inWord == 0) {
            inWord = 1;
            wordTotal++;
        }
    }
    fclose(fp);

    if (wordTotal == 0) {
        fprintf(stderr, "File contains no words\n");
        return 1;
    }

    // Pass 2: read the words into memory, folded to lower case.
    char (*words)[MAX_WORD_LEN] = malloc(wordTotal * sizeof(*words));
    if (words == NULL) {
        fprintf(stderr, "Could not allocate memory for %d words\n", wordTotal);
        return 1;
    }

    fp = fopen(filename, "r");
    if (fp == NULL) {
        fprintf(stderr, "Could not re-open file: %s\n", filename);
        free(words);
        return 1;
    }
    for (int i = 0; i < wordTotal; i++) {
        if (fscanf(fp, "%49s", words[i]) != 1) {
            fprintf(stderr, "Unexpected end of file at word %d\n", i);
            fclose(fp);
            free(words);
            return 1;
        }
        for (int j = 0; words[i][j] != '\0'; j++) {
            if (words[i][j] >= 'A' && words[i][j] <= 'Z') {
                words[i][j] = words[i][j] + 32;
            }
        }
    }
    fclose(fp);

    if (threadCount > wordTotal) {
        threadCount = wordTotal;
        printf("More threads than words; using %d threads.\n", threadCount);
    }

    // Dynamic slicing by word index; the remainder is spread over the first threads.
    int *sliceList = malloc(threadCount * sizeof(int));
    int *startList = malloc(threadCount * sizeof(int));
    int *finishList = malloc(threadCount * sizeof(int));
    struct threadArgs *args = malloc(threadCount * sizeof(struct threadArgs));
    pthread_t *tids = malloc(threadCount * sizeof(pthread_t));
    if (!sliceList || !startList || !finishList || !args || !tids) {
        fprintf(stderr, "Could not allocate scheduling arrays\n");
        free(words); free(sliceList); free(startList);
        free(finishList); free(args); free(tids);
        return 1;
    }

    int remainder = wordTotal % threadCount;
    for (int i = 0; i < threadCount; i++) {
        sliceList[i] = wordTotal / threadCount;
    }
    for (int j = 0; j < remainder; j++) {
        sliceList[j] = sliceList[j] + 1;
    }
    for (int k = 0; k < threadCount; k++) {
        if (k == 0) {
            startList[k] = 0;
            finishList[k] = startList[k] + sliceList[k] - 1;
        } else {
            startList[k] = finishList[k - 1] + 1;
            finishList[k] = startList[k] + sliceList[k] - 1;
        }
    }
    for (int i = 0; i < threadCount; i++) {
        printf("Thread %d: start = %d  finish = %d  (%d words)\n",
               i, startList[i], finishList[i], sliceList[i]);
    }

    pthread_mutex_init(&mutex, NULL);

    for (int i = 0; i < threadCount; i++) {
        args[i].start = startList[i];
        args[i].finish = finishList[i];
        args[i].words = words;
        pthread_create(&tids[i], NULL, countMethod, &args[i]);
    }

    long grandTotal = 0;
    for (int j = 0; j < threadCount; j++) {
        long *result;
        pthread_join(tids[j], (void **)&result);
        if (result != NULL) {
            grandTotal += *result;
            free(result);
        }
    }

    pthread_mutex_destroy(&mutex);

    printf("Threads processed %ld words in total (file has %d).\n",
           grandTotal, wordTotal);

    // Sort: count descending, ties alphabetical.
    for (int i = 0; i < uniqueCount - 1; i++) {
        int best = i;
        for (int j = i + 1; j < uniqueCount; j++) {
            if (table[j].count > table[best].count ||
                (table[j].count == table[best].count &&
                 strcmp(table[j].word, table[best].word) < 0)) {
                best = j;
            }
        }
        if (best != i) {
            struct wordCount tmp = table[i];
            table[i] = table[best];
            table[best] = tmp;
        }
    }

    FILE *out = fopen("result.txt", "w");
    if (out == NULL) {
        fprintf(stderr, "Could not open result.txt for writing\n");
        free(words); free(sliceList); free(startList);
        free(finishList); free(args); free(tids);
        return 1;
    }
    fprintf(out, "Word Occurrence Results\n");
    fprintf(out, "File: %s   Threads: %d   Total words: %d   Unique words: %d\n",
            filename, threadCount, wordTotal, uniqueCount);
    fprintf(out, "--------------------------------\n");
    for (int i = 0; i < uniqueCount; i++) {
        fprintf(out, "%-30s %d\n", table[i].word, table[i].count);
    }
    fclose(out);

    printf("Wrote %d unique words to result.txt\n", uniqueCount);

    free(words);
    free(sliceList);
    free(startList);
    free(finishList);
    free(args);
    free(tids);
    return 0;
}
