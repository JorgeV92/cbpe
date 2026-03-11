#include "cbpe/tokenizer.h"

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int* ids;
    size_t len;
} cbpe_int_seq_t;

typedef struct {
    int left;
    int right;
} cbpe_pair_t;

static void cbpe_set_error(char** error_out, const char* message) {
    if (error_out == NULL) {
        return;
    }
    size_t len = strlen(message);
    char* copy = (char*)malloc(len+1);
    if (copy == NULL) {
        return;
    }
    memcpy(copy, message, len+1);
    *error_out = copy;
}
