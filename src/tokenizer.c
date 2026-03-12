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
  char* copy = (char*)malloc(len + 1);
  if (copy == NULL) {
    return;
  }
  memcpy(copy, message, len + 1);
  *error_out = copy;
}

void cbpe_free_error(char* error) { free(error); }
void cbpe_free_mem(void* ptr) { free(ptr); }

static int cbpe_pair_compare(const void* a, const void* b) {
  const cbpe_pair_t* lhs = (const cbpe_pair_t*)a;
  const cbpe_pair_t* rhs = (const cbpe_pair_t*)b;
  if (lhs->left != rhs->left) {
    return (lhs->left < rhs->left) ? -1 : 1;
  }
  if (lhs->right != rhs->right) {
    return (lhs->right < rhs->right) ? -1 : 1;
  }
  return 0;
}

static int cbpe_init_base_vocab(cbpe_tokenizer_t* tokenizer) {
  tokenizer->vocab_size = 256;
  tokenizer->merge_count = 0;
  tokenizer->token_bytes =
      (cbpe_bytes_t*)calloc(tokenizer->vocab_size, sizeof(cbpe_bytes_t));
  tokenizer->merges = NULL;
  if (tokenizer->token_bytes == NULL) {
    return 0;
  }
  for (size_t i = 0; i < 256; ++i) {
    tokenizer->token_bytes[i].data = (unsigned char*)malloc(1);
    if (tokenizer->token_bytes[i].data == NULL) {
      return 0;
    }
    tokenizer->token_bytes[i].data[0] = (unsigned char)i;
    tokenizer->token_bytes[i].len = 1;
  }
  return 1;
}