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

cbpe_tokenizer_t* cbpe_tokenizer_new(void) {
  cbpe_tokenizer_t* tokenizer =
      (cbpe_tokenizer_t*)calloc(1, sizeof(cbpe_tokenizer_t));
  if (tokenizer == NULL) {
    return NULL;
  }
  if (!cbpe_init_base_vocab(tokenizer)) {
    cbpe_tokenizer_free(tokenizer);
    return NULL;
  }
  return tokenizer;
}

void cbpe_tokenizer_free(cbpe_tokenizer_t* tokenizer) {
  if (tokenizer == NULL) {
    return;
  }
  if (tokenizer->token_bytes != NULL) {
    for (size_t i = 0; i < tokenizer->vocab_size; ++i) {
      free(tokenizer->token_bytes[i].data);
    }
  }
  free(tokenizer->token_bytes);
  free(tokenizer->merges);
  free(tokenizer);
}

static void cbpe_reset_to_base_vocab(cbpe_tokenizer_t* tokenizer) {
  if (tokenizer == NULL) {
    return;
  }
  if (tokenizer->token_bytes != NULL) {
    for (size_t i = 0; i < tokenizer->vocab_size; ++i) {
      free(tokenizer->token_bytes[i].data);
    }
    free(tokenizer->token_bytes);
  }
  free(tokenizer->merges);
  tokenizer->token_bytes = NULL;
  tokenizer->merges = NULL;
  tokenizer->vocab_size = 0;
  tokenizer->merge_count = 0;
  cbpe_init_base_vocab(tokenizer);
}

static int cbpe_append_token(cbpe_tokenizer_t* tokenizer, int left, int right) {
  size_t new_vocab_size = tokenizer->vocab_size + 1;
  cbpe_bytes_t* grown_vocab = (cbpe_bytes_t*)realloc(
      tokenizer->token_bytes, new_vocab_size * sizeof(cbpe_bytes_t));
  if (grown_vocab == NULL) {
    return 0;
  }
  tokenizer->token_bytes = grown_vocab;

  cbpe_bytes_t* left_bytes = &tokenizer->token_bytes[left];
  cbpe_bytes_t* right_bytes = &tokenizer->token_bytes[right];
  size_t new_len = left_bytes->len + right_bytes->len;
  unsigned char* merged = (unsigned char*)malloc(new_len);
  if (merged == NULL) {
    return 0;
  }
  memcpy(merged, left_bytes->data, left_bytes->len);
  memcpy(merged + left_bytes->len, right_bytes->data, right_bytes->len);

  tokenizer->token_bytes[tokenizer->vocab_size].data = merged;
  tokenizer->token_bytes[tokenizer->vocab_size].len = new_len;
  tokenizer->vocab_size = new_vocab_size;
  return 1;
}