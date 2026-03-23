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

static int cbpe_append_merge(cbpe_tokenizer_t* tokenizer,
                             int left,
                             int right,
                             int new_id,
                             int rank) {
  size_t new_count = tokenizer->merge_count + 1;
  cbpe_merge_t* grown =
      (cbpe_merge_t*)realloc(tokenizer->merges, new_count * sizeof(cbpe_merge_t));
  if (grown == NULL) {
    return 0;
  }
  tokenizer->merges = grown;
  tokenizer->merges[tokenizer->merge_count].left = left;
  tokenizer->merges[tokenizer->merge_count].right = right;
  tokenizer->merges[tokenizer->merge_count].new_id = new_id;
  tokenizer->merges[tokenizer->merge_count].rank = rank;
  tokenizer->merge_count = new_count;
  return 1;
}

static int cbpe_replace_pair(cbpe_int_seq_t* seq, int left, int right, int new_id) {
  if (seq->len < 2) {
    return 0;
  }
  int changed = 0;
  int* out = (int*)malloc(seq->len * sizeof(int));
  if (out == NULL) {
    return -1;
  }
  size_t out_len = 0;
  size_t i = 0;
  while (i < seq->len) {
    if (i + 1 < seq->len && seq->ids[i] == left && seq->ids[i + 1] == right) {
      out[out_len++] = new_id;
      i += 2;
      changed = 1;
    } else {
      out[out_len++] = seq->ids[i++];
    }
  }
  free(seq->ids);
  seq->ids = out;
  seq->len = out_len;
  return changed;
}

static void cbpe_free_sequences(cbpe_int_seq_t* seqs, size_t n) {
  if (seqs == NULL) {
    return;
  }
  for (size_t i = 0; i < n; ++i) {
    free(seqs[i].ids);
  }
  free(seqs);
}

static cbpe_int_seq_t* cbpe_make_training_sequences(const unsigned char** texts,
                                                    const size_t* lengths,
                                                    size_t n_texts,
                                                    char** error_out) {
  cbpe_int_seq_t* seqs = (cbpe_int_seq_t*)calloc(n_texts, sizeof(cbpe_int_seq_t));
  if (seqs == NULL) {
    cbpe_set_error(error_out, "failed to allocate training sequences");
    return NULL;
  }
  for (size_t i = 0; i < n_texts; ++i) {
    seqs[i].len = lengths[i];
    if (lengths[i] == 0) {
      seqs[i].ids = NULL;
      continue;
    }
    seqs[i].ids = (int*)malloc(lengths[i] * sizeof(int));
    if (seqs[i].ids == NULL) {
      cbpe_free_sequences(seqs, n_texts);
      cbpe_set_error(error_out, "failed to allocate training token buffer");
      return NULL;
    }
    for (size_t j = 0; j < lengths[i]; ++j) {
      seqs[i].ids[j] = texts[i][j];
    }
  }
  return seqs;
}

int cbpe_tokenizer_train(cbpe_tokenizer_t* tokenizer,
                         const unsigned char** texts,
                         const size_t* lengths,
                         size_t n_texts,
                         size_t target_vocab_size,
                         char** error_out) {
  if (tokenizer == NULL) {
    cbpe_set_error(error_out, "tokenizer must not be NULL");
    return 0;
  }
  if (target_vocab_size < 256) {
    cbpe_set_error(error_out, "target_vocab_size must be at least 256");
    return 0;
  }

  cbpe_reset_to_base_vocab(tokenizer);
  cbpe_int_seq_t* seqs = cbpe_make_training_sequences(texts, lengths, n_texts, error_out);
  if (seqs == NULL && n_texts != 0) {
    return 0;
  }

  while (tokenizer->vocab_size < target_vocab_size) {
    size_t total_pairs = 0;
    for (size_t i = 0; i < n_texts; ++i) {
      if (seqs[i].len >= 2) {
        total_pairs += seqs[i].len - 1;
      }
    }
    if (total_pairs == 0) {
      break;
    }

    cbpe_pair_t* pairs = (cbpe_pair_t*)malloc(total_pairs * sizeof(cbpe_pair_t));
    if (pairs == NULL) {
      cbpe_free_sequences(seqs, n_texts);
      cbpe_set_error(error_out, "failed to allocate pair buffer during training");
      return 0;
    }

    size_t k = 0;
    for (size_t i = 0; i < n_texts; ++i) {
      for (size_t j = 0; j + 1 < seqs[i].len; ++j) {
        pairs[k].left = seqs[i].ids[j];
        pairs[k].right = seqs[i].ids[j + 1];
        ++k;
      }
    }

    qsort(pairs, total_pairs, sizeof(cbpe_pair_t), cbpe_pair_compare);

    int best_left = -1;
    int best_right = -1;
    size_t best_count = 0;
    size_t run_count = 1;

    for (size_t i = 1; i <= total_pairs; ++i) {
      if (i < total_pairs && pairs[i].left == pairs[i - 1].left &&
          pairs[i].right == pairs[i - 1].right) {
        ++run_count;
      } else {
        if (run_count > best_count) {
          best_count = run_count;
          best_left = pairs[i - 1].left;
          best_right = pairs[i - 1].right;
        }
        run_count = 1;
      }
    }
    free(pairs);

    if (best_count < 2 || best_left < 0 || best_right < 0) {
      break;
    }

    int new_id = (int)tokenizer->vocab_size;
    int rank = (int)tokenizer->merge_count;
    if (!cbpe_append_token(tokenizer, best_left, best_right) ||
        !cbpe_append_merge(tokenizer, best_left, best_right, new_id, rank)) {
      cbpe_free_sequences(seqs, n_texts);
      cbpe_set_error(error_out, "failed to grow tokenizer vocabulary");
      return 0;
    }

    int any_changed = 0;
    for (size_t i = 0; i < n_texts; ++i) {
      int changed = cbpe_replace_pair(&seqs[i], best_left, best_right, new_id);
      if (changed < 0) {
        cbpe_free_sequences(seqs, n_texts);
        cbpe_set_error(error_out, "failed to apply merge during training");
        return 0;
      }
      any_changed |= changed;
    }
    if (!any_changed) {
      break;
    }
  }

  cbpe_free_sequences(seqs, n_texts);
  return 1;
}

static const cbpe_merge_t* cbpe_find_merge(const cbpe_tokenizer_t* tokenizer,
                                           int left,
                                           int right) {
  for (size_t i = 0; i < tokenizer->merge_count; ++i) {
    if (tokenizer->merges[i].left == left && tokenizer->merges[i].right == right) {
      return &tokenizer->merges[i];
    }
  }
  return NULL;
}