#ifndef CBPE_TOKENIZER_H_
#define CBPE_TOKENIZER_H_

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  unsigned char* data;
  size_t len;
} cbpe_bytes_t;

typedef struct {
  int left;
  int right;
  int new_id;
  int rank;
} cbpe_merge_t;

typedef struct {
  size_t vocab_size;
  size_t merge_count;
  cbpe_bytes_t* token_bytes;
  cbpe_merge_t* merges;
} cbpe_tokenizer_t;

cbpe_tokenizer_t* cbpe_tokenizer_new(void);
void cbpe_tokenizer_free(cbpe_tokenizer_t* tokenizer);

int cbpe_tokenizer_train(cbpe_tokenizer_t* tokenizer,
                         const unsigned char** texts,
                         const size_t* lengths,
                         size_t n_texts,
                         size_t target_vocab_size,
                         char** error_out);

int cbpe_tokenizer_encode(const cbpe_tokenizer_t* tokenizer,
                          const unsigned char* text,
                          size_t len,
                          int** out_ids,
                          size_t* out_len,
                          char** error_out);

int cbpe_tokenizer_decode(const cbpe_tokenizer_t* tokenizer,
                          const int* ids,
                          size_t ids_len,
                          unsigned char** out_text,
                          size_t* out_len,
                          char** error_out);

size_t cbpe_tokenizer_vocab_size(const cbpe_tokenizer_t* tokenizer);
const cbpe_bytes_t* cbpe_tokenizer_vocab_item(const cbpe_tokenizer_t* tokenizer,
                                              size_t token_id);
const cbpe_merge_t* cbpe_tokenizer_merges(const cbpe_tokenizer_t* tokenizer,
                                          size_t* out_count);

void cbpe_free_mem(void* ptr);
void cbpe_free_error(char* error);

#ifdef __cplusplus
}
#endif

#endif  // CBPE_TOKENIZER_H_