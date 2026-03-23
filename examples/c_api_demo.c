#include <stdio.h>
#include <stdlib.h>

#include "cbpe/tokenizer.h"

int main(void) {
  const unsigned char* texts[] = {
      (const unsigned char*)"hello world",
      (const unsigned char*)"hello there",
      (const unsigned char*)"hello hello world",
  };
  size_t lengths[] = {11, 11, 17};

  cbpe_tokenizer_t* tokenizer = cbpe_tokenizer_new();
  if (tokenizer == NULL) {
    fprintf(stderr, "failed to create tokenizer\n");
    return 1;
  }

  char* error = NULL;
  if (!cbpe_tokenizer_train(tokenizer, texts, lengths, 3, 300, &error)) {
    fprintf(stderr, "train error: %s\n", error ? error : "unknown");
    cbpe_free_error(error);
    cbpe_tokenizer_free(tokenizer);
    return 1;
  }

  int* ids = NULL;
  size_t ids_len = 0;
  if (!cbpe_tokenizer_encode(tokenizer,
                             (const unsigned char*)"hello world",
                             11,
                             &ids,
                             &ids_len,
                             &error)) {
    fprintf(stderr, "encode error: %s\n", error ? error : "unknown");
    cbpe_free_error(error);
    cbpe_tokenizer_free(tokenizer);
    return 1;
  }

  printf("encoded: ");
  for (size_t i = 0; i < ids_len; ++i) {
    printf("%d ", ids[i]);
  }
  printf("\n");

  unsigned char* decoded = NULL;
  size_t decoded_len = 0;
  if (!cbpe_tokenizer_decode(tokenizer, ids, ids_len, &decoded, &decoded_len, &error)) {
    fprintf(stderr, "decode error: %s\n", error ? error : "unknown");
    cbpe_free_error(error);
    cbpe_free_mem(ids);
    cbpe_tokenizer_free(tokenizer);
    return 1;
  }

  printf("decoded: %.*s\n", (int)decoded_len, decoded);
  printf("vocab_size: %zu\n", cbpe_tokenizer_vocab_size(tokenizer));

  cbpe_free_mem(decoded);
  cbpe_free_mem(ids);
  cbpe_tokenizer_free(tokenizer);
  return 0;
}
