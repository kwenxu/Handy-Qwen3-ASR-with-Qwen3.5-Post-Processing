#include <stdlib.h>
#include <string.h>

typedef struct {
  char *response;
  int success;
  char *error_message;
} AppleLLMResponse;

int is_apple_intelligence_available(void) { return 0; }

AppleLLMResponse *process_text_with_system_prompt_apple(const char *system_prompt,
                                                        const char *user_content,
                                                        int max_tokens) {
  (void)system_prompt;
  (void)user_content;
  (void)max_tokens;

  AppleLLMResponse *resp = (AppleLLMResponse *)calloc(1, sizeof(AppleLLMResponse));
  if (!resp) {
    return NULL;
  }

  const char *msg =
      "Apple Intelligence is not available in this build (SDK requirement not met).";
  resp->response = NULL;
  resp->success = 0;
  resp->error_message = strdup(msg);
  return resp;
}

void free_apple_llm_response(AppleLLMResponse *response) {
  if (!response) {
    return;
  }
  free(response->response);
  free(response->error_message);
  free(response);
}
