#pragma once

#include "premise-retrieval.hpp"

static std::string server_context_hook_task_created(int task_id, const json & data, bool stream, int n_cmpl) {
    return premise_task_created(task_id, data, stream, n_cmpl);
}

static void server_context_hook_prefill_complete(int task_id, const llama_tokens & prompt_tokens) {
    premise_prefill_complete(task_id, prompt_tokens);
}

static std::string server_context_hook_take_initial_stream_prefix(int task_id) {
    return premise_take_initial_stream_prefix(task_id);
}
