#pragma once

#include "llama.h"
#include "json.hpp"

#include <string>
#include <vector>

struct server_context;
struct server_http_context;

enum class PremiseMode {
    Auto,
    Joint,
    Embedding,
};

// Setup / teardown (after model load).
void premise_setup(server_context & ctx_server,
                   const std::string & vecs_path,
                   const std::string & names_path,
                   PremiseMode mode);
void premise_cleanup();

// HTTP: POST /version, /cache, /select
void premise_register_http_routes(const server_http_context & ctx_http);

// Joint-generation hooks (used by joint-server via server-context).
std::string premise_task_created(int task_id, const nlohmann::ordered_json & data, bool stream, int n_cmpl);
void premise_prefill_complete(int task_id, const std::vector<llama_token> & prompt_tokens);
std::string premise_take_initial_stream_prefix(int task_id);
