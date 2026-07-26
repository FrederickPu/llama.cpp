#pragma once
#include "llama.h"
#include "premise-index.hpp"
#include "json.hpp"
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

struct LeanDeclaration {
    std::string name;
    std::string decl;
};

struct PremiseRetrievalRequest {
    int top_k = 0;
    bool scoped = false;
    std::vector<std::string> imports;
    std::vector<LeanDeclaration> declarations;
};

// Shared state between server.cpp (initialization) and server-context.cpp (use at
// SLOT_STATE_DONE_PROMPT). Initialized once after model load; embedding contexts
// are leased so concurrent HTTP requests do not share a llama_context.

struct PremiseRetrievalState {
    std::vector<llama_context *>  emb_ctxs;
    std::vector<llama_context *>  idle_emb_ctxs;
    std::unique_ptr<PremiseIndex> premise_index;
    llama_token                   emb_token_id = -1;
    int                           embedding_dim = 0;
    bool                          joint_generation = false;

    std::mutex emb_mu;
    std::condition_variable emb_cv;

    // pending_premises and task_topk are written by the server loop thread and read
    // by the HTTP handler thread, so they share a mutex.
    std::mutex                           pending_mu;
    std::condition_variable              pending_cv;
    std::unordered_map<int, std::string> pending_premises; // task_id -> SSE chunk
    std::unordered_map<int, PremiseRetrievalRequest> task_requests;
};

extern PremiseRetrievalState * g_premise_state;

std::string premise_task_created(int task_id, const nlohmann::ordered_json & data, bool stream, int n_cmpl);
void premise_prefill_complete(int task_id, const std::vector<llama_token> & prompt_tokens);
std::string premise_take_initial_stream_prefix(int task_id);

nlohmann::ordered_json premise_api_version(const nlohmann::ordered_json & data);
nlohmann::ordered_json premise_api_cache(const nlohmann::ordered_json & data);
nlohmann::ordered_json premise_api_select(const nlohmann::ordered_json & data);

nlohmann::ordered_json premise_api_version_batch(const nlohmann::ordered_json & data);
nlohmann::ordered_json premise_api_cache_batch(const nlohmann::ordered_json & data);
