#pragma once
#include "llama.h"
#include "premise-index.hpp"
#include "json.hpp"
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <vector>

struct LeanDeclaration {
    std::string name;
    std::string decl;
};

struct LeanPremiseRecord {
    std::string name;
    std::string decl;
    std::string module;
    std::vector<float> embedding;
};

struct LeanModuleCacheEntry {
    std::string version_token;
    std::vector<std::string> imports;
    std::vector<LeanPremiseRecord> declarations;
};

struct PremiseRetrievalRequest {
    int top_k = 0;
    bool scoped = false;
    std::vector<std::string> imports;
    std::vector<LeanDeclaration> declarations;
};

// Shared state between server.cpp (initialization) and server-context.cpp (use at
// SLOT_STATE_DONE_PROMPT). Initialized once after model load; thereafter only written
// by the server loop thread (no concurrent emb_ctx access needed).

struct PremiseRetrievalState {
    llama_context *               emb_ctx      = nullptr;
    std::unique_ptr<PremiseIndex> premise_index;
    std::unique_ptr<PremiseEmbedCache> embed_cache;
    llama_token                   emb_token_id = -1;
    int                           embedding_dim = 0;
    bool                          joint_generation = false;

    std::mutex emb_mu;
    std::mutex cache_mu;
    std::unordered_map<std::string, LeanModuleCacheEntry> module_cache;

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

nlohmann::ordered_json premise_get_module_version(const nlohmann::ordered_json & data);
nlohmann::ordered_json premise_cache_module(const nlohmann::ordered_json & data);
nlohmann::ordered_json premise_retrieve(const nlohmann::ordered_json & data);
