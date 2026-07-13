#include "premise-retrieval.hpp"

#include "common.h"
#include "server-common.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <exception>
#include <stdexcept>
#include <thread>
#include <unordered_set>

using json = nlohmann::ordered_json;

PremiseRetrievalState * g_premise_state = nullptr;

static llama_context * premise_first_embedding_context() {
    if (!g_premise_state || g_premise_state->emb_ctxs.empty()) {
        throw std::runtime_error("joint retrieval is not initialized");
    }
    return g_premise_state->emb_ctxs.front();
}

struct PremiseEmbeddingContextLease {
    llama_context * ctx = nullptr;

    PremiseEmbeddingContextLease() = default;
    explicit PremiseEmbeddingContextLease(llama_context * c) : ctx(c) {}
    PremiseEmbeddingContextLease(const PremiseEmbeddingContextLease &) = delete;
    PremiseEmbeddingContextLease & operator=(const PremiseEmbeddingContextLease &) = delete;

    PremiseEmbeddingContextLease(PremiseEmbeddingContextLease && other) noexcept : ctx(other.ctx) {
        other.ctx = nullptr;
    }

    PremiseEmbeddingContextLease & operator=(PremiseEmbeddingContextLease && other) noexcept {
        if (this != &other) {
            release();
            ctx = other.ctx;
            other.ctx = nullptr;
        }
        return *this;
    }

    ~PremiseEmbeddingContextLease() {
        release();
    }

    void release() {
        if (!ctx || !g_premise_state) {
            ctx = nullptr;
            return;
        }
        {
            std::lock_guard<std::mutex> lk(g_premise_state->emb_mu);
            g_premise_state->idle_emb_ctxs.push_back(ctx);
        }
        g_premise_state->emb_cv.notify_one();
        ctx = nullptr;
    }
};

static PremiseEmbeddingContextLease premise_acquire_embedding_context() {
    if (!g_premise_state) {
        throw std::runtime_error("joint retrieval is not initialized");
    }
    std::unique_lock<std::mutex> lk(g_premise_state->emb_mu);
    g_premise_state->emb_cv.wait(lk, [] {
        return !g_premise_state || !g_premise_state->idle_emb_ctxs.empty();
    });
    if (!g_premise_state || g_premise_state->idle_emb_ctxs.empty()) {
        throw std::runtime_error("joint retrieval is not initialized");
    }
    llama_context * ctx = g_premise_state->idle_emb_ctxs.back();
    g_premise_state->idle_emb_ctxs.pop_back();
    return PremiseEmbeddingContextLease(ctx);
}

static void premise_store_pending_premises(int task_id, std::string sse) {
    {
        std::lock_guard<std::mutex> lk(g_premise_state->pending_mu);
        g_premise_state->task_requests.erase(task_id);
        g_premise_state->pending_premises[task_id] = std::move(sse);
    }
    g_premise_state->pending_cv.notify_all();
}

static std::string json_string_value(const json & data, const char * key, const std::string & def = {}) {
    auto it = data.find(key);
    return it != data.end() && it->is_string() ? it->get<std::string>() : def;
}

static int json_int_value(const json & data, const char * key, int def) {
    auto it = data.find(key);
    return it != data.end() && it->is_number_integer() ? it->get<int>() : def;
}

static std::vector<std::string> parse_string_array(const json & data, const char * key) {
    std::vector<std::string> out;
    auto it = data.find(key);
    if (it == data.end() || !it->is_array()) {
        return out;
    }
    for (const auto & item : *it) {
        if (item.is_string()) {
            out.push_back(item.get<std::string>());
        }
    }
    return out;
}

static std::vector<LeanDeclaration> parse_declarations(const json & data, const char * key = "declarations") {
    std::vector<LeanDeclaration> out;
    auto it = data.find(key);
    if (it == data.end() || !it->is_array()) {
        return out;
    }
    for (const auto & item : *it) {
        if (!item.is_object()) {
            continue;
        }
        LeanDeclaration decl;
        decl.name = json_string_value(item, "name");
        decl.decl = json_string_value(item, "decl");
        if (!decl.name.empty() && !decl.decl.empty()) {
            out.push_back(std::move(decl));
        }
    }
    return out;
}

static std::vector<llama_token> tokenize_text(const std::string & text) {
    llama_context * emb_ctx = premise_first_embedding_context();
    const llama_model * model = llama_get_model(emb_ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    int n_max = (int) text.size() + 64;
    std::vector<llama_token> tokens(std::max(8, n_max));
    int n = llama_tokenize(vocab, text.c_str(), (int) text.size(), tokens.data(), (int) tokens.size(), false, true);
    if (n < 0) {
        tokens.resize((size_t) -n);
        n = llama_tokenize(vocab, text.c_str(), (int) text.size(), tokens.data(), -n, false, true);
    }
    if (n <= 0) {
        throw std::runtime_error("tokenization produced no tokens");
    }
    tokens.resize((size_t) n);
    const int max_tokens = std::max(1, (int) llama_n_ctx(emb_ctx) - 1);
    if ((int) tokens.size() > max_tokens) {
        tokens.resize((size_t) max_tokens);
    }
    return tokens;
}

static std::vector<float> embed_tokens(std::vector<llama_token> tokens, bool append_emb) {
    if (!g_premise_state || g_premise_state->emb_ctxs.empty()) {
        throw std::runtime_error("joint retrieval is not initialized");
    }
    if (append_emb) {
        if (g_premise_state->emb_token_id < 0) {
            throw std::runtime_error("joint retrieval requires a model with [EMB] token");
        }
        tokens.push_back(g_premise_state->emb_token_id);
    }
    if (tokens.empty()) {
        throw std::runtime_error("cannot embed empty token sequence");
    }
    auto emb_lease = premise_acquire_embedding_context();
    llama_context * emb_ctx = emb_lease.ctx;

    const int max_tokens = (int) llama_n_ctx(emb_ctx);
    if ((int) tokens.size() > max_tokens) {
        tokens.erase(tokens.begin(), tokens.begin() + ((int) tokens.size() - max_tokens));
    }

    llama_memory_clear(llama_get_memory(emb_ctx), true);

    const enum llama_pooling_type pooling_type = llama_pooling_type(emb_ctx);
    llama_batch batch = llama_batch_init((int) tokens.size(), 0, 1);
    try {
        for (int i = 0; i < (int) tokens.size(); ++i) {
            const bool output = pooling_type != LLAMA_POOLING_TYPE_NONE || i == (int) tokens.size() - 1;
            common_batch_add(batch, tokens[i], i, { 0 }, output);
        }
        if (llama_decode(emb_ctx, batch) != 0) {
            throw std::runtime_error("embedding decode failed");
        }

        const int n_embd = g_premise_state->embedding_dim > 0 ? g_premise_state->embedding_dim : llama_model_n_embd_out(llama_get_model(emb_ctx));
        const float * raw = pooling_type == LLAMA_POOLING_TYPE_NONE
            ? llama_get_embeddings_ith(emb_ctx, batch.n_tokens - 1)
            : llama_get_embeddings_seq(emb_ctx, 0);
        if (!raw) {
            throw std::runtime_error("embedding output was not available");
        }

        std::vector<float> embd(n_embd);
        common_embd_normalize(raw, embd.data(), n_embd, 2);
        llama_batch_free(batch);
        return embd;
    } catch (...) {
        llama_batch_free(batch);
        throw;
    }
}

static std::vector<float> embed_text(const std::string & text, bool append_emb) {
    return embed_tokens(tokenize_text(text), append_emb);
}

static std::vector<LeanPremiseRecord> embed_declarations(
        const std::vector<LeanDeclaration> & declarations,
        const std::string & module) {
    std::vector<LeanPremiseRecord> out(declarations.size());
    std::vector<size_t> missing;
    missing.reserve(declarations.size());

    for (size_t i = 0; i < declarations.size(); ++i) {
        const auto & decl = declarations[i];
        out[i].name = decl.name;
        out[i].decl = decl.decl;
        out[i].module = module;
        missing.push_back(i);
    }

    if (missing.empty()) {
        return out;
    }

    const size_t n_workers = std::min(missing.size(), std::max<size_t>(1, g_premise_state ? g_premise_state->emb_ctxs.size() : 1));
    std::atomic<size_t> next{0};
    std::mutex error_mu;
    std::exception_ptr error;

    auto worker = [&]() {
        for (;;) {
            const size_t pos = next.fetch_add(1);
            if (pos >= missing.size()) {
                return;
            }

            const size_t index = missing[pos];
            try {
                out[index].embedding = embed_text(out[index].decl, false);
            } catch (...) {
                std::lock_guard<std::mutex> lk(error_mu);
                if (!error) {
                    error = std::current_exception();
                }
                next.store(missing.size());
                return;
            }
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(n_workers > 0 ? n_workers - 1 : 0);
    for (size_t i = 1; i < n_workers; ++i) {
        threads.emplace_back(worker);
    }
    worker();
    for (auto & thread : threads) {
        thread.join();
    }

    if (error) {
        std::rethrow_exception(error);
    }
    return out;
}

static void collect_module_declarations_locked(
        const std::string & module,
        std::unordered_set<std::string> & visited_modules,
        std::unordered_set<std::string> & seen_names,
        std::vector<const LeanPremiseRecord *> & out) {
    if (!visited_modules.insert(module).second) {
        return;
    }
    auto it = g_premise_state->module_cache.find(module);
    if (it == g_premise_state->module_cache.end()) {
        return;
    }
    for (const auto & imported : it->second.imports) {
        collect_module_declarations_locked(imported, visited_modules, seen_names, out);
    }
    for (const auto & decl : it->second.declarations) {
        if (seen_names.insert(decl.name).second) {
            out.push_back(&decl);
        }
    }
}

static std::vector<const LeanPremiseRecord *> collect_cached_candidates(
        const std::vector<std::string> & imports,
        std::unordered_set<std::string> & seen_names) {
    std::vector<const LeanPremiseRecord *> candidates;
    std::unordered_set<std::string> visited_modules;
    std::lock_guard<std::mutex> lk(g_premise_state->cache_mu);
    for (const auto & module : imports) {
        collect_module_declarations_locked(module, visited_modules, seen_names, candidates);
    }
    return candidates;
}

static json search_records_json(
        const std::vector<float> & query,
        const std::vector<const LeanPremiseRecord *> & cached,
        const std::vector<LeanPremiseRecord> & local,
        int top_k) {
    struct Scored {
        float score;
        const LeanPremiseRecord * record;
    };

    std::vector<Scored> scored;
    scored.reserve(cached.size() + local.size());
    auto add_record = [&](const LeanPremiseRecord & record) {
        float dot = 0.0f;
        for (int i = 0; i < (int) query.size(); ++i) {
            dot += query[i] * record.embedding[i];
        }
        scored.push_back({dot, &record});
    };
    for (const auto * record : cached) {
        add_record(*record);
    }
    for (const auto & record : local) {
        add_record(record);
    }

    const int k = std::min(top_k, (int) scored.size());
    if (k > 0) {
        std::partial_sort(scored.begin(), scored.begin() + k, scored.end(),
                [](const Scored & a, const Scored & b) { return a.score > b.score; });
    }

    json arr = json::array();
    for (int i = 0; i < k; ++i) {
        const auto * record = scored[i].record;
        arr.push_back({
            {"name", record->name},
            {"statement", record->decl},
            {"decl", record->decl},
            {"module", record->module},
            {"score", scored[i].score},
        });
    }
    return arr;
}

static json search_request_json(const PremiseRetrievalRequest & request, const std::vector<float> & query) {
    std::unordered_set<std::string> seen_names;
    auto cached = collect_cached_candidates(request.imports, seen_names);
    std::vector<LeanDeclaration> local_decls;
    local_decls.reserve(request.declarations.size());
    for (const auto & decl : request.declarations) {
        if (seen_names.insert(decl.name).second) {
            local_decls.push_back(decl);
        }
    }
    auto local = embed_declarations(local_decls, "");
    return search_records_json(query, cached, local, request.top_k);
}

static json search_global_json(const std::vector<float> & query, int top_k) {
    auto hits = g_premise_state->premise_index->search(query.data(), top_k);
    json arr = json::array();
    for (auto & [stmt, score] : hits) {
        arr.push_back({{"statement", stmt}, {"decl", stmt}, {"score", score}});
    }
    return arr;
}

std::string premise_task_created(int task_id, const json & data, bool stream, int n_cmpl) {
    const int top_k = stream && n_cmpl == 1 ? json_int_value(data, "retrieval_topk", json_int_value(data, "k", 5)) : -1;
    if (!g_premise_state) {
        return {};
    }

    if (!g_premise_state->joint_generation) {
        return "autoregressive generation requires a joint model with [EMB] token; use /select for premise retrieval";
    }

    if (top_k < 0) {
        return {};
    }

    PremiseRetrievalRequest request;
    request.top_k = top_k;
    request.imports = parse_string_array(data, "imports");
    request.declarations = parse_declarations(data);
    request.scoped = data.contains("imports") || data.contains("declarations");
    if (!request.scoped && !g_premise_state->premise_index) {
        return {};
    }

    std::lock_guard<std::mutex> lk(g_premise_state->pending_mu);
    g_premise_state->task_requests[task_id] = std::move(request);
    return {};
}

void premise_prefill_complete(int task_id, const std::vector<llama_token> & prompt_tokens) {
    if (!g_premise_state || g_premise_state->emb_ctxs.empty() || !g_premise_state->joint_generation) {
        return;
    }

    PremiseRetrievalRequest request;
    {
        std::lock_guard<std::mutex> lk(g_premise_state->pending_mu);
        auto it = g_premise_state->task_requests.find(task_id);
        if (it == g_premise_state->task_requests.end()) {
            return;
        }
        request = it->second;
    }

    if (request.top_k <= 0) {
        premise_store_pending_premises(task_id, {});
        return;
    }

    try {
        auto query = embed_tokens(prompt_tokens, true);
        json arr = request.scoped
            ? search_request_json(request, query)
            : (g_premise_state->premise_index ? search_global_json(query, request.top_k) : json::array());

        json evt = {{"type", "premises"}, {"premises", arr}};
        std::string sse = "data: " + evt.dump() + "\n\n";

        premise_store_pending_premises(task_id, std::move(sse));
    } catch (const std::exception & e) {
        SRV_WRN("joint retrieval: %s\n", e.what());
        premise_store_pending_premises(task_id, {});
    }
}

std::string premise_take_initial_stream_prefix(int task_id) {
    if (!g_premise_state || task_id < 0) {
        return {};
    }

    std::unique_lock<std::mutex> lk(g_premise_state->pending_mu);
    g_premise_state->pending_cv.wait_for(lk, std::chrono::seconds(10), [task_id] {
        return g_premise_state->pending_premises.find(task_id) != g_premise_state->pending_premises.end() ||
               g_premise_state->task_requests.find(task_id) == g_premise_state->task_requests.end();
    });

    auto it = g_premise_state->pending_premises.find(task_id);
    if (it == g_premise_state->pending_premises.end()) {
        g_premise_state->task_requests.erase(task_id);
        return {};
    }

    std::string sse = std::move(it->second);
    g_premise_state->pending_premises.erase(it);
    return sse;
}

json premise_get_module_version(const json & data) {
    const std::string module = json_string_value(data, "module");
    if (!g_premise_state || module.empty()) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lk(g_premise_state->cache_mu);
    auto it = g_premise_state->module_cache.find(module);
    return it == g_premise_state->module_cache.end() ? json(nullptr) : json(it->second.version_token);
}

json premise_cache_module(const json & data) {
    if (!g_premise_state) {
        throw std::runtime_error("joint retrieval is not initialized");
    }
    const std::string module = json_string_value(data, "module");
    const std::string token = json_string_value(data, "token");
    if (module.empty()) {
        throw std::runtime_error("missing module");
    }
    if (token.empty()) {
        throw std::runtime_error("missing token");
    }

    {
        std::lock_guard<std::mutex> lk(g_premise_state->cache_mu);
        auto it = g_premise_state->module_cache.find(module);
        if (it != g_premise_state->module_cache.end() && it->second.version_token == token) {
            return json{{"ok", true}};
        }
    }

    const std::vector<std::string> imports = parse_string_array(data, "imports");
    LeanModuleCacheEntry entry;
    entry.version_token = token;
    entry.imports = imports;
    entry.declarations = embed_declarations(parse_declarations(data), module);

    if (g_premise_state->embed_cache) {
        std::vector<std::string> names;
        std::vector<std::vector<float>> embeddings;
        names.reserve(entry.declarations.size());
        embeddings.reserve(entry.declarations.size());
        for (const auto & decl : entry.declarations) {
            names.push_back(decl.name);
            embeddings.push_back(decl.embedding);
        }
        g_premise_state->embed_cache->replace_module(module, token, imports, names, embeddings);
    }

    {
        std::lock_guard<std::mutex> lk(g_premise_state->cache_mu);
        g_premise_state->module_cache[module] = std::move(entry);
    }
    if (g_premise_state->embed_cache) {
        g_premise_state->embed_cache->maybe_save(false);
    }
    return json{{"ok", true}};
}

json premise_retrieve(const json & data) {
    if (!g_premise_state) {
        throw std::runtime_error("joint retrieval is not initialized");
    }
    PremiseRetrievalRequest request;
    request.top_k = json_int_value(data, "k", 5);
    request.imports = parse_string_array(data, "imports");
    request.declarations = parse_declarations(data);
    request.scoped = true;

    const std::string goal = json_string_value(data, "goal");
    if (goal.empty()) {
        throw std::runtime_error("missing goal");
    }

    auto query = embed_text(goal, g_premise_state->joint_generation);
    json premises = search_request_json(request, query);
    json suggestions = json::array();
    for (const auto & premise : premises) {
        suggestions.push_back({{"name", premise.value("name", "")}, {"score", premise.value("score", 0.0f)}});
    }
    return suggestions;
}
