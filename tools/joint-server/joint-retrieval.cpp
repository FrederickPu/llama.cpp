#include "joint-retrieval.hpp"

#include "common.h"
#include "server-common.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <unordered_set>

using json = nlohmann::ordered_json;

JointRetrievalState * g_joint_state = nullptr;

static void joint_store_pending_premises(int task_id, std::string sse) {
    {
        std::lock_guard<std::mutex> lk(g_joint_state->pending_mu);
        g_joint_state->task_requests.erase(task_id);
        g_joint_state->pending_premises[task_id] = std::move(sse);
    }
    g_joint_state->pending_cv.notify_all();
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
    if (!g_joint_state || !g_joint_state->emb_ctx) {
        throw std::runtime_error("joint retrieval is not initialized");
    }
    const llama_model * model = llama_get_model(g_joint_state->emb_ctx);
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
    const int max_tokens = std::max(1, (int) llama_n_ctx(g_joint_state->emb_ctx) - 1);
    if ((int) tokens.size() > max_tokens) {
        tokens.resize((size_t) max_tokens);
    }
    return tokens;
}

static std::vector<float> embed_tokens(std::vector<llama_token> tokens, bool append_emb) {
    if (!g_joint_state || !g_joint_state->emb_ctx) {
        throw std::runtime_error("joint retrieval is not initialized");
    }
    if (append_emb) {
        if (g_joint_state->emb_token_id < 0) {
            throw std::runtime_error("joint retrieval requires a model with [EMB] token");
        }
        tokens.push_back(g_joint_state->emb_token_id);
    }
    if (tokens.empty()) {
        throw std::runtime_error("cannot embed empty token sequence");
    }
    const int max_tokens = (int) llama_n_ctx(g_joint_state->emb_ctx);
    if ((int) tokens.size() > max_tokens) {
        tokens.erase(tokens.begin(), tokens.begin() + ((int) tokens.size() - max_tokens));
    }

    std::lock_guard<std::mutex> emb_lock(g_joint_state->emb_mu);
    llama_memory_clear(llama_get_memory(g_joint_state->emb_ctx), true);

    const enum llama_pooling_type pooling_type = llama_pooling_type(g_joint_state->emb_ctx);
    llama_batch batch = llama_batch_init((int) tokens.size(), 0, 1);
    try {
        for (int i = 0; i < (int) tokens.size(); ++i) {
            const bool output = pooling_type != LLAMA_POOLING_TYPE_NONE || i == (int) tokens.size() - 1;
            common_batch_add(batch, tokens[i], i, { 0 }, output);
        }
        if (llama_decode(g_joint_state->emb_ctx, batch) != 0) {
            throw std::runtime_error("embedding decode failed");
        }

        const int n_embd = g_joint_state->embedding_dim > 0 ? g_joint_state->embedding_dim : llama_model_n_embd_out(llama_get_model(g_joint_state->emb_ctx));
        const float * raw = pooling_type == LLAMA_POOLING_TYPE_NONE
            ? llama_get_embeddings_ith(g_joint_state->emb_ctx, batch.n_tokens - 1)
            : llama_get_embeddings_seq(g_joint_state->emb_ctx, 0);
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
    std::vector<LeanPremiseRecord> out;
    out.reserve(declarations.size());
    for (const auto & decl : declarations) {
        LeanPremiseRecord record;
        record.name = decl.name;
        record.decl = decl.decl;
        record.module = module;
        record.embedding = embed_text(decl.decl, false);
        out.push_back(std::move(record));
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
    auto it = g_joint_state->module_cache.find(module);
    if (it == g_joint_state->module_cache.end()) {
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
    std::lock_guard<std::mutex> lk(g_joint_state->cache_mu);
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

static json search_request_json(const JointRetrievalRequest & request, const std::vector<float> & query) {
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
    auto hits = g_joint_state->premise_index->search(query.data(), top_k);
    json arr = json::array();
    for (auto & [stmt, score] : hits) {
        arr.push_back({{"statement", stmt}, {"decl", stmt}, {"score", score}});
    }
    return arr;
}

std::string joint_task_created(int task_id, const json & data, bool stream, int n_cmpl) {
    const int top_k = stream && n_cmpl == 1 ? json_int_value(data, "retrieval_topk", json_int_value(data, "k", 5)) : -1;
    if (!g_joint_state) {
        return {};
    }

    if (!g_joint_state->joint_generation) {
        return "autoregressive generation requires a joint model with [EMB] token; use /select for premise retrieval";
    }

    if (top_k < 0) {
        return {};
    }

    JointRetrievalRequest request;
    request.top_k = top_k;
    request.imports = parse_string_array(data, "imports");
    request.declarations = parse_declarations(data);
    request.scoped = data.contains("imports") || data.contains("declarations");
    if (!request.scoped && !g_joint_state->premise_index) {
        return {};
    }

    std::lock_guard<std::mutex> lk(g_joint_state->pending_mu);
    g_joint_state->task_requests[task_id] = std::move(request);
    return {};
}

void joint_prefill_complete(int task_id, const std::vector<llama_token> & prompt_tokens) {
    if (!g_joint_state || !g_joint_state->emb_ctx || !g_joint_state->joint_generation) {
        return;
    }

    JointRetrievalRequest request;
    {
        std::lock_guard<std::mutex> lk(g_joint_state->pending_mu);
        auto it = g_joint_state->task_requests.find(task_id);
        if (it == g_joint_state->task_requests.end()) {
            return;
        }
        request = it->second;
    }

    if (request.top_k <= 0) {
        joint_store_pending_premises(task_id, {});
        return;
    }

    try {
        auto query = embed_tokens(prompt_tokens, true);
        json arr = request.scoped
            ? search_request_json(request, query)
            : (g_joint_state->premise_index ? search_global_json(query, request.top_k) : json::array());

        json evt = {{"type", "premises"}, {"premises", arr}};
        std::string sse = "data: " + evt.dump() + "\n\n";

        joint_store_pending_premises(task_id, std::move(sse));
    } catch (const std::exception & e) {
        SRV_WRN("joint retrieval: %s\n", e.what());
        joint_store_pending_premises(task_id, {});
    }
}

std::string joint_take_initial_stream_prefix(int task_id) {
    if (!g_joint_state || task_id < 0) {
        return {};
    }

    std::unique_lock<std::mutex> lk(g_joint_state->pending_mu);
    g_joint_state->pending_cv.wait_for(lk, std::chrono::seconds(10), [task_id] {
        return g_joint_state->pending_premises.find(task_id) != g_joint_state->pending_premises.end() ||
               g_joint_state->task_requests.find(task_id) == g_joint_state->task_requests.end();
    });

    auto it = g_joint_state->pending_premises.find(task_id);
    if (it == g_joint_state->pending_premises.end()) {
        g_joint_state->task_requests.erase(task_id);
        return {};
    }

    std::string sse = std::move(it->second);
    g_joint_state->pending_premises.erase(it);
    return sse;
}

json joint_get_module_version(const json & data) {
    const std::string module = json_string_value(data, "module");
    if (!g_joint_state || module.empty()) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lk(g_joint_state->cache_mu);
    auto it = g_joint_state->module_cache.find(module);
    return it == g_joint_state->module_cache.end() ? json(nullptr) : json(it->second.version_token);
}

json joint_cache_module(const json & data) {
    if (!g_joint_state) {
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

    LeanModuleCacheEntry entry;
    entry.version_token = token;
    entry.imports = parse_string_array(data, "imports");
    entry.declarations = embed_declarations(parse_declarations(data), module);

    {
        std::lock_guard<std::mutex> lk(g_joint_state->cache_mu);
        g_joint_state->module_cache[module] = std::move(entry);
    }
    return json{{"ok", true}};
}

json joint_retrieve(const json & data) {
    if (!g_joint_state) {
        throw std::runtime_error("joint retrieval is not initialized");
    }
    JointRetrievalRequest request;
    request.top_k = json_int_value(data, "k", 5);
    request.imports = parse_string_array(data, "imports");
    request.declarations = parse_declarations(data);
    request.scoped = true;

    const std::string goal = json_string_value(data, "goal");
    if (goal.empty()) {
        throw std::runtime_error("missing goal");
    }

    auto query = embed_text(goal, g_joint_state->joint_generation);
    json premises = search_request_json(request, query);
    json suggestions = json::array();
    for (const auto & premise : premises) {
        suggestions.push_back({{"name", premise.value("name", "")}, {"score", premise.value("score", 0.0f)}});
    }
    return suggestions;
}
