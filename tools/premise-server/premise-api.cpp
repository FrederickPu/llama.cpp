#include "premise-retrieval-internal.hpp"

#include "server-common.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <stdexcept>

using json = nlohmann::ordered_json;

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

static json hits_json(const std::vector<PremiseIndex::Hit> & hits) {
    json arr = json::array();
    for (const auto & hit : hits) {
        json item = {
            {"statement", hit.statement},
            {"decl", hit.decl},
            {"score", hit.score},
        };
        if (!hit.name.empty()) {
            item["name"] = hit.name;
        }
        if (!hit.module.empty()) {
            item["module"] = hit.module;
        }
        arr.push_back(std::move(item));
    }
    return arr;
}

static json search_request_json(const PremiseRetrievalRequest & request, const std::vector<float> & query) {
    if (!g_premise_state->premise_index) {
        return json::array();
    }
    auto local = premise_embed_declarations(request.declarations, "");
    return hits_json(g_premise_state->premise_index->search(query.data(), request.imports, local, request.top_k));
}

static json search_global_json(const std::vector<float> & query, int top_k) {
    if (!g_premise_state->premise_index) {
        return json::array();
    }
    return hits_json(g_premise_state->premise_index->search(query.data(), top_k));
}

static bool cache_module_if_stale(const std::string & module,
                                  const std::string & token,
                                  const std::vector<std::string> & imports,
                                  const std::vector<LeanDeclaration> & parsed_declarations) {
    if (!g_premise_state->premise_index) {
        throw std::runtime_error("premise index is not initialized");
    }
    if (g_premise_state->premise_index->module_version(module) == token) {
        return false;
    }

    auto declarations = premise_embed_declarations(parsed_declarations, module);
    if (g_premise_state->embed_cache) {
        std::vector<std::string> names;
        std::vector<std::vector<float>> embeddings;
        names.reserve(declarations.size());
        embeddings.reserve(declarations.size());
        for (const auto & decl : declarations) {
            names.push_back(decl.name);
            embeddings.push_back(decl.embedding);
        }
        g_premise_state->embed_cache->replace_module(module, token, imports, names, embeddings);
    }
    g_premise_state->premise_index->replace_module(module, token, imports, declarations);
    return true;
}

// Joint-generation hook API.

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
        auto query = premise_embed_tokens(prompt_tokens, true);
        json arr = request.scoped
            ? search_request_json(request, query)
            : search_global_json(query, request.top_k);

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

// /version

json premise_api_version(const json & data) {
    const std::string module = json_string_value(data, "module");
    if (!g_premise_state || !g_premise_state->premise_index || module.empty()) {
        return nullptr;
    }
    const std::string token = g_premise_state->premise_index->module_version(module);
    return token.empty() ? json(nullptr) : json(token);
}

json premise_api_version_batch(const json & data) {
    json result = json::object();
    if (!g_premise_state || !g_premise_state->premise_index) {
        return result;
    }
    auto it = data.find("modules");
    if (it == data.end() || !it->is_array()) {
        return result;
    }
    std::vector<std::string> modules;
    for (const auto & item : *it) {
        if (item.is_string()) {
            modules.push_back(item.get<std::string>());
        }
    }
    const auto versions = g_premise_state->premise_index->module_versions(modules);
    for (const auto & module : modules) {
        auto version = versions.find(module);
        result[module] = version == versions.end() ? json(nullptr) : json(version->second);
    }
    return result;
}

// /cache

json premise_api_cache(const json & data) {
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

    const std::vector<std::string> imports = parse_string_array(data, "imports");
    const bool changed = cache_module_if_stale(module, token, imports, parse_declarations(data));
    if (changed && g_premise_state->embed_cache) {
        g_premise_state->embed_cache->maybe_save(false);
    }
    return json{{"ok", true}};
}

json premise_api_cache_batch(const json & data) {
    if (!g_premise_state) {
        throw std::runtime_error("joint retrieval is not initialized");
    }
    if (!g_premise_state->premise_index) {
        throw std::runtime_error("premise index is not initialized");
    }
    auto it = data.find("modules");
    if (it == data.end() || !it->is_array()) {
        throw std::runtime_error("missing modules array");
    }

    int count = 0;
    for (const auto & item : *it) {
        if (!item.is_object()) {
            continue;
        }
        const std::string module = json_string_value(item, "module");
        const std::string token = json_string_value(item, "token");
        if (module.empty() || token.empty()) {
            continue;
        }
        const std::vector<std::string> imports = parse_string_array(item, "imports");
        if (cache_module_if_stale(module, token, imports, parse_declarations(item))) {
            count++;
        }
    }

    if (count > 0 && g_premise_state->embed_cache) {
        g_premise_state->embed_cache->maybe_save(false);
    }
    return json{{"ok", true}, {"count", count}};
}

// /select

json premise_api_select(const json & data) {
    if (!g_premise_state) {
        throw std::runtime_error("joint retrieval is not initialized");
    }
    PremiseRetrievalRequest request;
    request.top_k = json_int_value(data, "k", 5);
    request.imports = parse_string_array(data, "imports");
    request.declarations = parse_declarations(data);
    request.scoped = data.contains("imports") || data.contains("declarations");

    const std::string goal = json_string_value(data, "goal");
    if (goal.empty()) {
        throw std::runtime_error("missing goal");
    }

    auto query = premise_embed_text(goal, g_premise_state->joint_generation);
    json premises = request.scoped
        ? search_request_json(request, query)
        : search_global_json(query, request.top_k);
    json suggestions = json::array();
    for (const auto & premise : premises) {
        suggestions.push_back({{"name", premise.value("name", "")}, {"score", premise.value("score", 0.0f)}});
    }
    return suggestions;
}
