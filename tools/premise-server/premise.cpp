#include "premise.hpp"
#include "premise-index.hpp"
#include "server-common.h"
#include "server-context.h"
#include "server-http.h"

#include "common.h"
#include "llama.h"
#include "log.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <vector>

using json = nlohmann::ordered_json;

// ---------------------------------------------------------------------------
// Process state
// ---------------------------------------------------------------------------

struct LocalDecl {
    std::string name;
    std::string body; // source text used only to embed; never stored in the index
};

struct SelectParams {
    int top_k = 0;
    bool use_scoped_search = false; // true if request has imports and/or locals
    std::vector<std::string> imports;
    std::vector<LocalDecl> local_decls;
};

struct PremiseState {
    llama_context * embed_ctx = nullptr;
    std::unique_ptr<PremiseIndex> index;
    llama_token embed_token = -1; // optional [EMB] for joint models
    int embed_dim = 0;
    bool joint_retrieval = false;
    std::mutex embed_mutex;

    // Joint stream handshake only (server loop <-> HTTP).
    std::mutex joint_mutex;
    std::condition_variable joint_cv;
    std::unordered_map<int, SelectParams> joint_select_by_task;
    std::unordered_map<int, std::string> joint_sse_by_task;
};

static PremiseState * g_premise = nullptr;

// ---------------------------------------------------------------------------
// Small JSON / HTTP helpers
// ---------------------------------------------------------------------------

static std::string json_string(const json & obj, const char * key, const std::string & fallback = {}) {
    auto it = obj.find(key);
    return it != obj.end() && it->is_string() ? it->get<std::string>() : fallback;
}

static int json_int(const json & obj, const char * key, int fallback) {
    auto it = obj.find(key);
    return it != obj.end() && it->is_number_integer() ? it->get<int>() : fallback;
}

static std::vector<std::string> json_string_array(const json & obj, const char * key) {
    std::vector<std::string> out;
    auto it = obj.find(key);
    if (it == obj.end() || !it->is_array()) {
        return out;
    }
    for (const auto & item : *it) {
        if (item.is_string()) {
            out.push_back(item.get<std::string>());
        }
    }
    return out;
}

static std::vector<LocalDecl> json_local_decls(const json & obj) {
    std::vector<LocalDecl> out;
    auto it = obj.find("declarations");
    if (it == obj.end() || !it->is_array()) {
        return out;
    }
    for (const auto & item : *it) {
        if (!item.is_object()) {
            continue;
        }
        LocalDecl decl{json_string(item, "name"), json_string(item, "decl")};
        if (!decl.name.empty() && !decl.body.empty()) {
            out.push_back(std::move(decl));
        }
    }
    return out;
}

static server_http_res_ptr make_json_response(const json & body) {
    auto res = std::make_unique<server_http_res>();
    res->data = body.dump();
    return res;
}

static server_http_res_ptr make_error_response(const std::exception & e) {
    auto res = std::make_unique<server_http_res>();
    res->status = 400;
    res->data = json{{"error", {{"message", e.what()}}}}.dump();
    return res;
}

static json hits_to_json(const std::vector<PremiseIndex::Hit> & hits, bool include_module) {
    json arr = json::array();
    for (const auto & hit : hits) {
        json item = {{"name", hit.name}, {"score", hit.score}};
        if (include_module && !hit.module.empty()) {
            item["module"] = hit.module;
        }
        arr.push_back(std::move(item));
    }
    return arr;
}

// ---------------------------------------------------------------------------
// Setup / cleanup
// ---------------------------------------------------------------------------

void premise_setup(server_context & ctx_server,
                   const std::string & vecs_path,
                   const std::string & names_path,
                   PremiseMode mode) {
    const bool has_cache_paths = !vecs_path.empty() && !names_path.empty();
    if (!has_cache_paths && mode == PremiseMode::Auto) {
        return;
    }
    if (vecs_path.empty() != names_path.empty()) {
        SRV_WRN("%s", "--index-vecs and --index-names must both be set\n");
    }

    auto * server_ctx = ctx_server.get_llama_context();
    if (!server_ctx) {
        return;
    }

    llama_model * model = const_cast<llama_model *>(llama_get_model(server_ctx));
    const llama_vocab * vocab = llama_model_get_vocab(model);

    llama_token embed_token = -1;
    char piece[64];
    for (int i = llama_vocab_n_tokens(vocab) - 1; i >= 0; --i) {
        const int n = llama_token_to_piece(vocab, i, piece, sizeof(piece), 0, true);
        if (n > 0 && std::string(piece, n) == "[EMB]") {
            embed_token = i;
            break;
        }
    }
    if (mode == PremiseMode::Joint && embed_token < 0) {
        SRV_ERR("%s", "joint mode needs [EMB] token\n");
        return;
    }

    // One embedding context. n_seq_max > 1 so /cache can pack many decls per decode.
    llama_context_params embed_params = llama_context_default_params();
    embed_params.n_ctx = std::min(2048u, (uint32_t) std::max(1, llama_model_n_ctx_train(model)));
    embed_params.n_batch = embed_params.n_ctx;
    embed_params.n_ubatch = embed_params.n_ctx;
    embed_params.n_seq_max = std::min(embed_params.n_ctx, (uint32_t) llama_max_parallel_sequences());
    embed_params.embeddings = true;
    embed_params.kv_unified = true;
    embed_params.pooling_type = llama_pooling_type(server_ctx);

    llama_context * embed_ctx = llama_init_from_model(model, embed_params);
    if (!embed_ctx) {
        SRV_ERR("%s", "failed to create embedding context\n");
        return;
    }

    try {
        const int embed_dim = llama_model_n_embd_out(model);
        auto index = std::make_unique<PremiseIndex>();
        if (mode == PremiseMode::Embedding) {
            if (has_cache_paths) {
                index->load_cache(vecs_path, names_path, embed_dim);
            } else {
                index->initialize_empty(embed_dim);
            }
        } else if (has_cache_paths) {
            index->load_offline(vecs_path.c_str(), names_path.c_str());
            if (index->embedding_dim() != embed_dim) {
                SRV_ERR("%s", "index dim mismatch; offline index disabled\n");
                index.reset();
            }
        } else {
            index.reset();
        }

        auto state = std::make_unique<PremiseState>();
        state->embed_ctx = embed_ctx;
        state->embed_token = embed_token;
        state->joint_retrieval = mode != PremiseMode::Embedding && embed_token >= 0;
        state->embed_dim = embed_dim;
        state->index = std::move(index);
        g_premise = state.release();

        SRV_INF("premise ready: rows=%d joint=%d\n",
                g_premise->index ? (int) g_premise->index->size() : 0,
                (int) g_premise->joint_retrieval);
    } catch (const std::exception & e) {
        SRV_ERR("premise init failed: %s\n", e.what());
        llama_free(embed_ctx);
    }
}

void premise_cleanup() {
    if (!g_premise) {
        return;
    }
    if (g_premise->index) {
        g_premise->index->flush(true);
    }
    if (g_premise->embed_ctx) {
        llama_free(g_premise->embed_ctx);
    }
    delete g_premise;
    g_premise = nullptr;
}

// ---------------------------------------------------------------------------
// Embedding
//
// Only multi-declaration batching matters (one /cache module's decls).
// One context; embed_mutex serializes any overlapping callers.
// ---------------------------------------------------------------------------

// Pack independent token sequences into llama_batch slots and decode.
static std::vector<std::vector<float>> embed_token_batch(
        std::vector<std::vector<llama_token>> sequences,
        bool append_embed_token) {
    if (!g_premise || !g_premise->embed_ctx) {
        throw std::runtime_error("premise not initialized");
    }
    if (sequences.empty()) {
        return {};
    }

    std::lock_guard<std::mutex> lock(g_premise->embed_mutex);
    llama_context * ctx = g_premise->embed_ctx;

    if (append_embed_token) {
        if (g_premise->embed_token < 0) {
            throw std::runtime_error("no [EMB] token");
        }
        for (auto & tokens : sequences) {
            tokens.push_back(g_premise->embed_token);
        }
    }

    const int n_batch = (int) llama_n_batch(ctx);
    const int n_seq_max = (int) llama_n_seq_max(ctx);
    const int max_tokens = std::min((int) llama_n_ctx(ctx), n_batch);
    const enum llama_pooling_type pooling = llama_pooling_type(ctx);
    const int n_embd = g_premise->embed_dim > 0
        ? g_premise->embed_dim
        : llama_model_n_embd_out(llama_get_model(ctx));

    for (auto & tokens : sequences) {
        if (tokens.empty()) {
            throw std::runtime_error("empty token sequence");
        }
        if ((int) tokens.size() > max_tokens) {
            tokens.erase(tokens.begin(), tokens.begin() + ((int) tokens.size() - max_tokens));
        }
    }

    std::vector<std::vector<float>> embeddings(sequences.size(), std::vector<float>(n_embd));
    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    try {
        for (size_t begin = 0; begin < sequences.size(); ) {
            common_batch_clear(batch);
            std::vector<int> last_token_pos;
            size_t end = begin;

            // Fit as many sequences as n_batch / n_seq_max allow.
            while (end < sequences.size() &&
                    end - begin < (size_t) n_seq_max &&
                    batch.n_tokens + (int) sequences[end].size() <= n_batch) {
                const llama_seq_id seq_id = (llama_seq_id) (end - begin);
                const auto & tokens = sequences[end];
                for (int i = 0; i < (int) tokens.size(); ++i) {
                    const bool output =
                        pooling != LLAMA_POOLING_TYPE_NONE || i + 1 == (int) tokens.size();
                    common_batch_add(batch, tokens[i], i, { seq_id }, output);
                }
                last_token_pos.push_back(batch.n_tokens - 1);
                ++end;
            }

            llama_memory_clear(llama_get_memory(ctx), true);
            if (llama_decode(ctx, batch) != 0) {
                throw std::runtime_error("embedding decode failed");
            }

            for (size_t i = begin; i < end; ++i) {
                const int seq_id = (int) (i - begin);
                const float * raw = pooling == LLAMA_POOLING_TYPE_NONE
                    ? llama_get_embeddings_ith(ctx, last_token_pos[(size_t) seq_id])
                    : llama_get_embeddings_seq(ctx, seq_id);
                if (!raw) {
                    throw std::runtime_error("missing embedding");
                }
                common_embd_normalize(raw, embeddings[i].data(), n_embd, 2);
            }
            begin = end;
        }
    } catch (...) {
        llama_batch_free(batch);
        throw;
    }
    llama_batch_free(batch);
    return embeddings;
}

static std::vector<llama_token> tokenize_for_embed(const std::string & text) {
    auto tokens = common_tokenize(g_premise->embed_ctx, text, /*add_special*/ false, /*parse_special*/ true);
    const int max_tokens = std::max(1, (int) llama_n_ctx(g_premise->embed_ctx) - 1);
    if ((int) tokens.size() > max_tokens) {
        tokens.resize((size_t) max_tokens);
    }
    return tokens;
}

static std::vector<PremiseIndex::Candidate> embed_declarations(
        const std::vector<LocalDecl> & decls,
        const std::string & module) {
    std::vector<std::vector<llama_token>> sequences;
    sequences.reserve(decls.size());
    std::vector<PremiseIndex::Candidate> out(decls.size());
    for (size_t i = 0; i < decls.size(); ++i) {
        out[i].name = decls[i].name;
        out[i].module = module;
        sequences.push_back(tokenize_for_embed(decls[i].body));
    }
    auto vectors = embed_token_batch(std::move(sequences), /*append_embed_token*/ false);
    for (size_t i = 0; i < out.size(); ++i) {
        out[i].embedding = std::move(vectors[i]);
    }
    return out;
}

// ---------------------------------------------------------------------------
// Select core (shared by HTTP /select and joint hooks)
// ---------------------------------------------------------------------------

static SelectParams select_params_from_json(const json & data, int top_k) {
    SelectParams params;
    params.top_k = top_k;
    params.imports = json_string_array(data, "imports");
    params.local_decls = json_local_decls(data);
    // Key presence chooses scoped vs global, including empty arrays.
    params.use_scoped_search = data.contains("imports") || data.contains("declarations");
    return params;
}

static std::vector<PremiseIndex::Hit> select_hits(
        const SelectParams & params,
        const std::vector<float> & query) {
    if (!g_premise->index) {
        return {};
    }
    if (params.use_scoped_search) {
        auto locals = embed_declarations(params.local_decls, "");
        return g_premise->index->search_scoped(
                query.data(), params.imports, locals, params.top_k);
    }
    return g_premise->index->search_global(query.data(), params.top_k);
}

// ---------------------------------------------------------------------------
// Joint hooks
//
// task_created: store SelectParams for this generation task
// prefill_complete: embed prompt (+[EMB]), run select, store SSE prefix
// take_initial_stream_prefix: HTTP waits for that SSE (or timeout)
// ---------------------------------------------------------------------------

std::string premise_task_created(int task_id, const json & data, bool stream, int n_cmpl) {
    if (!g_premise) {
        return {};
    }
    if (!g_premise->joint_retrieval) {
        return "joint model with [EMB] required; use /select";
    }
    // Only single-completion streams request premises.
    const int top_k = (stream && n_cmpl == 1)
        ? json_int(data, "retrieval_topk", json_int(data, "k", 5))
        : -1;
    if (top_k < 0) {
        return {};
    }
    SelectParams params = select_params_from_json(data, top_k);
    if (!params.use_scoped_search && !g_premise->index) {
        return {};
    }
    std::lock_guard<std::mutex> lock(g_premise->joint_mutex);
    g_premise->joint_select_by_task[task_id] = std::move(params);
    return {};
}

void premise_prefill_complete(int task_id, const std::vector<llama_token> & prompt_tokens) {
    if (!g_premise || !g_premise->embed_ctx || !g_premise->joint_retrieval) {
        return;
    }

    SelectParams params;
    {
        std::lock_guard<std::mutex> lock(g_premise->joint_mutex);
        auto it = g_premise->joint_select_by_task.find(task_id);
        if (it == g_premise->joint_select_by_task.end()) {
            return;
        }
        params = it->second;
    }

    auto finish = [&](std::string sse) {
        std::lock_guard<std::mutex> lock(g_premise->joint_mutex);
        g_premise->joint_select_by_task.erase(task_id);
        g_premise->joint_sse_by_task[task_id] = std::move(sse);
        g_premise->joint_cv.notify_all();
    };

    if (params.top_k <= 0) {
        finish({});
        return;
    }

    try {
        auto query = embed_token_batch({ prompt_tokens }, /*append_embed_token*/ true).front();
        json event = {
            {"type", "premises"},
            {"premises", hits_to_json(select_hits(params, query), /*include_module*/ true)},
        };
        finish("data: " + event.dump() + "\n\n");
    } catch (const std::exception & e) {
        SRV_WRN("joint retrieval: %s\n", e.what());
        finish({});
    }
}

std::string premise_take_initial_stream_prefix(int task_id) {
    if (!g_premise || task_id < 0) {
        return {};
    }
    std::unique_lock<std::mutex> lock(g_premise->joint_mutex);
    g_premise->joint_cv.wait_for(lock, std::chrono::seconds(10), [task_id] {
        return g_premise->joint_sse_by_task.count(task_id) > 0 ||
               g_premise->joint_select_by_task.count(task_id) == 0;
    });
    auto it = g_premise->joint_sse_by_task.find(task_id);
    if (it == g_premise->joint_sse_by_task.end()) {
        g_premise->joint_select_by_task.erase(task_id);
        return {};
    }
    std::string sse = std::move(it->second);
    g_premise->joint_sse_by_task.erase(it);
    return sse;
}

// ---------------------------------------------------------------------------
// HTTP: POST /version, /cache, /select
// ---------------------------------------------------------------------------

// /version { "module": "..." } -> token string | null
static json handle_version(const json & body) {
    const std::string module = json_string(body, "module");
    if (!g_premise || !g_premise->index || module.empty()) {
        return nullptr;
    }
    const std::string token = g_premise->index->get_module_version(module);
    return token.empty() ? json(nullptr) : json(token);
}

// /cache one module; declarations[] are batch-embedded then replace_module.
static json handle_cache(const json & body) {
    if (!g_premise || !g_premise->index) {
        throw std::runtime_error("premise not initialized");
    }
    const std::string module = json_string(body, "module");
    const std::string token = json_string(body, "token");
    if (module.empty() || token.empty()) {
        throw std::runtime_error("missing module or token");
    }
    if (g_premise->index->get_module_version(module) == token) {
        return json{{"ok", true}};
    }
    auto candidates = embed_declarations(json_local_decls(body), module);
    g_premise->index->replace_module(module, token, json_string_array(body, "imports"), candidates);
    g_premise->index->flush(false);
    return json{{"ok", true}};
}

// /select goal + optional imports/locals -> [{name,score}, ...]
static json handle_select(const json & body) {
    if (!g_premise) {
        throw std::runtime_error("premise not initialized");
    }
    const std::string goal = json_string(body, "goal");
    if (goal.empty()) {
        throw std::runtime_error("missing goal");
    }
    SelectParams params = select_params_from_json(body, json_int(body, "k", 5));
    auto query = embed_token_batch(
            { tokenize_for_embed(goal) },
            /*append_embed_token*/ g_premise->joint_retrieval).front();
    return hits_to_json(select_hits(params, query), /*include_module*/ false);
}

void premise_register_http_routes(const server_http_context & ctx_http) {
    auto post_json = [](auto handler) {
        return [handler](const server_http_req & req) -> server_http_res_ptr {
            try {
                json body = req.body.empty() ? json::object() : json::parse(req.body);
                return make_json_response(handler(body));
            } catch (const std::exception & e) {
                return make_error_response(e);
            }
        };
    };

    ctx_http.post("/version", post_json(handle_version));
    ctx_http.post("/cache",   post_json(handle_cache));
    ctx_http.post("/select",  post_json(handle_select));
}
