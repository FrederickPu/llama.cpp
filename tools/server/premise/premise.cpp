#include "premise.hpp"
#include "premise-index.hpp"
#include "server-common.h"
#include "server-context.h"
#include "server-http.h"

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "log.h"

#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <clocale>
#include <csignal>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#endif

using json = nlohmann::ordered_json;

// ---------------------------------------------------------------------------
// Process state
// ---------------------------------------------------------------------------

struct LocalDecl {
    std::string name;
    std::string body; // embed only; never stored in the index
};

struct SelectParams {
    int top_k = 0;
    bool use_scoped_search = false;
    std::vector<std::string> imports;
    std::vector<LocalDecl> local_decls;
};

struct PremiseState {
    llama_context * embed_ctx = nullptr;
    std::unique_ptr<PremiseIndex> index;
    int embed_dim = 0;
    std::mutex embed_mutex;
};

static PremiseState * g_premise = nullptr;

// ---------------------------------------------------------------------------
// JSON / HTTP helpers
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

static json hits_to_json(const std::vector<PremiseIndex::Hit> & hits) {
    json arr = json::array();
    for (const auto & hit : hits) {
        arr.push_back({{"name", hit.name}, {"score", hit.score}});
    }
    return arr;
}

// ---------------------------------------------------------------------------
// Setup / cleanup
// ---------------------------------------------------------------------------

bool premise_setup(server_context & ctx_server, const std::string & database_path) {
    auto * server_ctx = ctx_server.get_llama_context();
    if (!server_ctx) {
        return false;
    }

    llama_model * model = const_cast<llama_model *>(llama_get_model(server_ctx));

    // Batch window can exceed n_ctx_train so many short decls pack into one
    // decode. Each sequence is still truncated to n_ctx_train (RoPE limit).
    const uint32_t n_ctx_train = (uint32_t) (std::max)(1, llama_model_n_ctx_train(model));
    const uint32_t n_batch_window = (std::max)(n_ctx_train, 4096u);
    llama_context_params embed_params = llama_context_default_params();
    embed_params.n_ctx = n_batch_window;
    embed_params.n_batch = n_batch_window;
    embed_params.n_ubatch = n_batch_window;
    embed_params.n_seq_max = (std::min)(256u, (uint32_t) llama_max_parallel_sequences());
    embed_params.embeddings = true;
    embed_params.kv_unified = true;
    embed_params.pooling_type = llama_pooling_type(server_ctx);

    llama_context * embed_ctx = llama_init_from_model(model, embed_params);
    if (!embed_ctx) {
        SRV_ERR("%s", "failed to create embedding context\n");
        return false;
    }

    try {
        const int embed_dim = llama_model_n_embd_out(model);
        auto index = std::make_unique<PremiseIndex>();
        index->open(database_path, embed_dim);

        auto state = std::make_unique<PremiseState>();
        state->embed_ctx = embed_ctx;
        state->embed_dim = embed_dim;
        state->index = std::move(index);
        g_premise = state.release();

        SRV_INF("premise ready: rows=%d\n", (int) g_premise->index->size());
        return true;
    } catch (const std::exception & e) {
        SRV_ERR("premise init failed: %s\n", e.what());
        llama_free(embed_ctx);
        return false;
    }
}

void premise_cleanup() {
    if (!g_premise) {
        return;
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
// /cache sends many declaration strings. We tokenize them, pack as many as
// fit into one llama_batch (by token budget and n_seq_max), decode, read one
// embedding per sequence, then repeat for the rest.
// ---------------------------------------------------------------------------

static std::vector<llama_token> tokenize_for_embed(llama_context * ctx, const std::string & text) {
    auto tokens = common_tokenize(ctx, text, /*add_special*/ false, /*parse_special*/ true);
    // Cap each sequence at the model's trained context (not the batch window).
    const int max_tokens = (std::max)(1, llama_model_n_ctx_train(llama_get_model(ctx)));
    if ((int) tokens.size() > max_tokens) {
        tokens.resize((size_t) max_tokens);
    }
    if (tokens.empty()) {
        throw std::runtime_error("tokenization produced no tokens");
    }
    return tokens;
}

// Add one token sequence as seq_id. Returns batch index of its last token.
static int batch_add_sequence(
        llama_batch & batch,
        const std::vector<llama_token> & tokens,
        llama_seq_id seq_id,
        enum llama_pooling_type pooling) {
    for (int i = 0; i < (int) tokens.size(); ++i) {
        const bool need_output =
            pooling != LLAMA_POOLING_TYPE_NONE || i + 1 == (int) tokens.size();
        common_batch_add(batch, tokens[i], i, { seq_id }, need_output);
    }
    return batch.n_tokens - 1;
}

// Decode one packed batch; write normalized embeddings for sequences [begin, end).
static void decode_embedding_chunk(
        llama_context * ctx,
        llama_batch & batch,
        const std::vector<std::vector<llama_token>> & sequences,
        size_t begin,
        size_t end,
        enum llama_pooling_type pooling,
        int n_embd,
        std::vector<std::vector<float>> & out) {
    common_batch_clear(batch);
    std::vector<int> last_token_pos;
    last_token_pos.reserve(end - begin);

    for (size_t i = begin; i < end; ++i) {
        const llama_seq_id seq_id = (llama_seq_id) (i - begin);
        last_token_pos.push_back(batch_add_sequence(batch, sequences[i], seq_id, pooling));
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
        common_embd_normalize(raw, out[i].data(), n_embd, 2);
    }
}

// How many sequences starting at `begin` fit in one batch.
static size_t count_sequences_for_chunk(
        const std::vector<std::vector<llama_token>> & sequences,
        size_t begin,
        int n_batch,
        int n_seq_max) {
    size_t end = begin;
    int n_tokens = 0;
    while (end < sequences.size() &&
            (int) (end - begin) < n_seq_max &&
            n_tokens + (int) sequences[end].size() <= n_batch) {
        n_tokens += (int) sequences[end].size();
        ++end;
    }
    return end;
}

static std::vector<std::vector<float>> embed_token_batch(
        std::vector<std::vector<llama_token>> sequences) {
    if (!g_premise || !g_premise->embed_ctx) {
        throw std::runtime_error("premise not initialized");
    }
    if (sequences.empty()) {
        return {};
    }

    std::lock_guard<std::mutex> lock(g_premise->embed_mutex);
    llama_context * ctx = g_premise->embed_ctx;
    const int n_batch = (int) llama_n_batch(ctx);
    const int n_seq_max = (int) llama_n_seq_max(ctx);
    const enum llama_pooling_type pooling = llama_pooling_type(ctx);
    const int n_embd = g_premise->embed_dim > 0
        ? g_premise->embed_dim
        : llama_model_n_embd_out(llama_get_model(ctx));

    std::vector<std::vector<float>> embeddings(sequences.size(), std::vector<float>(n_embd));
    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    try {
        for (size_t begin = 0; begin < sequences.size(); ) {
            const size_t end = count_sequences_for_chunk(sequences, begin, n_batch, n_seq_max);
            decode_embedding_chunk(ctx, batch, sequences, begin, end, pooling, n_embd, embeddings);
            begin = end;
        }
    } catch (...) {
        llama_batch_free(batch);
        throw;
    }
    llama_batch_free(batch);
    return embeddings;
}

static std::vector<PremiseIndex::Candidate> embed_declarations(
        const std::vector<LocalDecl> & decls,
        const std::string & module,
        bool cache_by_body = false) {
    std::vector<std::vector<llama_token>> sequences;
    sequences.reserve(decls.size());
    std::vector<PremiseIndex::Candidate> out(decls.size());
    std::vector<size_t> missing;
    missing.reserve(decls.size());
    for (size_t i = 0; i < decls.size(); ++i) {
        out[i].name = decls[i].name;
        out[i].module = module;
        if (cache_by_body &&
                g_premise->index->find_local_embedding(decls[i].body, out[i].embedding)) {
            continue;
        }
        missing.push_back(i);
        sequences.push_back(tokenize_for_embed(g_premise->embed_ctx, decls[i].body));
    }
    auto vectors = embed_token_batch(std::move(sequences));
    for (size_t i = 0; i < missing.size(); ++i) {
        const size_t out_index = missing[i];
        out[out_index].embedding = std::move(vectors[i]);
        if (cache_by_body) {
            g_premise->index->cache_local_embedding(
                    decls[out_index].body, out[out_index].embedding);
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Select
// ---------------------------------------------------------------------------

static SelectParams select_params_from_json(const json & data, int top_k) {
    SelectParams params;
    params.top_k = top_k;
    params.imports = json_string_array(data, "imports");
    params.local_decls = json_local_decls(data);
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
        auto locals = embed_declarations(params.local_decls, "", true);
        return g_premise->index->search_scoped(
                query.data(), params.imports, locals, params.top_k);
    }
    return g_premise->index->search_global(query.data(), params.top_k);
}

// ---------------------------------------------------------------------------
// HTTP: POST /version, /cache, /select
// ---------------------------------------------------------------------------

// /version { "module": "..." } -> token | null
static json handle_version(const json & body) {
    const std::string module = json_string(body, "module");
    if (!g_premise || !g_premise->index || module.empty()) {
        return nullptr;
    }
    const std::string token = g_premise->index->get_module_version(module);
    return token.empty() ? json(nullptr) : json(token);
}

// milliseconds elapsed since `since`, as a double
static double ms_since(const std::chrono::steady_clock::time_point & since) {
    const std::chrono::duration<double, std::milli> elapsed =
        std::chrono::steady_clock::now() - since;
    return elapsed.count();
}

// /cache one module; declarations[] batch-embedded then replace_module
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
        LOG_INF("%s: module '%s' already at token '%s', nothing to do\n",
                __func__, module.c_str(), token.c_str());
        return json{
            {"ok", true},
            {"cached", true},
            {"n_declarations", 0},
            {"embed_ms", 0.0},
            {"replace_ms", 0.0},
        };
    }

    const auto decls = json_local_decls(body);

    const auto t_embed_start = std::chrono::steady_clock::now();
    auto candidates = embed_declarations(decls, module);
    const double embed_ms = ms_since(t_embed_start);

    const auto t_replace_start = std::chrono::steady_clock::now();
    g_premise->index->replace_module(module, token, json_string_array(body, "imports"), candidates);
    const double replace_ms = ms_since(t_replace_start);

    LOG_INF("%s: module '%s': %zu decls, embed %.2f ms, replace_module %.2f ms\n",
            __func__, module.c_str(), decls.size(), embed_ms, replace_ms);

    return json{
        {"ok", true},
        {"cached", false},
        {"n_declarations", decls.size()},
        {"embed_ms", embed_ms},
        {"replace_ms", replace_ms},
    };
}

// /select { goal, k, imports?, declarations? } -> [{name,score}, ...]
static json handle_select(const json & body) {
    if (!g_premise) {
        throw std::runtime_error("premise not initialized");
    }
    const std::string goal = json_string(body, "goal");
    if (goal.empty()) {
        throw std::runtime_error("missing goal");
    }
    SelectParams params = select_params_from_json(body, json_int(body, "k", 5));
    auto query = embed_token_batch({ tokenize_for_embed(g_premise->embed_ctx, goal) }).front();
    return hits_to_json(select_hits(params, query));
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

// ---------------------------------------------------------------------------
// Entry point (llama-server --premise)
// ---------------------------------------------------------------------------

static std::function<void(int)> g_shutdown_handler;
static std::atomic_flag g_terminating = ATOMIC_FLAG_INIT;

static void print_premise_usage(int, char **) {
    printf("\n\n----- premise mode (llama-server --premise) -----\n\n");
    printf("  --premise\n");
    printf("      run as premise retrieval server instead of generation server\n\n");
    printf("  --index-db FILE\n");
    printf("      path to the SQLite premise database\n\n");
}

static void premise_signal_handler(int signal) {
    if (g_terminating.test_and_set()) {
        fprintf(stderr, "Received second interrupt, terminating immediately.\n");
        exit(1);
    }
    if (g_shutdown_handler) {
        g_shutdown_handler(signal);
    }
}

static server_http_res_ptr health_ok(const server_http_req &) {
    auto res = std::make_unique<server_http_res>();
    res->data = "{\"status\":\"ok\"}";
    return res;
}

int premise_server(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    std::string database_path;
    std::vector<char *> args;
    args.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--premise") {
            // Mode flag consumed by main dispatcher / this entry point.
        } else if (a == "--index-db" && i + 1 < argc) {
            database_path = argv[++i];
        } else {
            args.push_back(argv[i]);
        }
    }
    argc = (int) args.size();
    argv = args.data();

    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SERVER, print_premise_usage)) {
        return 1;
    }
    if (database_path.empty()) {
        SRV_ERR("%s", "--index-db is required in premise mode\n");
        return 1;
    }

    // Generation slots unused; one slot is enough for model load.
    if (params.n_parallel < 0) {
        params.n_parallel = 1;
        params.kv_unified = true;
    }

    llama_backend_init();
    llama_numa_init(params.numa);
    common_params_print_info(params, true);

    server_context ctx_server;
    server_http_context ctx_http;
    if (!ctx_http.init(params)) {
        SRV_ERR("%s", "failed to initialize HTTP server\n");
        llama_backend_free();
        return 1;
    }

    ctx_http.get("/health", health_ok);
    ctx_http.get("/v1/health", health_ok);
    premise_register_http_routes(ctx_http);

    auto clean_up = [&]() {
        SRV_INF("%s: cleaning up before exit...\n", __func__);
        ctx_http.stop();
        ctx_server.terminate();
        premise_cleanup();
        llama_backend_free();
    };

    if (!ctx_http.start()) {
        clean_up();
        SRV_ERR("%s", "exiting due to HTTP server error\n");
        return 1;
    }

    SRV_INF("%s", "loading model\n");
    if (!ctx_server.load_model(params)) {
        clean_up();
        if (ctx_http.thread.joinable()) {
            ctx_http.thread.join();
        }
        SRV_ERR("%s", "exiting due to model loading error\n");
        return 1;
    }

    if (!premise_setup(ctx_server, database_path)) {
        clean_up();
        if (ctx_http.thread.joinable()) {
            ctx_http.thread.join();
        }
        SRV_ERR("%s", "exiting due to premise initialization error\n");
        return 1;
    }
    ctx_http.is_ready.store(true);
    SRV_INF("llama-server --premise listening on %s\n", ctx_http.listening_address.c_str());

    g_shutdown_handler = [&](int) {
        ctx_server.terminate();
        ctx_http.stop();
    };

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
    struct sigaction sigint_action;
    sigint_action.sa_handler = premise_signal_handler;
    sigemptyset(&sigint_action.sa_mask);
    sigint_action.sa_flags = 0;
    sigaction(SIGINT, &sigint_action, NULL);
    sigaction(SIGTERM, &sigint_action, NULL);
#elif defined (_WIN32)
    auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
        return (ctrl_type == CTRL_C_EVENT) ? (premise_signal_handler(SIGINT), true) : false;
    };
    SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif

    if (ctx_http.thread.joinable()) {
        ctx_http.thread.join();
    }
    clean_up();
    return 0;
}
