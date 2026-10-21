#include "premise.hpp"
#include "premise-index.hpp"
#include "server-http.h"

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "log.h"

#include <nlohmann/json.hpp>

#include <atomic>
#include <clocale>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <list>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

using json = nlohmann::ordered_json;

// An embedding model, and the embeddings it has computed for `/select`'s local declarations,
// which clients resend with every request.
struct premise_model {
    common_init_result_ptr init; // owns the llama_model
    llama_context * ctx = nullptr;
    int dim = 0;

    ~premise_model() {
        llama_free(ctx);
    }

    static std::unique_ptr<premise_model> load(common_params params) {
        params.fit_params = false; // there is no generation context to fit
        auto model = std::make_unique<premise_model>();
        model->init = common_init_from_params(params, /* model_only */ true);
        if (!model->init || !model->init->model()) {
            throw std::runtime_error("failed to load model " + params.model.path);
        }
        llama_context_params cparams = common_context_params_to_llama(params);
        // pack many short declarations into each decode, whatever the model's trained context
        cparams.n_ctx = cparams.n_batch = cparams.n_ubatch = 4096;
        cparams.n_seq_max = std::min(256u, (uint32_t) llama_max_parallel_sequences());
        cparams.kv_unified = true;
        cparams.embeddings = true;
        model->ctx = llama_init_from_model(model->init->model(), cparams);
        if (!model->ctx) {
            throw std::runtime_error("failed to create the embedding context");
        }
        if (llama_pooling_type(model->ctx) == LLAMA_POOLING_TYPE_NONE) {
            throw std::runtime_error("premise mode needs a pooled embedding model (see --pooling)");
        }
        model->dim = llama_model_n_embd_out(model->init->model());
        return model;
    }

    // unit-length embeddings of `texts`, packing as many into each decode as fit
    std::vector<std::vector<float>> embed(const std::vector<std::string> & texts) {
        std::vector<std::vector<llama_token>> seqs;
        seqs.reserve(texts.size());
        for (const auto & text : texts) {
            seqs.push_back(tokenize(text));
        }
        const size_t n_batch = llama_n_batch(ctx);
        const size_t n_seq_max = llama_n_seq_max(ctx);
        std::vector<std::vector<float>> out(texts.size(), std::vector<float>(dim));
        llama_batch batch = llama_batch_init((int32_t) n_batch, 0, 1);
        try {
            for (size_t begin = 0, end; begin < seqs.size(); begin = end) {
                common_batch_clear(batch);
                size_t n_tokens = 0;
                for (end = begin; end < seqs.size() && end - begin < n_seq_max &&
                        n_tokens + seqs[end].size() <= n_batch; ++end) {
                    n_tokens += seqs[end].size();
                    for (size_t pos = 0; pos < seqs[end].size(); ++pos) {
                        common_batch_add(batch, seqs[end][pos], (llama_pos) pos, {(llama_seq_id) (end - begin)}, true);
                    }
                }
                llama_memory_clear(llama_get_memory(ctx), true);
                if (llama_decode(ctx, batch) != 0) {
                    throw std::runtime_error("embedding decode failed");
                }
                for (size_t i = begin; i < end; ++i) {
                    common_embd_normalize(llama_get_embeddings_seq(ctx, (llama_seq_id) (i - begin)), out[i].data(), dim, 2);
                }
            }
        } catch (...) {
            llama_batch_free(batch);
            throw;
        }
        llama_batch_free(batch);
        return out;
    }

    // `embed`, reusing the embeddings of recently seen texts
    std::vector<std::vector<float>> embed_cached(const std::vector<std::string> & texts) {
        std::vector<std::vector<float>> out(texts.size());
        std::vector<std::string> missing;
        std::vector<size_t> where;
        for (size_t i = 0; i < texts.size(); ++i) {
            auto it = cache_index.find(texts[i]);
            if (it == cache_index.end()) {
                missing.push_back(texts[i]);
                where.push_back(i);
            } else {
                cache.splice(cache.begin(), cache, it->second);
                out[i] = it->second->second;
            }
        }
        auto fresh = embed(missing);
        for (size_t j = 0; j < missing.size(); ++j) {
            out[where[j]] = fresh[j];
            if (cache_index.find(missing[j]) == cache_index.end()) {
                cache.emplace_front(missing[j], std::move(fresh[j]));
                cache_index.emplace(missing[j], cache.begin());
            }
        }
        while (cache.size() > CACHE_CAPACITY) {
            cache_index.erase(cache.back().first);
            cache.pop_back();
        }
        return out;
    }

private:
    static constexpr size_t CACHE_CAPACITY = 16384;
    std::list<std::pair<std::string, std::vector<float>>> cache; // most recently used first
    std::unordered_map<std::string, decltype(cache)::iterator> cache_index;

    std::vector<llama_token> tokenize(const std::string & text) const {
        auto tokens = common_tokenize(ctx, text, /* add_special */ true, /* parse_special */ true);
        // truncate to what one decode holds, keeping a final end-of-sequence token
        const llama_model * model = llama_get_model(ctx);
        const int max_tokens = std::max(1, std::min(llama_model_n_ctx_train(model), (int) llama_n_batch(ctx)));
        if ((int) tokens.size() > max_tokens) {
            const llama_vocab * vocab = llama_model_get_vocab(model);
            const llama_token last = tokens.back();
            tokens.resize(max_tokens);
            if (last == llama_vocab_eos(vocab) || last == llama_vocab_sep(vocab)) {
                tokens.back() = last;
            }
        }
        if (tokens.empty()) {
            throw std::runtime_error("tokenization produced no tokens");
        }
        return tokens;
    }
};

// What is being served; the HTTP handlers hold `mutex`, so requests run one at a time.
static struct {
    std::mutex mutex;
    common_params params; // as parsed; `PUT /config` substitutes the model path
    std::string model_path;
    std::string index_path;
    std::unique_ptr<premise_model> model;
    std::unique_ptr<premise_index> index;
} g;

static premise_index & index() {
    if (!g.index) {
        throw std::runtime_error("no premise database; set one with PUT /config");
    }
    return *g.index;
}

// a request's `declarations` [{name, decl}], embedded
static std::vector<premise_index::decl> embed_declarations(const json & body, bool cached) {
    std::vector<std::string> names;
    std::vector<std::string> texts;
    for (const auto & d : body.value("declarations", json::array())) {
        names.push_back(d.at("name").get<std::string>());
        texts.push_back(d.at("decl").get<std::string>());
    }
    auto embeddings = cached ? g.model->embed_cached(texts) : g.model->embed(texts);
    std::vector<premise_index::decl> decls;
    for (size_t i = 0; i < names.size(); ++i) {
        decls.push_back({names[i], std::move(embeddings[i])});
    }
    return decls;
}

// PUT /config {index, model} -> {ok}: serve this database with this model
static json handle_config(const json & body) {
    const auto index_path = body.at("index").get<std::string>();
    const auto model_path = body.at("model").get<std::string>();
    for (const std::string & path : {index_path, model_path}) {
        if (path.empty() || path.find('\0') != std::string::npos) {
            throw std::runtime_error("index and model must be paths");
        }
    }
    if (index_path == g.index_path && model_path == g.model_path) {
        return {{"ok", true}};
    }
    // load the new pair before releasing the old one, so that a failure leaves it serving
    std::unique_ptr<premise_model> model;
    if (model_path != g.model_path) {
        common_params params = g.params;
        params.model.path = model_path;
        model = premise_model::load(params);
    }
    auto index = std::make_unique<premise_index>(index_path, (model ? model : g.model)->dim);
    if (model) {
        g.model = std::move(model);
        g.model_path = model_path;
    }
    g.index = std::move(index);
    g.index_path = index_path;
    LOG_INF("premise: serving %s with %s\n", index_path.c_str(), model_path.c_str());
    return {{"ok", true}};
}

// POST /version {modules} -> [token | null]: each module's cached version
static json handle_version(const json & body) {
    json tokens = json::array();
    for (const auto & module : body.at("modules").get<std::vector<std::string>>()) {
        const std::string token = index().version(module);
        tokens.push_back(token.empty() ? json(nullptr) : json(token));
    }
    return tokens;
}

// POST /cache {module, token, declarations} -> {ok}: embed and store a module's declarations
static json handle_cache(const json & body) {
    const auto module = body.at("module").get<std::string>();
    const auto token = body.at("token").get<std::string>();
    if (module.empty() || token.empty()) {
        throw std::runtime_error("module and token must not be empty");
    }
    if (index().version(module) != token) {
        index().replace(module, token, embed_declarations(body, /* cached */ false));
    }
    return {{"ok", true}};
}

// POST /select {goal, k, modules, declarations} -> [{name, score}]: the goal's k best premises
// among the modules' cached declarations and the request's own
static json handle_select(const json & body) {
    const premise_index & idx = index();
    const auto query = g.model->embed({body.at("goal").get<std::string>()})[0];
    const auto locals = embed_declarations(body, /* cached */ true);
    json hits = json::array();
    for (const auto & hit : idx.search(query, body.value("modules", std::vector<std::string>()), locals,
                                       body.value("k", 5))) {
        hits.push_back({{"name", hit.name}, {"score", hit.score}});
    }
    return hits;
}

static server_http_context::handler_t json_handler(json (*handle)(const json &)) {
    return [handle](const server_http_req & req) {
        auto res = std::make_unique<server_http_res>();
        try {
            const json body = req.body.empty() ? json::object() : json::parse(req.body);
            std::lock_guard<std::mutex> lock(g.mutex);
            res->data = handle(body).dump();
        } catch (const std::exception & e) {
            res->status = 400;
            res->data = json{{"error", {{"message", e.what()}}}}.dump();
        }
        return res;
    };
}

static server_http_context * g_http = nullptr;

static void on_signal(int) {
    static std::atomic_flag stopping = ATOMIC_FLAG_INIT;
    if (stopping.test_and_set()) {
        fprintf(stderr, "Received second interrupt, terminating immediately.\n");
        exit(1);
    }
    g_http->stop();
}

static void print_usage(int, char **) {
    printf("\n----- premise mode -----\n\n");
    printf("  --premise         serve premise retrieval instead of generation\n");
    printf("  --index-db FILE   premise database to open at startup (or set one with PUT /config)\n\n");
}

int premise_server(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    std::string index_path;
    std::vector<char *> args = {argv[0]};
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--premise") == 0) {
            continue;
        }
        if (std::strcmp(argv[i], "--index-db") == 0) {
            if (i + 1 == argc) {
                fprintf(stderr, "error: --index-db needs a path\n");
                return 1;
            }
            index_path = argv[++i];
            continue;
        }
        args.push_back(argv[i]);
    }

    common_params params;
    common_init();
    if (!common_params_parse((int) args.size(), args.data(), params, LLAMA_EXAMPLE_SERVER, print_usage)) {
        return 1;
    }
    llama_backend_init();
    llama_numa_init(params.numa);
    common_params_print_info(params, true);

    server_http_context http;
    if (!http.init(params)) {
        LOG_ERR("premise: failed to initialize the HTTP server\n");
        llama_backend_free();
        return 1;
    }
    const auto health = [](const server_http_req &) {
        auto res = std::make_unique<server_http_res>();
        res->data = R"({"status":"ok"})";
        return res;
    };
    http.get("/health", health);
    http.get("/v1/health", health);
    http.put("/config", json_handler(handle_config));
    http.post("/version", json_handler(handle_version));
    http.post("/cache", json_handler(handle_cache));
    http.post("/select", json_handler(handle_select));
    if (!http.start()) {
        LOG_ERR("premise: failed to start the HTTP server\n");
        llama_backend_free();
        return 1;
    }

    // the HTTP server answers 503 until the model is loaded
    int status = 0;
    try {
        g.params = params;
        g.model = premise_model::load(params);
        g.model_path = params.model.path;
        if (!index_path.empty()) {
            g.index = std::make_unique<premise_index>(index_path, g.model->dim);
            g.index_path = index_path;
        }
        http.is_ready.store(true);
        LOG_INF("premise: listening on %s\n", http.listening_address.c_str());
    } catch (const std::exception & e) {
        LOG_ERR("premise: %s\n", e.what());
        http.stop();
        status = 1;
    }

    g_http = &http;
#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
    struct sigaction action = {};
    action.sa_handler = on_signal;
    sigemptyset(&action.sa_mask);
    sigaction(SIGINT, &action, nullptr);
    sigaction(SIGTERM, &action, nullptr);
#elif defined(_WIN32)
    SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(+[](DWORD type) -> BOOL {
        return type == CTRL_C_EVENT ? (on_signal(SIGINT), TRUE) : FALSE;
    }), TRUE);
#endif

    if (http.thread.joinable()) {
        http.thread.join();
    }
    g.index.reset();
    g.model.reset();
    llama_backend_free();
    return status;
}
