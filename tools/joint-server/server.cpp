// Joint retrieval+generation server, wrapping llama.cpp server infrastructure.
//
// Modified copy of vendor/llama.cpp/tools/server/server.cpp.
// Additions vs vendor are marked with `// joint:` comments.
//
// The embedding computation and premises SSE injection happen in server-context.cpp
// at SLOT_STATE_DONE_PROMPT.

#include "server-context.h"
#include "server-http.h"
#include "server-models.h"
#include "server-cors-proxy.h"
#include "server-tools.h"

#include "arg.h"
#include "build-info.h"
#include "common.h"
#include "fit.h"
#include "llama.h"
#include "log.h"

#include "premise-init.hpp" // joint: setup/cleanup for retrieval state
#include "premise-http.hpp" // joint: premise cache/retrieve routes

#include <atomic>
#include <clocale>
#include <exception>
#include <signal.h>
#include <thread>

#if defined(_WIN32)
#include <windows.h>
#endif

static std::function<void(int)> shutdown_handler;
static std::atomic_flag is_terminating = ATOMIC_FLAG_INIT;

static inline void signal_handler(int signal) {
    if (is_terminating.test_and_set()) {
        fprintf(stderr, "Received second interrupt, terminating immediately.\n");
        exit(1);
    }
    shutdown_handler(signal);
}

static server_http_context::handler_t ex_wrapper(server_http_context::handler_t func) {
    return [func = std::move(func)](const server_http_req & req) -> server_http_res_ptr {
        std::string message;
        error_type error;
        try {
            return func(req);
        } catch (const std::invalid_argument & e) {
            error = ERROR_TYPE_INVALID_REQUEST;
            message = e.what();
        } catch (const std::exception & e) {
            error = ERROR_TYPE_SERVER;
            message = e.what();
        } catch (...) {
            error = ERROR_TYPE_SERVER;
            message = "unknown error";
        }

        auto res = std::make_unique<server_http_res>();
        res->status = 500;
        try {
            json error_data = format_error_response(message, error);
            res->status = json_value(error_data, "code", 500);
            res->data = safe_json_to_str({{ "error", error_data }});
            SRV_WRN("got exception: %s\n", res->data.c_str());
        } catch (const std::exception & e) {
            SRV_ERR("got another exception: %s | while handling exception: %s\n", e.what(), message.c_str());
            res->data = "Internal Server Error";
        }
        return res;
    };
}

// satisfies -Wmissing-declarations
int llama_server(int argc, char ** argv);

int llama_server(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    // Strip our custom flags before common_params_parse sees them.
    // common_params_parse fails on unknown flags, so we pull out ours first.
    std::string joint_vec_path;
    std::string joint_str_path;
    PremiseMode premise_mode = PremiseMode::Auto;
    std::vector<char *> filtered_args;
    {
        filtered_args.push_back(argv[0]);
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "--index-vecs" && i + 1 < argc) {
                joint_vec_path = argv[++i];
            } else if (a == "--index-strings" && i + 1 < argc) {
                joint_str_path = argv[++i];
            } else if (a == "--joint") {
                premise_mode = PremiseMode::Joint;
            } else if (a == "--no-joint") {
                premise_mode = PremiseMode::Embedding;
            } else {
                filtered_args.push_back(argv[i]);
            }
        }
        argc = (int)filtered_args.size();
        argv = filtered_args.data();
    }

    common_params params;
    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SERVER)) {
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    const bool is_router_server = params.model.path.empty();
    common_params_print_info(params, !is_router_server);

    if (!is_router_server) {
        if (params.embedding && params.n_batch > params.n_ubatch) {
            SRV_WRN("embeddings enabled with n_batch (%d) > n_ubatch (%d)\n", params.n_batch, params.n_ubatch);
            SRV_WRN("setting n_batch = n_ubatch = %d to avoid assertion failure\n", params.n_ubatch);
            params.n_batch = params.n_ubatch;
        }

        if (params.n_parallel < 0) {
            SRV_INF("%s", "n_parallel is set to auto, using n_parallel = 4 and kv_unified = true\n");
            params.n_parallel = 4;
            params.kv_unified = true;
        }
    }

    if (params.model_alias.empty() && !params.model.name.empty()) {
        params.model_alias.insert(params.model.name);
    }

    server_context ctx_server;
    server_http_context ctx_http;
    if (!ctx_http.init(params)) {
        SRV_ERR("%s", "failed to initialize HTTP server\n");
        return 1;
    }

    // ── Routes ────────────────────────────────────────────────────────────────
    server_routes routes(params, ctx_server);
    server_tools tools;

    std::optional<server_models_routes> models_routes{};
    if (is_router_server) {
        try {
            models_routes.emplace(params, argc, argv);
        } catch (const std::exception & e) {
            SRV_ERR("failed to initialize router models: %s\n", e.what());
            return 1;
        }

        routes.get_metrics                 = models_routes->proxy_get;
        routes.post_props                  = models_routes->proxy_post;
        routes.post_completions            = models_routes->proxy_post;
        routes.post_completions_oai        = models_routes->proxy_post;
        routes.post_chat_completions       = models_routes->proxy_post;
        routes.post_control                = models_routes->proxy_post;
        routes.post_responses_oai          = models_routes->proxy_post;
        routes.post_transcriptions_oai     = models_routes->proxy_post;
        routes.post_anthropic_messages     = models_routes->proxy_post;
        routes.post_anthropic_count_tokens = models_routes->proxy_post;
        routes.post_infill                 = models_routes->proxy_post;
        routes.post_embeddings             = models_routes->proxy_post;
        routes.post_embeddings_oai         = models_routes->proxy_post;
        routes.post_rerank                 = models_routes->proxy_post;
        routes.post_tokenize               = models_routes->proxy_post;
        routes.post_detokenize             = models_routes->proxy_post;
        routes.post_apply_template         = models_routes->proxy_post;
        routes.post_chat_completions_tok   = models_routes->proxy_post;
        routes.post_responses_tok_oai      = models_routes->proxy_post;
        routes.get_lora_adapters           = models_routes->proxy_get;
        routes.post_lora_adapters          = models_routes->proxy_post;
        routes.get_slots                   = models_routes->proxy_get;
        routes.post_slots                  = models_routes->proxy_post;

        routes.get_props   = models_routes->get_router_props;
        routes.get_models  = models_routes->get_router_models;

        ctx_http.post("/models/load",   ex_wrapper(models_routes->post_router_models_load));
        ctx_http.post("/models/unload", ex_wrapper(models_routes->post_router_models_unload));
    }

    ctx_http.get ("/health",                   ex_wrapper(routes.get_health));
    ctx_http.get ("/v1/health",                ex_wrapper(routes.get_health));
    ctx_http.get ("/metrics",                  ex_wrapper(routes.get_metrics));
    ctx_http.get ("/props",                    ex_wrapper(routes.get_props));
    ctx_http.post("/props",                    ex_wrapper(routes.post_props));
    ctx_http.get ("/models",                   ex_wrapper(routes.get_models));
    ctx_http.get ("/v1/models",                ex_wrapper(routes.get_models));
    ctx_http.post("/completion",               ex_wrapper(routes.post_completions));
    ctx_http.post("/completions",              ex_wrapper(routes.post_completions));
    ctx_http.post("/v1/completions",           ex_wrapper(routes.post_completions_oai));
    ctx_http.post("/chat/completions",         ex_wrapper(routes.post_chat_completions));
    ctx_http.post("/v1/chat/completions",      ex_wrapper(routes.post_chat_completions));
    ctx_http.post("/v1/chat/completions/control", ex_wrapper(routes.post_control));
    ctx_http.post("/v1/responses",             ex_wrapper(routes.post_responses_oai));
    ctx_http.post("/responses",                ex_wrapper(routes.post_responses_oai));
    ctx_http.post("/v1/audio/transcriptions",  ex_wrapper(routes.post_transcriptions_oai));
    ctx_http.post("/audio/transcriptions",     ex_wrapper(routes.post_transcriptions_oai));
    ctx_http.post("/v1/messages",              ex_wrapper(routes.post_anthropic_messages));
    ctx_http.post("/infill",                   ex_wrapper(routes.post_infill));
    ctx_http.post("/embedding",                ex_wrapper(routes.post_embeddings));
    ctx_http.post("/embeddings",               ex_wrapper(routes.post_embeddings));
    ctx_http.post("/v1/embeddings",            ex_wrapper(routes.post_embeddings_oai));
    ctx_http.post("/rerank",                   ex_wrapper(routes.post_rerank));
    ctx_http.post("/reranking",                ex_wrapper(routes.post_rerank));
    ctx_http.post("/v1/rerank",                ex_wrapper(routes.post_rerank));
    ctx_http.post("/v1/reranking",             ex_wrapper(routes.post_rerank));
    ctx_http.post("/tokenize",                 ex_wrapper(routes.post_tokenize));
    ctx_http.post("/detokenize",               ex_wrapper(routes.post_detokenize));
    ctx_http.post("/apply-template",           ex_wrapper(routes.post_apply_template));
    ctx_http.post("/chat/completions/input_tokens",    ex_wrapper(routes.post_chat_completions_tok));
    ctx_http.post("/v1/chat/completions/input_tokens", ex_wrapper(routes.post_chat_completions_tok));
    ctx_http.post("/responses/input_tokens",           ex_wrapper(routes.post_responses_tok_oai));
    ctx_http.post("/v1/responses/input_tokens",        ex_wrapper(routes.post_responses_tok_oai));
    ctx_http.post("/v1/messages/count_tokens",         ex_wrapper(routes.post_anthropic_count_tokens));
    ctx_http.get ("/lora-adapters",            ex_wrapper(routes.get_lora_adapters));
    ctx_http.post("/lora-adapters",            ex_wrapper(routes.post_lora_adapters));
    ctx_http.get ("/slots",                    ex_wrapper(routes.get_slots));
    ctx_http.post("/slots/:id_slot",           ex_wrapper(routes.post_slots));

    premise_register_http_routes(ctx_http);

    ctx_http.register_gcp_compat();

    if (params.ui_mcp_proxy || params.webui_mcp_proxy) {
        SRV_WRN("%s", "-----------------\n");
        SRV_WRN("%s", "CORS proxy is enabled, do not expose server to untrusted environments\n");
        SRV_WRN("%s", "This feature is EXPERIMENTAL and may be removed or changed in future versions\n");
        SRV_WRN("%s", "-----------------\n");
        ctx_http.get ("/cors-proxy", ex_wrapper(proxy_handler_get));
        ctx_http.post("/cors-proxy", ex_wrapper(proxy_handler_post));
    }
    if (!params.server_tools.empty()) {
        try {
            tools.setup(params.server_tools);
        } catch (const std::exception & e) {
            SRV_ERR("tools setup failed: %s\n", e.what());
            return 1;
        }
        SRV_WRN("%s", "-----------------\n");
        SRV_WRN("%s", "Built-in tools are enabled, do not expose server to untrusted environments\n");
        SRV_WRN("%s", "This feature is EXPERIMENTAL and may be changed in the future\n");
        SRV_WRN("%s", "-----------------\n");
        ctx_http.get ("/tools",  ex_wrapper(tools.handle_get));
        ctx_http.post("/tools",  ex_wrapper(tools.handle_post));
    }

    //
    // Start the server
    //

    std::function<void()> clean_up;

    if (is_router_server) {
        SRV_INF("%s", "starting router server, no model will be loaded in this process\n");

        clean_up = [&models_routes]() {
            SRV_INF("%s: cleaning up before exit...\n", __func__);
            if (models_routes.has_value()) {
                models_routes->models.unload_all();
            }
            llama_backend_free();
        };

        if (!ctx_http.start()) {
            clean_up();
            SRV_ERR("%s", "exiting due to HTTP server error\n");
            return 1;
        }
        ctx_http.is_ready.store(true);

        shutdown_handler = [&](int) {
            ctx_http.stop();
        };

    } else {
        clean_up = [&ctx_http, &ctx_server]() {
            SRV_INF("%s: cleaning up before exit...\n", __func__);
            ctx_http.stop();
            ctx_server.terminate();
            premise_cleanup(); // joint: free retrieval state
            llama_backend_free();
        };

        if (!ctx_http.start()) {
            clean_up();
            SRV_ERR("%s", "exiting due to HTTP server error\n");
            return 1;
        }

        SRV_INF("%s", "loading model\n");

        if (server_models::is_child_server()) {
            ctx_server.on_sleeping_changed([&](bool sleeping) {
                server_models::notify_router_sleeping_state(sleeping);
            });
        }

        if (!ctx_server.load_model(params)) {
            clean_up();
            if (ctx_http.thread.joinable()) {
                ctx_http.thread.join();
            }
            SRV_ERR("%s", "exiting due to model loading error\n");
            return 1;
        }

        // joint: initialize premise index and embedding context
        premise_setup(ctx_server, joint_vec_path, joint_str_path, premise_mode);

        routes.update_meta(ctx_server);
        ctx_http.is_ready.store(true);
        SRV_INF("%s", "model loaded\n");

        shutdown_handler = [&](int) {
            ctx_server.terminate();
        };
    }

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
    struct sigaction sigint_action;
    sigint_action.sa_handler = signal_handler;
    sigemptyset (&sigint_action.sa_mask);
    sigint_action.sa_flags = 0;
    sigaction(SIGINT, &sigint_action, NULL);
    sigaction(SIGTERM, &sigint_action, NULL);
#elif defined (_WIN32)
    auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
        return (ctrl_type == CTRL_C_EVENT) ? (signal_handler(SIGINT), true) : false;
    };
    SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif

    if (is_router_server) {
        SRV_INF("router server is listening on %s\n", ctx_http.listening_address.c_str());
        SRV_WRN("%s", "NOTE: router mode is experimental\n");
        SRV_WRN("%s", "      it is not recommended to use this mode in untrusted environments\n");
        if (ctx_http.thread.joinable()) {
            ctx_http.thread.join();
        }
        clean_up();
    } else {
        SRV_INF("server is listening on %s\n", ctx_http.listening_address.c_str());

        std::thread monitor_thread;
        if (server_models::is_child_server()) {
            json model_info = routes.get_model_info();
            monitor_thread = server_models::setup_child_server(shutdown_handler, model_info);
        }

        ctx_server.start_loop();

        clean_up();
        if (ctx_http.thread.joinable()) {
            ctx_http.thread.join();
        }
        if (monitor_thread.joinable()) {
            monitor_thread.join();
        }

        auto * ll_ctx = ctx_server.get_llama_context();
        if (ll_ctx != nullptr) {
            common_memory_breakdown_print(ll_ctx);
        }
    }

    return 0;
}

int main(int argc, char ** argv) {
    return llama_server(argc, argv);
}
