// Premise-only server: loads an embedding model and exposes Lean premise APIs.

#include "premise-http.hpp"
#include "premise-init.hpp"

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "log.h"
#include "server-context.h"
#include "server-http.h"

#include <atomic>
#include <clocale>
#include <csignal>
#include <exception>
#include <functional>
#include <thread>
#include <vector>

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

static server_http_res_ptr json_response(const std::string & data) {
    auto res = std::make_unique<server_http_res>();
    res->data = data;
    return res;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    std::string premise_vec_path;
    std::string premise_meta_path;
    std::vector<char *> filtered_args;
    filtered_args.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--index-vecs" && i + 1 < argc) {
            premise_vec_path = argv[++i];
        } else if ((a == "--index-names" || a == "--index-strings") && i + 1 < argc) {
            premise_meta_path = argv[++i];
        } else if (a == "--no-joint") {
            // Accepted for compatibility with older local commands.
        } else {
            filtered_args.push_back(argv[i]);
        }
    }
    argc = (int) filtered_args.size();
    argv = filtered_args.data();

    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SERVER)) {
        return 1;
    }

    // LLAMA_EXAMPLE_SERVER defaults n_parallel to -1 (auto); llama-server's
    // main resolves it before load_model, so we must too or context creation
    // fails. A single slot suffices: generation is never used here, and the
    // premise embedding work runs on a dedicated context (premise-init.cpp).
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

    ctx_http.get("/health", [](const server_http_req &) {
        return json_response("{\"status\":\"ok\"}");
    });
    ctx_http.get("/v1/health", [](const server_http_req &) {
        return json_response("{\"status\":\"ok\"}");
    });
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

    premise_setup(ctx_server, premise_vec_path, premise_meta_path, PremiseMode::Embedding, params.n_parallel);
    ctx_http.is_ready.store(true);
    SRV_INF("server is listening on %s\n", ctx_http.listening_address.c_str());

    shutdown_handler = [&](int) {
        ctx_server.terminate();
        ctx_http.stop();
    };

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

    if (ctx_http.thread.joinable()) {
        ctx_http.thread.join();
    }
    clean_up();
    return 0;
}
