#pragma once

#include <string>

struct server_context;
struct server_http_context;

// Premise mode entry point (llama-server --premise).
int premise_server(int argc, char ** argv);

// After model load: one embedding context + optional FAISS cache.
void premise_setup(server_context & ctx_server,
                   const std::string & vecs_path,
                   const std::string & names_path);
void premise_cleanup();

// HTTP: POST /version, /cache, /select
void premise_register_http_routes(const server_http_context & ctx_http);
