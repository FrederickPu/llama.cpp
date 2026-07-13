#pragma once
#include <string>

struct server_context;

enum class PremiseMode {
    Auto,
    Joint,
    Embedding,
};

// Initialize g_premise_state from --index-vecs / --index-names paths or, when
// requested, as a cache/select-only embedding premise server.
// Must be called after ctx_server.load_model() succeeds.
void premise_setup(server_context & ctx_server,
                   const std::string & premise_vec_path,
                   const std::string & premise_str_path,
                   PremiseMode premise_mode,
                   int embedding_workers);

// Free g_premise_state. Safe to call when g_premise_state is null.
void premise_cleanup();
