#pragma once
#include <string>

struct server_context;

enum class JointMode {
    Auto,
    Joint,
    Embedding,
};

// Initialize g_joint_state from --index-vecs / --index-strings paths or, when
// requested, as a cache/select-only embedding premise server.
// Must be called after ctx_server.load_model() succeeds.
void joint_setup(server_context & ctx_server,
                 const std::string & joint_vec_path,
                 const std::string & joint_str_path,
                 JointMode joint_mode);

// Free g_joint_state. Safe to call when g_joint_state is null.
void joint_cleanup();
