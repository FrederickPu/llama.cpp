#include "joint-init.hpp"
#include "joint-retrieval.hpp"
#include "server-context.h"

#include "common.h"
#include "llama.h"
#include "log.h"

#include <algorithm>
#include <memory>

void joint_setup(server_context & ctx_server,
                 const std::string & joint_vec_path,
                 const std::string & joint_str_path,
                 JointMode joint_mode) {
    const bool has_index_paths = !joint_vec_path.empty() && !joint_str_path.empty();
    if (!has_index_paths && joint_mode == JointMode::Auto) {
        return;
    }

    if (joint_vec_path.empty() != joint_str_path.empty()) {
        SRV_WRN("%s", "joint retrieval: --index-vecs and --index-strings must be provided together; offline index disabled\n");
    }

    auto * gen_ctx = ctx_server.get_llama_context();
    if (!gen_ctx) {
        return;
    }

    llama_model * model = const_cast<llama_model *>(llama_get_model(gen_ctx));
    const llama_vocab * vocab = llama_model_get_vocab(model);

    llama_token emb_token_id = -1;
    char piece[64];
    for (int i = llama_vocab_n_tokens(vocab) - 1; i >= 0; --i) {
        int n = llama_token_to_piece(vocab, i, piece, sizeof(piece), 0, true);
        if (n > 0 && std::string(piece, n) == "[EMB]") {
            emb_token_id = i;
            break;
        }
    }

    if (joint_mode == JointMode::Joint && emb_token_id < 0) {
        SRV_ERR("%s", "joint retrieval: --joint requires a model with [EMB] token\n");
        return;
    }

    llama_context_params ep = llama_context_default_params();
    ep.n_ctx      = std::min(2048u, (uint32_t) std::max(1, llama_model_n_ctx_train(model)));
    ep.n_batch    = ep.n_ctx;
    ep.n_ubatch   = ep.n_ctx;
    ep.embeddings = true;
    ep.pooling_type = llama_pooling_type(gen_ctx);

    llama_context * emb_ctx = llama_init_from_model(model, ep);
    if (!emb_ctx) {
        SRV_ERR("%s", "joint retrieval: failed to create embedding context\n");
        return;
    }

    try {
        auto js = std::make_unique<JointRetrievalState>();
        js->emb_ctx      = emb_ctx;
        js->emb_token_id = emb_token_id;
        js->joint_generation = joint_mode != JointMode::Embedding && emb_token_id >= 0;
        js->embedding_dim = llama_model_n_embd_out(model);

        if (has_index_paths) {
            js->premise_index = std::make_unique<PremiseIndex>();
            js->premise_index->load(
                joint_vec_path.c_str(),
                joint_str_path.c_str(),
                0 /* load all */);
            const int n_embd_out = js->embedding_dim;
            if (js->premise_index->dim != n_embd_out) {
                SRV_ERR("joint retrieval: premise index dimension %d does not match model embedding dimension %d; offline index disabled\n",
                        js->premise_index->dim, n_embd_out);
                js->premise_index.reset();
            }
        }

        g_joint_state = js.release();
        const int n_premises = g_joint_state->premise_index ? g_joint_state->premise_index->n_premises : 0;
        if (g_joint_state->joint_generation) {
            SRV_INF("joint retrieval ready: %d offline premises, [EMB] token=%d\n",
                    n_premises, emb_token_id);
        } else {
            SRV_INF("joint retrieval ready: cache/select embedding mode, %d offline premises, no [EMB] token\n",
                    n_premises);
        }
    } catch (const std::exception & e) {
        SRV_ERR("joint retrieval: initialization failed: %s\n", e.what());
        llama_free(emb_ctx);
    }
}

void joint_cleanup() {
    if (g_joint_state) {
        if (g_joint_state->emb_ctx) {
            llama_free(g_joint_state->emb_ctx);
            g_joint_state->emb_ctx = nullptr;
        }
        delete g_joint_state;
        g_joint_state = nullptr;
    }
}
