#include "premise-init.hpp"
#include "premise-retrieval.hpp"
#include "server-context.h"

#include "common.h"
#include "llama.h"
#include "log.h"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <vector>

void premise_setup(server_context & ctx_server,
                   const std::string & premise_vec_path,
                   const std::string & premise_str_path,
                   PremiseMode premise_mode,
                   int embedding_workers) {
    const bool has_index_paths = !premise_vec_path.empty() && !premise_str_path.empty();
    if (!has_index_paths && premise_mode == PremiseMode::Auto) {
        return;
    }

    if (premise_vec_path.empty() != premise_str_path.empty()) {
        SRV_WRN("%s", "joint retrieval: --index-vecs and --index-names must be provided together; offline index disabled\n");
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

    if (premise_mode == PremiseMode::Joint && emb_token_id < 0) {
        SRV_ERR("%s", "joint retrieval: --joint requires a model with [EMB] token\n");
        return;
    }

    llama_context_params ep = llama_context_default_params();
    ep.n_ctx      = std::min(2048u, (uint32_t) std::max(1, llama_model_n_ctx_train(model)));
    ep.n_batch    = ep.n_ctx;
    ep.n_ubatch   = ep.n_ctx;
    ep.embeddings = true;
    ep.pooling_type = llama_pooling_type(gen_ctx);

    std::vector<llama_context *> emb_ctxs;
    const int n_emb_ctxs = std::max(1, embedding_workers);

    try {
        emb_ctxs.reserve(n_emb_ctxs);
        for (int i = 0; i < n_emb_ctxs; ++i) {
            llama_context * emb_ctx = llama_init_from_model(model, ep);
            if (!emb_ctx) {
                throw std::runtime_error("failed to create embedding context");
            }
            emb_ctxs.push_back(emb_ctx);
        }

        const int embedding_dim = llama_model_n_embd_out(model);
        std::unique_ptr<PremiseEmbedCache> embed_cache;
        std::unique_ptr<PremiseIndex> premise_index;

        if (premise_mode == PremiseMode::Embedding) {
            premise_index = std::make_unique<PremiseIndex>();
            premise_index->init_empty(embedding_dim);
        }

        if (has_index_paths && premise_mode == PremiseMode::Embedding) {
            // cache/select mode: the index paths name a persistent embedding
            // cache that this server creates and updates itself, so missing
            // files just mean a cold cache.
            embed_cache = std::make_unique<PremiseEmbedCache>();
            embed_cache->load(premise_vec_path, premise_str_path, embedding_dim);
        } else if (has_index_paths) {
            premise_index = std::make_unique<PremiseIndex>();
            premise_index->load(
                premise_vec_path.c_str(),
                premise_str_path.c_str(),
                0 /* load all */);
            const int n_embd_out = embedding_dim;
            if (premise_index->dim != n_embd_out) {
                SRV_ERR("joint retrieval: premise index dimension %d does not match model embedding dimension %d; offline index disabled\n",
                        premise_index->dim, n_embd_out);
                premise_index.reset();
            }
        }

        auto js = std::make_unique<PremiseRetrievalState>();
        js->emb_ctxs      = std::move(emb_ctxs);
        js->idle_emb_ctxs = js->emb_ctxs;
        js->emb_token_id = emb_token_id;
        js->joint_generation = premise_mode != PremiseMode::Embedding && emb_token_id >= 0;
        js->embedding_dim = embedding_dim;
        js->embed_cache = std::move(embed_cache);
        js->premise_index = std::move(premise_index);

        if (js->embed_cache && js->premise_index) {
            for (const auto & snap : js->embed_cache->snapshot_modules()) {
                std::vector<PremiseIndex::Record> declarations;
                declarations.reserve(snap.declarations.size());
                for (const auto & decl : snap.declarations) {
                    PremiseIndex::Record record;
                    record.name = decl.first;
                    record.module = snap.module;
                    record.embedding = decl.second;
                    declarations.push_back(std::move(record));
                }
                js->premise_index->replace_module(snap.module, snap.version_token, snap.imports, declarations);
            }
        }

        g_premise_state = js.release();
        const int n_premises = g_premise_state->premise_index ? g_premise_state->premise_index->n_premises : 0;
        if (g_premise_state->joint_generation) {
            SRV_INF("joint retrieval ready: %d offline premises, [EMB] token=%d, embedding workers=%d\n",
                    n_premises, emb_token_id, n_emb_ctxs);
        } else {
            SRV_INF("joint retrieval ready: cache/select embedding mode, %d offline premises, embedding workers=%d\n",
                    n_premises, n_emb_ctxs);
        }
    } catch (const std::exception & e) {
        SRV_ERR("joint retrieval: initialization failed: %s\n", e.what());
        for (llama_context * emb_ctx : emb_ctxs) {
            llama_free(emb_ctx);
        }
    }
}

void premise_cleanup() {
    if (g_premise_state) {
        if (g_premise_state->embed_cache) {
            g_premise_state->embed_cache->maybe_save(true);
        }
        for (llama_context * emb_ctx : g_premise_state->emb_ctxs) {
            llama_free(emb_ctx);
        }
        g_premise_state->emb_ctxs.clear();
        g_premise_state->idle_emb_ctxs.clear();
        delete g_premise_state;
        g_premise_state = nullptr;
    }
}
