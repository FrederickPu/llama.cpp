#include "premise-retrieval-internal.hpp"

#include "common.h"

#include <algorithm>
#include <stdexcept>

PremiseRetrievalState * g_premise_state = nullptr;

static llama_context * premise_first_embedding_context() {
    if (!g_premise_state || g_premise_state->emb_ctxs.empty()) {
        throw std::runtime_error("joint retrieval is not initialized");
    }
    return g_premise_state->emb_ctxs.front();
}

static llama_context * premise_acquire_embedding_context() {
    if (!g_premise_state) {
        throw std::runtime_error("joint retrieval is not initialized");
    }
    std::unique_lock<std::mutex> lk(g_premise_state->emb_mu);
    g_premise_state->emb_cv.wait(lk, [] {
        return !g_premise_state || !g_premise_state->idle_emb_ctxs.empty();
    });
    if (!g_premise_state || g_premise_state->idle_emb_ctxs.empty()) {
        throw std::runtime_error("joint retrieval is not initialized");
    }
    llama_context * ctx = g_premise_state->idle_emb_ctxs.back();
    g_premise_state->idle_emb_ctxs.pop_back();
    return ctx;
}

static void premise_release_embedding_context(llama_context * ctx) {
    if (!ctx || !g_premise_state) {
        return;
    }
    {
        std::lock_guard<std::mutex> lk(g_premise_state->emb_mu);
        g_premise_state->idle_emb_ctxs.push_back(ctx);
    }
    g_premise_state->emb_cv.notify_one();
}

void premise_store_pending_premises(int task_id, std::string sse) {
    {
        std::lock_guard<std::mutex> lk(g_premise_state->pending_mu);
        g_premise_state->task_requests.erase(task_id);
        g_premise_state->pending_premises[task_id] = std::move(sse);
    }
    g_premise_state->pending_cv.notify_all();
}

static std::vector<llama_token> tokenize_text(const std::string & text) {
    llama_context * emb_ctx = premise_first_embedding_context();
    const llama_model * model = llama_get_model(emb_ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    int n_max = (int) text.size() + 64;
    std::vector<llama_token> tokens(std::max(8, n_max));
    int n = llama_tokenize(vocab, text.c_str(), (int) text.size(), tokens.data(), (int) tokens.size(), false, true);
    if (n < 0) {
        tokens.resize((size_t) -n);
        n = llama_tokenize(vocab, text.c_str(), (int) text.size(), tokens.data(), -n, false, true);
    }
    if (n <= 0) {
        throw std::runtime_error("tokenization produced no tokens");
    }
    tokens.resize((size_t) n);
    const int max_tokens = std::max(1, (int) llama_n_ctx(emb_ctx) - 1);
    if ((int) tokens.size() > max_tokens) {
        tokens.resize((size_t) max_tokens);
    }
    return tokens;
}

std::vector<float> premise_embed_tokens(std::vector<llama_token> tokens, bool append_emb) {
    if (!g_premise_state || g_premise_state->emb_ctxs.empty()) {
        throw std::runtime_error("joint retrieval is not initialized");
    }
    if (append_emb) {
        if (g_premise_state->emb_token_id < 0) {
            throw std::runtime_error("joint retrieval requires a model with [EMB] token");
        }
        tokens.push_back(g_premise_state->emb_token_id);
    }
    if (tokens.empty()) {
        throw std::runtime_error("cannot embed empty token sequence");
    }
    llama_context * emb_ctx = premise_acquire_embedding_context();

    const int max_tokens = (int) llama_n_ctx(emb_ctx);
    if ((int) tokens.size() > max_tokens) {
        tokens.erase(tokens.begin(), tokens.begin() + ((int) tokens.size() - max_tokens));
    }

    llama_memory_clear(llama_get_memory(emb_ctx), true);

    const enum llama_pooling_type pooling_type = llama_pooling_type(emb_ctx);
    llama_batch batch = llama_batch_init((int) tokens.size(), 0, 1);
    try {
        for (int i = 0; i < (int) tokens.size(); ++i) {
            const bool output = pooling_type != LLAMA_POOLING_TYPE_NONE || i == (int) tokens.size() - 1;
            common_batch_add(batch, tokens[i], i, { 0 }, output);
        }
        if (llama_decode(emb_ctx, batch) != 0) {
            throw std::runtime_error("embedding decode failed");
        }

        const int n_embd = g_premise_state->embedding_dim > 0 ? g_premise_state->embedding_dim : llama_model_n_embd_out(llama_get_model(emb_ctx));
        const float * raw = pooling_type == LLAMA_POOLING_TYPE_NONE
            ? llama_get_embeddings_ith(emb_ctx, batch.n_tokens - 1)
            : llama_get_embeddings_seq(emb_ctx, 0);
        if (!raw) {
            throw std::runtime_error("embedding output was not available");
        }

        std::vector<float> embd(n_embd);
        common_embd_normalize(raw, embd.data(), n_embd, 2);
        llama_batch_free(batch);
        premise_release_embedding_context(emb_ctx);
        return embd;
    } catch (...) {
        llama_batch_free(batch);
        premise_release_embedding_context(emb_ctx);
        throw;
    }
}

std::vector<float> premise_embed_text(const std::string & text, bool append_emb) {
    return premise_embed_tokens(tokenize_text(text), append_emb);
}

std::vector<PremiseIndex::Record> premise_embed_declarations(
        const std::vector<LeanDeclaration> & declarations,
        const std::string & module) {
    std::vector<PremiseIndex::Record> out(declarations.size());

    for (size_t i = 0; i < declarations.size(); ++i) {
        const auto & decl = declarations[i];
        out[i].name = decl.name;
        out[i].decl = decl.decl;
        out[i].module = module;
        out[i].embedding = premise_embed_text(out[i].decl, false);
    }
    return out;
}
