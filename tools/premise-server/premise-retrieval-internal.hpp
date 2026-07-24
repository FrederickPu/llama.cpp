#pragma once

#include "premise-retrieval.hpp"

#include <string>
#include <vector>

std::vector<float> premise_embed_tokens(std::vector<llama_token> tokens, bool append_emb);
std::vector<float> premise_embed_text(const std::string & text, bool append_emb);
std::vector<PremiseIndex::Record> premise_embed_declarations(
        const std::vector<LeanDeclaration> & declarations,
        const std::string & module);

void premise_store_pending_premises(int task_id, std::string sse);
