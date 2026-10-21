# llama-server --premise

Embedding-only retrieval of Lean premises for [Canonical](https://github.com/chasenorman/CanonicalLean).
Build with `-DLLAMA_SERVER_PREMISE=ON`, then:

```sh
llama-server --premise -m model.gguf --pooling mean --index-db premise.db
```

The server embeds the declarations of each Lean module once, keyed by the module's version
token, and answers which declarations are most relevant to a goal.

| Request | Body | Response |
|---|---|---|
| `PUT /config` | `{index, model}` | `{ok}`: serve this database with this model; on failure the served pair is unchanged |
| `POST /version` | `{modules}` | each module's cached version token, or `null` |
| `POST /cache` | `{module, token, declarations: [{name, decl}]}` | `{ok}`: embed and store a module's declarations |
| `POST /select` | `{goal, k, modules, declarations: [{name, decl}]}` | `[{name, score}]`: the goal's `k` best premises among the modules' cached declarations and the request's own `declarations` |

Scores are cosine similarities; a name appears at most once. Requests run one at a time.

The database (`premise-schema.hpp`) stores each module's version token and its declarations'
embeddings as f16. The server holds all of it in memory and searches exactly.

`test-premise-index` tests the index; `test_regression.py --model MODEL.gguf` tests the HTTP API.
