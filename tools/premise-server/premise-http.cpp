#include "premise-http.hpp"

#include "premise-retrieval.hpp"
#include "server-http.h"

#include <nlohmann/json.hpp>

using json = nlohmann::ordered_json;

static server_http_res_ptr premise_json_response(const json & data) {
    auto res = std::make_unique<server_http_res>();
    res->data = data.dump();
    return res;
}

static server_http_res_ptr premise_error_response(const std::exception & e) {
    auto res = std::make_unique<server_http_res>();
    res->status = 400;
    res->data = json{{"error", {{"message", e.what()}}}}.dump();
    return res;
}

static server_http_res_ptr premise_handle_module_version(const server_http_req & req) {
    try {
        json body = req.body.empty() ? json::object() : json::parse(req.body);
        return premise_json_response(premise_get_module_version(body));
    } catch (const std::exception & e) {
        return premise_error_response(e);
    }
}

static server_http_res_ptr premise_handle_cache_module(const server_http_req & req) {
    try {
        return premise_json_response(premise_cache_module(json::parse(req.body)));
    } catch (const std::exception & e) {
        return premise_error_response(e);
    }
}

static server_http_res_ptr premise_handle_retrieve(const server_http_req & req) {
    try {
        return premise_json_response(premise_retrieve(json::parse(req.body)));
    } catch (const std::exception & e) {
        return premise_error_response(e);
    }
}

void premise_register_http_routes(const server_http_context & ctx_http) {
    ctx_http.post("/version", premise_handle_module_version);
    ctx_http.post("/cache",   premise_handle_cache_module);
    ctx_http.post("/select",  premise_handle_retrieve);
}
