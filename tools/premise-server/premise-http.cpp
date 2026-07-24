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

static server_http_res_ptr premise_handle_version(const server_http_req & req) {
    try {
        json body = req.body.empty() ? json::object() : json::parse(req.body);
        return premise_json_response(premise_api_version(body));
    } catch (const std::exception & e) {
        return premise_error_response(e);
    }
}

static server_http_res_ptr premise_handle_cache(const server_http_req & req) {
    try {
        return premise_json_response(premise_api_cache(json::parse(req.body)));
    } catch (const std::exception & e) {
        return premise_error_response(e);
    }
}

static server_http_res_ptr premise_handle_select(const server_http_req & req) {
    try {
        return premise_json_response(premise_api_select(json::parse(req.body)));
    } catch (const std::exception & e) {
        return premise_error_response(e);
    }
}

static server_http_res_ptr premise_handle_version_batch(const server_http_req & req) {
    try {
        json body = req.body.empty() ? json::object() : json::parse(req.body);
        return premise_json_response(premise_api_version_batch(body));
    } catch (const std::exception & e) {
        return premise_error_response(e);
    }
}

static server_http_res_ptr premise_handle_cache_batch(const server_http_req & req) {
    try {
        return premise_json_response(premise_api_cache_batch(json::parse(req.body)));
    } catch (const std::exception & e) {
        return premise_error_response(e);
    }
}

void premise_register_http_routes(const server_http_context & ctx_http) {
    ctx_http.post("/version",       premise_handle_version);
    ctx_http.post("/version/batch", premise_handle_version_batch);
    ctx_http.post("/cache",         premise_handle_cache);
    ctx_http.post("/cache/batch",   premise_handle_cache_batch);
    ctx_http.post("/select",        premise_handle_select);
}
