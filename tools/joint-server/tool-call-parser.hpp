#pragma once
#include <string>
#include <vector>
#include "json.hpp"

// ---------------------------------------------------------------------------
// Parse the tool-call arguments from a model generation.
//
// Our model double-serialises arguments:
//   <tool_call>
//   {"name":"have","arguments":"{\"name\":\"h\",\"type\":\"...\",\"clear\":[]}"}
//   </tool_call>
//
// common_chat_parse gets tc.name right but returns tc.arguments="{}" because
// it expects an object, not a string.  These helpers handle both cases.
// ---------------------------------------------------------------------------

struct ToolCallArgs {
    std::string              hyp_name;  // hypothesis name (args["name"])
    std::string              type;      // hypothesis type (args["type"])
    std::vector<std::string> clear;     // hypotheses to clear (args["clear"])
};

// Locate <tool_call>...</tool_call> in text, parse the outer JSON, and return
// the value of the "arguments" field as a plain JSON string.
// Handles both:
//   - string value  → returns the unescaped string as-is
//   - object value  → returns object serialised to a string
// Returns "" if not found or outer JSON is malformed.
inline std::string extract_arguments_string(const std::string & text) {
    const std::string TAG_OPEN  = "<tool_call>";
    const std::string TAG_CLOSE = "</tool_call>";
    auto p0 = text.find(TAG_OPEN);
    auto p1 = text.find(TAG_CLOSE);
    if (p0 == std::string::npos || p1 == std::string::npos || p1 <= p0)
        return "";

    const std::string block = text.substr(p0 + TAG_OPEN.size(),
                                          p1 - (p0 + TAG_OPEN.size()));
    try {
        auto outer = nlohmann::json::parse(block);
        if (!outer.contains("arguments")) return "";
        const auto & av = outer["arguments"];
        if (av.is_string()) return av.get<std::string>();
        if (av.is_object()) return av.dump();
    } catch (...) {}
    return "";
}

// Returns false when the string is empty or not valid JSON.
inline bool parse_tool_call_arguments(const std::string & args_json,
                                       ToolCallArgs & out) {
    if (args_json.empty()) return false;
    try {
        auto j       = nlohmann::json::parse(args_json);
        out.hyp_name = j.value("name",  "");
        out.type     = j.value("type",  "");
        out.clear    = j.value("clear", std::vector<std::string>{});
        return true;
    } catch (...) { return false; }
}
