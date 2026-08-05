#include <cstring>

int llama_server(int argc, char ** argv);

#if defined(LLAMA_PREMISE_SERVER)
#include "premise.hpp"

static bool wants_premise_mode(int argc, char ** argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--premise") == 0) {
            return true;
        }
    }
    return false;
}
#endif

int main(int argc, char ** argv) {
#if defined(LLAMA_PREMISE_SERVER)
    if (wants_premise_mode(argc, argv)) {
        return premise_server(argc, argv);
    }
#endif
    return llama_server(argc, argv);
}
