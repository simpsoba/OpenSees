#ifndef ParameterUtils_h
#define ParameterUtils_h

#include <unordered_map>
#include <string>
#include <functional>

namespace ParameterUtils {
    // Helper function to strip dashes and find parameter
    template<typename ConfigType>
    static auto findParameter(const std::string& key, 
                            const std::unordered_map<std::string, std::function<void(ConfigType&)>>& parsers) {
        // Strip leading dash if present
        if (!key.empty() && key[0] == '-') {
            return parsers.find(key.substr(1));
        }
        return parsers.find(key);
    }
}

// Returns MPI_COMM_WORLD size when ParameterUtils.cpp is built with OPS_CUDA_PARALLEL_MPI_SIZE
// (OPS_Cuda_Parallel); otherwise 1.
int getNumProcesses();

#endif // ParameterUtils_h
