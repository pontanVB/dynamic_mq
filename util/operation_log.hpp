#pragma once

#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <optional>
#include <ostream>
#include <vector>

namespace operation_log {

struct Push {
    long long tick;
    unsigned long key;
    std::size_t index;
    friend bool operator<(Push const& a, Push const& b) noexcept {
        return a.tick < b.tick;
    }
};

struct Pop {
    long long tick;
    std::size_t ref_index;
    friend bool operator<(Pop const& a, Pop const& b) noexcept {
        return a.tick < b.tick;
    }
};

struct OperationLog {
    std::vector<Push> pushes;
    std::vector<Pop> pops;
};

struct Metrics {
    std::size_t rank_error;
    std::size_t delay;
};

void write(OperationLog const& log, std::ostream& out);
std::vector<Metrics> replay(OperationLog const& logs);

}  // namespace operation_log
