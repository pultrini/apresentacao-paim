#pragma once

#include <chrono>

namespace ctbrain {

/// Cronômetro monotônico simples.
class Stopwatch {
public:
    Stopwatch() : start_{Clock::now()} {}

    /// Segundos desde a criação ou do último `lap()`; reinicia a contagem.
    double lap() {
        const auto now = Clock::now();
        const std::chrono::duration<double> elapsed = now - start_;
        start_ = now;
        return elapsed.count();
    }

private:
    using Clock = std::chrono::steady_clock;
    Clock::time_point start_;
};

}  // namespace ctbrain
