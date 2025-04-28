#pragma once

#include "multiqueue/include/multiqueue/third_party/pcg/pcg_random.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <optional>
#include <random>
#include <iostream>
#include <filesystem>
#include <sstream>
#include <fstream>


namespace results{
    inline double max_contention = 0; //temp, prob remove
    thread_local int lock_fails = 0; //BAD NAME
    thread_local int dynamic_stickiness = 16;
    thread_local double fail_rate;
}

namespace multiqueue::mode {

template <int num_pop_candidates = 2>
class StickRandomDynamic {
    static_assert(num_pop_candidates > 0);

   public:
    struct Config {
        int seed{1};
        int stickiness{16};
        int punishment{-1};
        int reward{2};
        int lower_threshold{-5};
        int upper_threshold{5};
        std::filesystem::path stickiness_file = "stickiness_parameters.txt";
    
        Config() {
            std::ifstream file(stickiness_file);
            std::string line;
            int parameters[5];

            if (std::getline(file, line)) {
                std::stringstream ss(line);
                for (int i = 0; i < 5; ++i) {
                    std::string token;
                    if (std::getline(ss, token, ',')) {
                        parameters[i] = std::stoi(token);
                    }
                }
            }
            
            stickiness      = parameters[0];
            punishment      = parameters[1];
            reward          = parameters[2];
            lower_threshold = parameters[3];
            upper_threshold = parameters[4];
        }
    
    };

    struct SharedData {
        std::atomic_int id_count{0};

        explicit SharedData(std::size_t /*num_pqs*/) noexcept {
        }
    };

   private:
    pcg32 rng_{};
    std::array<std::size_t, static_cast<std::size_t>(num_pop_candidates)> pop_index_{};
    int count_{};
    double lock_fail_count_{};
    double lock_success_count_{};
    int lock_balance = 0;
    bool already_fetched = false;

    void refresh_pop_index(std::size_t num_pqs) noexcept {
        for (auto it = pop_index_.begin(); it != pop_index_.end(); ++it) {
            do {
                *it = std::uniform_int_distribution<std::size_t>{0, num_pqs - 1}(rng_);
            } while (std::find(pop_index_.begin(), it, *it) != it);
        }
    }

   protected:
    explicit StickRandomDynamic(Config const& config, SharedData& shared_data) noexcept {
        auto id = shared_data.id_count.fetch_add(1, std::memory_order_relaxed);
        auto seq = std::seed_seq{config.seed, id};
        rng_.seed(seq);
    }

    template <typename Context>
    std::optional<typename Context::value_type> try_pop(Context& ctx) {
        //std::clog << results::dynamic_stickiness;
        if (!already_fetched) {
            results::dynamic_stickiness = ctx.config().stickiness;
            already_fetched = true;
        }
        if (count_ == 0) {
            refresh_pop_index(ctx.num_pqs());
            count_ = results::dynamic_stickiness;
        }
        while (true) {
            std::size_t best = pop_index_[0];
            auto best_key = ctx.pq_guards()[best].top_key();
            for (std::size_t i = 1; i < static_cast<std::size_t>(num_pop_candidates); ++i) {
                auto key = ctx.pq_guards()[pop_index_[i]].top_key();
                if (ctx.compare(best_key, key)) {
                    best = pop_index_[i];
                    best_key = key;
                }
            }
            auto& guard = ctx.pq_guards()[best];
            if (guard.try_lock()) {
                if (guard.get_pq().empty()) {
                    guard.unlock();
                    count_ = 0;
                    return std::nullopt;
                }
                auto v = guard.get_pq().top();
                guard.get_pq().pop();
                guard.popped();
                guard.unlock();
                --count_;
                ++lock_success_count_;                
                lock_balance += ctx.config().reward;

                if (lock_balance >= ctx.config().upper_threshold) {
                    if (results::dynamic_stickiness > 1) {
                        results::dynamic_stickiness /= 2;
                    }
                    lock_balance = 0;
                }
                return v;
            }
            else { //lock fail
                ++results::lock_fails;
                ++lock_fail_count_;
                if(lock_fail_count_ > results::max_contention){
                    results::max_contention = lock_fail_count_;
                }
                lock_balance += ctx.config().punishment;
                if (lock_balance <= ctx.config().lower_threshold) {
                    results::dynamic_stickiness *= 2;
                    lock_balance = 0;
                }
            }
            refresh_pop_index(ctx.num_pqs());
            count_ = results::dynamic_stickiness;
            results::fail_rate = lock_success_count_ / (lock_fail_count_ + lock_success_count_);        
        }
    }

    template <typename Context>
    void push(Context& ctx, typename Context::value_type const& v) {
        if (!already_fetched) {
            results::dynamic_stickiness = ctx.config().stickiness;
            already_fetched = true;
        }
        if (count_ == 0) {
            refresh_pop_index(ctx.num_pqs());
            count_ = results::dynamic_stickiness;
        }
        std::size_t push_index = rng_() % num_pop_candidates;
        while (true) {
            auto& guard = ctx.pq_guards()[pop_index_[push_index]];
            if (guard.try_lock()) {
                guard.get_pq().push(v);
                guard.pushed();
                guard.unlock();
                --count_;
                ++lock_success_count_;
                lock_balance += ctx.config().reward;
                if (lock_balance >= ctx.config().upper_threshold) {
                    if (results::dynamic_stickiness > 1) {
                        results::dynamic_stickiness /= 2;
                    }
                    lock_balance = 0;
                }
                return;
            }
            else { //lock fail
                ++results::lock_fails;
                ++lock_fail_count_;
                if(lock_fail_count_ > results::max_contention){
                    results::max_contention = lock_fail_count_;
                }
                lock_balance += ctx.config().punishment;
                if (lock_balance <= ctx.config().lower_threshold) {
                    results::dynamic_stickiness *= 2;
                    lock_balance = 0;
                }
            }
            refresh_pop_index(ctx.num_pqs());
            count_ = results::dynamic_stickiness;
            results::fail_rate = lock_success_count_ / (lock_fail_count_ + lock_success_count_);        
        }
    }
};

}  // namespace multiqueue::mode
