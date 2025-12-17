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


namespace multiqueue::mode {

template <int num_pop_candidates = 2>
class StickRandomDynamicCentral {
    static_assert(num_pop_candidates > 0);

   public:
    struct Config {
        int seed{1};
        int stickiness{1};
        int punishment{-2};
        int reward{1};
        int lower_threshold{-50};
        int upper_threshold{50};
        double stick_factor{1};
        double stick_cap{8096};
        bool change_sampling{false};
        int CHECK_GLOBAL{100};

    };


    struct SharedData {
        std::atomic_int id_count{0};
        std::atomic_uint32_t lock_ = 0;

        double global_stickiness{1};
        int global_lock_balance{0};

        explicit SharedData(std::size_t /*num_pqs*/) noexcept {
        }
    };

   private:
    pcg32 rng_{};
    std::array<std::size_t, static_cast<std::size_t>(num_pop_candidates)> pop_index_{};
    int count_{};
    int lock_fail_count_{};
    double lock_success_count_{};
    int lock_balance = 0;
    bool already_fetched = false;

    double dynamic_stickiness{};
    double stick_factor_{};
    double initial_stick_factor_{};
    bool change_sampling_{};

    int iterations = 0;


    template <typename Context>
    void lock(Context& ctx) noexcept {
        while(true) {
            if (ctx.shared_data().lock_.load(std::memory_order_relaxed) == 0U && ctx.shared_data().lock_.exchange(1U, std::memory_order_acquire) == 0U){
                return;
            }
        }
    }

    template <typename Context>
    void unlock(Context& ctx) {
        ctx.shared_data().lock_.store(0U, std::memory_order_release);
    }


    void refresh_pop_index(std::size_t num_pqs) noexcept {
        for (auto it = pop_index_.begin(); it != pop_index_.end(); ++it) {
            do {
                *it = std::uniform_int_distribution<std::size_t>{0, num_pqs - 1}(rng_);
            } while (std::find(pop_index_.begin(), it, *it) != it);
        }
    }

    // Update lock balances.
    template <typename Context>
    void update_lock_balance(Context& ctx, bool success) {
        iterations++;
        if (success) {
            lock_balance += ctx.config().reward;
        }
        else {
            lock_balance += ctx.config().punishment;
        }
        if (iterations >= ctx.config().CHECK_GLOBAL) {
            lock(ctx);
            ctx.shared_data().global_lock_balance = 0.99*ctx.shared_data().global_lock_balance + 0.01*lock_balance;
            if (ctx.shared_data().global_lock_balance >= ctx.config().upper_threshold) {
                ctx.shared_data().global_stickiness = std::max(std::floor(ctx.shared_data().global_stickiness / stick_factor_), 1.0);
                ctx.shared_data().global_lock_balance = 0; 
            }
            else if (ctx.shared_data().global_lock_balance <= ctx.config().lower_threshold) {
                ctx.shared_data().global_stickiness = std::min(std::ceil(ctx.shared_data().global_stickiness * stick_factor_), ctx.config().stick_cap);
                ctx.shared_data().global_lock_balance = 0;
            }
            dynamic_stickiness = ctx.shared_data().global_stickiness;
            unlock(ctx);
            lock_balance = 0;
            iterations = 0;
        }
    }


   protected:
    explicit StickRandomDynamicCentral(Config const& config, SharedData& shared_data) noexcept {
        initial_stick_factor_ = config.stick_factor;
        stick_factor_ = initial_stick_factor_;
        change_sampling_ = config.change_sampling;
        auto id = shared_data.id_count.fetch_add(1, std::memory_order_relaxed);
        auto seq = std::seed_seq{config.seed, id};
        rng_.seed(seq);
    }


    template <typename Context>
    std::optional<typename Context::value_type> try_pop(Context& ctx) {
        if(count_ < 0){
            std::cerr << "Error: Negative count_ in stick_random_dynamic";
            return std::nullopt;
        }
        if (!already_fetched) {
            dynamic_stickiness = static_cast<double>(ctx.config().stickiness);
            already_fetched = true;
        }
        if (count_ == 0) {
            refresh_pop_index(ctx.num_pqs());
            count_ = static_cast<int>(dynamic_stickiness);
        }

        while (true) {
            
            // Track number of iterations (every operation?)
            // Track local lock_balance
            // If number of iterations reaches CHECK_GLOBAL
            //      Add local lock_balance to global_lock_balance (could be sub through negative balance)
            //      If global_lock_balance reaches UPPER_THRESHHOLD or LOWER_THRESHHOLD
            //          Reset global_lock_balance
            //          Update global_stickiness
            //      Set local_stickiness to global_stickiness
            //      Reset number of iterations
            //      Reset local lock_balance


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
                update_lock_balance(ctx, true);
                return v;
            }
            else { //lock fail
                ++lock_fail_count_;
                update_lock_balance(ctx, false);
            }
            refresh_pop_index(ctx.num_pqs());
            count_ = static_cast<int>(dynamic_stickiness);;
        }
    }

    template <typename Context>
    void push(Context& ctx, typename Context::value_type const& v) {
        if(count_ < 0){
            std::cerr << "Error: Negative count_ in stick_random_dynamic";
            return;
        }

        if (!already_fetched) {
            dynamic_stickiness = static_cast<double>(ctx.config().stickiness);
            already_fetched = true;
        }
        if (count_ == 0) {
            refresh_pop_index(ctx.num_pqs());
            count_ = static_cast<int>(dynamic_stickiness);;
        }
        std::size_t push_index = rng_() % num_pop_candidates;
        while (true) {
            auto& guard = ctx.pq_guards()[pop_index_[push_index]];
            if (guard.try_lock()) {
                guard.get_pq().push(v);
                guard.pushed();
                guard.unlock();
                --count_;
                update_lock_balance(ctx, true);
                return;
            }
            else { //lock fail
                ++lock_fail_count_;
                update_lock_balance(ctx, false);
            }
            refresh_pop_index(ctx.num_pqs());
            count_ = static_cast<int>(dynamic_stickiness);
        }
    }

  public:
    int get_lock_fails(){
        return lock_fail_count_;
    }

    double get_dynamic_stickiness(){
        return dynamic_stickiness;
    }

    void reset_lock_fails(){
        lock_fail_count_ = 0;
    }

    void reset_stick_factor(){
        stick_factor_ = 1;
    }

    void set_stick_factor(){
        stick_factor_ = initial_stick_factor_;

    }


};

}  // namespace multiqueue::mode