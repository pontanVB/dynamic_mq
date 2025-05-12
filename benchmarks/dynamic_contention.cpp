#include "util/build_info.hpp"
#include "util/selector.hpp"
#include "util/thread_coordination.hpp"

//#define LOG_OPERATIONS

#ifdef LOG_OPERATIONS
#include "util/operation_log.hpp"
#endif

#include <cxxopts.hpp>

#ifdef WITH_PAPI
#include <papi.h>
#include <pthread.h>
#endif
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>
#include <fstream>
#include <sstream>
#include <filesystem>

#ifdef __SSE2__
#include <emmintrin.h>
#define PAUSE _mm_pause()
#else
#define PAUSE void(0)
#endif

//TEST INCLUDES
#include "util/stick_random_dynamic.hpp"


using key_type = unsigned long;
using value_type = unsigned long;

using pq_type = PQ<true, key_type, value_type>;
using handle_type = pq_type::handle_type;

// TODO - Change settings into file format?

struct Settings {
    int num_threads = 4;
    long long prefill_per_thread = 1 << 20;
    long long iterations_per_thread = 1 << 24;
    key_type min_prefill = 1;
    key_type max_prefill = 1 << 20;
    long min_update = 0;
    long max_update = 1 << 20;
    long long batch_size = 1 << 12;
    int seed = 1;
    int affinity = 6;
    int timeout_s = 0;
    int sleep_us = 0;
    std::deque<std::pair<int,std::chrono::milliseconds>> thread_intervals;
    std::filesystem::path interval_file = "thread_intervals.txt";
#ifdef LOG_OPERATIONS
    std::filesystem::path log_file = "operation_log.txt";
    std::filesystem::path log_file_metrics = "metrics_log.txt";
#endif
#ifdef WITH_PAPI
    std::vector<std::string> papi_events;
#endif
    pq_type::settings_type pq_settings;
};

void read_thread_intervals(Settings& settings) {
    std::clog << "thread_interval file: " << settings.interval_file << '\n';
    std::ifstream file(settings.interval_file);
    std::string line;

    while (std::getline(file, line)) {
        int active_threads;
        int duration;
        char comma;

        std::stringstream ss(line);

        if (ss >> active_threads >> comma >> duration && comma == ',') {
            settings.thread_intervals.emplace_back(active_threads, std::chrono::milliseconds(duration));
        } else {
            std::cerr << "Wrong thread_intervals line format: " << line << std::endl;
        }
    }
}

void register_cmd_options(Settings& settings, cxxopts::Options& cmd) {
    cmd.add_options()
        // clang-format off
            ("j,threads", "Number of threads", cxxopts::value<int>(settings.num_threads), "NUMBER")
            ("p,prefill", "Prefill per thread", cxxopts::value<long long>(settings.prefill_per_thread), "NUMBER")
            ("n,iterations", "Number of iterations per thread", cxxopts::value<long long>(settings.iterations_per_thread), "NUMBER")
            ("min-prefill", "Min prefill key", cxxopts::value<key_type>(settings.min_prefill), "NUMBER")
            ("max-prefill", "Max prefill key", cxxopts::value<key_type>(settings.max_prefill), "NUMBER")
            ("min-update", "Min update", cxxopts::value<long>(settings.min_update), "NUMBER")
            ("max-update", "Max update", cxxopts::value<long>(settings.max_update), "NUMBER")
            ("batch-size", "Batch size", cxxopts::value<long long>(settings.batch_size), "NUMBER")
            ("s,seed", "Initial seed", cxxopts::value<int>(settings.seed), "NUMBER")
            ("a,affinity", "CPU affinity ("
                "0: None, "
                "1: Thread Id, "
                "2: Same, "
                "3: Close caches, "
                "4: Far caches, "
                "5: Close L3 Far L1, "
                "6: Far L1 Close L3)"
                , cxxopts::value<int>(settings.affinity), "NUMBER")
            ("t,timeout", "Timeout in seconds", cxxopts::value<int>(settings.timeout_s), "NUMBER")
            ("q,sleep", "Time in microseconds to wait between operations", cxxopts::value<int>(settings.sleep_us), "NUMBER")
            ("i,thread-interval-file", "File to read thread intervals from", cxxopts::value<std::filesystem::path>(settings.interval_file), "PATH")
#ifdef LOG_OPERATIONS
            ("l,log-file", "File to write the operation log to", cxxopts::value<std::filesystem::path>(settings.log_file), "PATH")
            ("m,log-file-metrics", "File to write the metric log to", cxxopts::value<std::filesystem::path>(settings.log_file_metrics), "PATH")
#endif
#ifdef WITH_PAPI
            ("r,count-event", "Papi event to count", cxxopts::value<std::vector<std::string>>(settings.papi_events), "STRING")
#endif
        // clang-format on
        ;
    settings.pq_settings.register_cmd_options(cmd);
}

bool validate_settings(Settings const& settings) {
    if (!std::filesystem::exists(settings.interval_file)){
        std::cerr << "Error: The thread_intervals file does not exist\n";
        return false;
    }
    if (settings.num_threads <= 0) {
        std::cerr << "Error: Number of threads must be greater than 0\n";
        return false;
    }
    if (settings.prefill_per_thread < 0) {
        std::cerr << "Error: Prefill must be nonnegative\n";
        return false;
    }
    if (settings.iterations_per_thread < 0) {
        std::cerr << "Error: Iterations must be nonnegative\n";
        return false;
    }
    if (settings.min_prefill <= 0) {
        std::cerr << "Error: Prefill keys must be greater than 0\n";
        return false;
    }
    if (settings.max_prefill < settings.min_prefill) {
        std::cerr << "Error: Invalid prefill range\n";
        return false;
    }
    if (settings.min_update < 0) {
        std::cerr << "Error: Min update must be nonnegative\n";
        return false;
    }
    if (settings.max_update < settings.min_update) {
        std::cerr << "Error: Invalid update range\n";
        return false;
    }
    if (settings.batch_size <= 0) {
        std::cerr << "Error: batch size must be greater than 0\n";
        return false;
    }
    if (settings.affinity < 0 || settings.affinity > 6) {
        std::cerr << "Error: Invalid affinity\n";
        return false;
    }
    if (settings.timeout_s < 0) {
        std::cerr << "Error: Timeout must be nonnegative\n";
        return false;
    }
    if (settings.sleep_us < 0) {
        std::cerr << "Error: Sleep must be nonnegative\n";
        return false;
    }
    if (settings.seed <= 0) {
        std::cerr << "Error: Seed must be greater than 0\n";
        return false;
    }
    for (auto const& e: settings.thread_intervals) {
        if (e.first < 0) {
            std::cerr << "Error: Active threads must be at least 0\n";
            return false;
        }
        if (e.second <= std::chrono::milliseconds(0)) {
            std::cerr << "Error: Contention interval must be greater than 0\n";
            return false;
        }
    }
#ifdef LOG_OPERATIONS
    if (settings.log_file.empty()) {
        std::cerr << "Error: Log file name must not be empty\n";
        return false;
    }
    auto out = std::ofstream(settings.log_file);
    if (out.fail()) {
        std::cerr << "Error: Could not open file " << settings.log_file << " for writing\n";
        return false;
    }
    out.close();
#endif
#ifdef WITH_PAPI
    if (!settings.papi_events.empty()) {
        if (int ret = PAPI_library_init(PAPI_VER_CURRENT); ret != PAPI_VER_CURRENT) {
            std::cerr << "Error: Failed to initialize PAPI library\n";
            return false;
        }
        if (int ret = PAPI_thread_init(pthread_self); ret != PAPI_OK) {
            std::cerr << "Error: Failed to initialize PAPI thread support\n";
            return false;
        }
        for (auto const& name : settings.papi_events) {
            if (PAPI_query_named_event(name.c_str()) != PAPI_OK) {
                std::cerr << "Error: PAPI event '" << name << "' not available\n";
                return false;
            }
        }
    }
#endif
    return settings.pq_settings.validate();
}

void write_settings_human_readable(Settings const& settings, std::ostream& out) {
    auto affinity_name = [](int a) {
        switch (a) {
            case 0:
                return "None";
            case 1:
                return "Thread Id";
            case 2:
                return "Same";
            case 3:
                return "Close caches";
            case 4:
                return "Far caches";
            case 5:
                return "Close L3 Far L1";
            case 6:
                return "Far L1 Close L3";
            default:
                return "";
        }
    };
    out << "Threads: " << settings.num_threads << '\n';
    out << "Prefill per thread: " << settings.prefill_per_thread << '\n';
    out << "Iterations per thread: " << settings.iterations_per_thread << '\n';
    out << "Prefill range: [" << settings.min_prefill << ", " << settings.max_prefill << "]\n";
    out << "Update range: [" << settings.min_update << ", " << settings.max_update << "]\n";
    out << "Batch size: " << settings.batch_size << '\n';
    out << "Seed: " << settings.seed << '\n';
    out << "Affinity: " << affinity_name(settings.affinity) << '\n';
    out << "Timeout: ";
    if (settings.timeout_s == 0) {
        out << "None\n";
    } else {
        out << settings.timeout_s << " s\n";
    }
    out << "Sleep: ";
    if (settings.sleep_us == 0) {
        out << "None\n";
    } else {
        out << settings.sleep_us << " us\n";
    }
    out << "Thread intervals (nr, time): ";
    for (size_t i = 0; i < settings.thread_intervals.size(); ++i) {
        out << "(" << settings.thread_intervals[i].first << ", " 
            << settings.thread_intervals[i].second.count() << " ms)";
        if (i != settings.thread_intervals.size() - 1) {
            out << ", ";
        }
    }
    out << "\n";
#ifdef LOG_OPERATIONS
    out << "Log file: " << settings.log_file << '\n';
    out << "Log file Metrics: " << settings.log_file_metrics << '\n';
#endif
#ifdef WITH_PAPI
    out << "PAPI events: [";
    for (std::size_t i = 0; i < settings.papi_events.size(); ++i) {
        out << settings.papi_events[i];
        if (i != settings.papi_events.size() - 1) {
            out << ", ";
        }
    }
    out << ']' << '\n';
#endif
    settings.pq_settings.write_human_readable(out);
}

void write_settings_json(Settings const& settings, std::ostream& out) {
    out << '{';
    out << std::quoted("num_threads") << ':' << settings.num_threads << ',';
    out << std::quoted("prefill_per_thread") << ':' << settings.prefill_per_thread << ',';
    out << std::quoted("iterations_per_thread") << ':' << settings.iterations_per_thread << ',';
    out << std::quoted("prefill_min") << ':' << settings.min_prefill << ',';
    out << std::quoted("prefill_max") << ':' << settings.max_prefill << ',';
    out << std::quoted("update_min") << ':' << settings.min_update << ',';
    out << std::quoted("update_max") << ':' << settings.max_update << ',';
    out << std::quoted("batch_size") << ':' << settings.batch_size << ',';
    out << std::quoted("seed") << ':' << settings.seed << ',';
    out << std::quoted("affinity") << ':' << settings.affinity << ',';
    out << std::quoted("timeout_s") << ':' << settings.timeout_s << ',';
    out << std::quoted("sleep_us") << ':' << settings.sleep_us << ',';
    out << std::quoted("thread_intervals") << ':';
    out << '[';
    for (std::size_t i = 0; i < settings.thread_intervals.size(); ++i) {
        out << '{';
        out << std::quoted("active_threads") << ':' << settings.thread_intervals[i].first << ',';
        out << std::quoted("duration") << ':' << settings.thread_intervals[i].second.count();
        out << '}';
        if (i != settings.thread_intervals.size() - 1) {
            out << ',';
        }
    }
    out << ']' << ',';
#ifdef WITH_PAPI
    out << std::quoted("papi_events") << ':';
    out << '[';
    for (std::size_t i = 0; i < settings.papi_events.size(); ++i) {
        out << std::quoted(settings.papi_events[i]);
        if (i != settings.papi_events.size() - 1) {
            out << ',';
        }
    }
    out << ']' << ',';
#endif
    out << std::quoted("pq") << ':';
    settings.pq_settings.write_json(out);
    out << '}';
}

struct ThreadData {
    long long iter_count = 0;
    long long failed_pop_count = 0;

    std::deque<std::pair<int,std::chrono::milliseconds>> thread_intervals;


    //Interval data (There's probably a more convenient way to do this)
    std::vector<std::pair<int,int>> fail_data;
    std::vector<int> interval_fails = {};
    std::vector<long long> interval_iterations = {};
    long long interval_prev_iter = 0;
    long long interval_prev_fails = 0;

#ifdef WITH_PAPI
    std::vector<long long> papi_event_counter{};
#endif
#ifdef LOG_OPERATIONS
    struct PushLog {
        std::chrono::high_resolution_clock::time_point tick;
        std::pair<key_type, value_type> element;
    };
    struct PopLog {
        std::chrono::high_resolution_clock::time_point tick;
        value_type val;
    };
    struct DynamicLog {
        std::chrono::high_resolution_clock::time_point tick;
        double stickiness;
        int thread_id;
        long long total_iterations; // For throughput measurement, might tweak.
        int lock_fail_count;
        int active_threads;
        
    };
    std::vector<PushLog> pushes;
    std::vector<PopLog> pops;
    std::vector<DynamicLog> metrics;
#endif
};

void write_thread_data_json(ThreadData const& data, std::ostream& out) {
    out << '{';
    out << std::quoted("iterations") << ':' << data.iter_count << ',';
    out << std::quoted("failed_pops") << ':' << data.failed_pop_count << ',';
    out << std::quoted("fail_data") << ": [\n"; 
    for (size_t i = 0; i < data.interval_fails.size(); ++i) {
        out << "    {" 
            << std::quoted("interval") << ':' << i << ','
            << std::quoted("interval_iterations") << ':' << data.interval_iterations[i] << ','
            << std::quoted("interval_fails") << ':' << data.interval_fails[i]
            << '}';
        
        if (i != data.interval_fails.size() - 1) {
            out << ','; 
        }
        out << '\n'; 
    }
    out << "]\n"; 
#ifdef WITH_PAPI
    out << ',';
    out << std::quoted("papi_event_counter") << ':';
    out << '[';
    for (std::size_t i = 0; i < data.papi_event_counter.size(); ++i) {
        out << data.papi_event_counter[i];
        if (i != data.papi_event_counter.size() - 1) {
            out << ',';
        }
    }
    out << ']';
#endif
    out << '}';
}

#ifdef LOG_OPERATIONS
void write_log(std::vector<ThreadData> const& thread_data, std::ostream& out) {
    std::vector<ThreadData::PushLog> pushes;
    pushes.reserve(std::accumulate(thread_data.begin(), thread_data.end(), 0UL,
                                   [](std::size_t sum, auto const& e) { return sum + e.pushes.size(); }));
    std::vector<ThreadData::PopLog> pops;
    pushes.reserve(std::accumulate(thread_data.begin(), thread_data.end(), 0UL,
                                   [](std::size_t sum, auto const& e) { return sum + e.pops.size(); }));
    for (auto const& e : thread_data) {
        pushes.insert(pushes.end(), e.pushes.begin(), e.pushes.end());
        pops.insert(pops.end(), e.pops.begin(), e.pops.end());
    }
    std::sort(pushes.begin(), pushes.end(), [](auto const& lhs, auto const& rhs) { return lhs.tick < rhs.tick; });

    // Suboptimal sizing?
    value_type max_value = 0;
    for (const auto& push : pushes) {
        max_value = std::max(max_value, push.element.second);
    }


    std::vector<std::size_t> push_index(max_value);
    for (std::size_t i = 0; i < pushes.size(); ++i) {
        push_index[pushes[i].element.second] = i;
    }
    std::sort(pops.begin(), pops.end(), [](auto const& lhs, auto const& rhs) { return lhs.tick < rhs.tick; });
    out << pushes.size() << ' ' << pops.size() << '\n';
    std::size_t i = 0;
    for (auto const& pop : pops) {
        while ((i != pushes.size() && pushes[i].tick < pop.tick)) {
            out << '+' << pushes[i].element.first << '\n';
            ++i;
        }
        out << '-' << push_index[static_cast<std::size_t>(pop.val)] << '\n';
    }
    for (; i < pushes.size(); ++i) {
        out << i << ' ' << pushes[i].element.first << '\n';
    }
}

void write_log_metrics(std::vector<ThreadData> const& thread_data, std::ostream& out) {
    std::vector<ThreadData::DynamicLog> metrics;
    metrics.reserve(std::accumulate(thread_data.begin(), thread_data.end(), 0UL,
                                   [](std::size_t sum, auto const& e) { return sum + e.metrics.size(); }));
    for (auto const& e : thread_data) {
        metrics.insert(metrics.end(), e.metrics.begin(), e.metrics.end());
    }
    std::sort(metrics.begin(), metrics.end(), [](auto const& lhs, auto const& rhs) { return lhs.tick < rhs.tick; });
    out << "tick,stickiness,thread_id,total_iterations,lock_fails,active_threads\n";
    for (auto const& metric : metrics) {
        out << metric.tick.time_since_epoch().count() << ',' 
            << metric.stickiness << ',' 
            << metric.thread_id << ',' 
            << metric.total_iterations << ','
            << metric.lock_fail_count << ','
            << metric.active_threads
            << '\n';
    }
}
#endif

struct SharedData {
    std::vector<long long> updates;
    std::atomic_llong counter{0};
    std::atomic_int active_threads = 0;
    std::atomic_bool stop = false;
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
    std::vector<ThreadData> thread_data;

    //test
    std::atomic_long lock_fails_count = 0;
};

void write_result_json(Settings const& settings, SharedData const& data, std::ostream& out) {
    out << '{';
    out << std::quoted("settings") << ':';
    write_settings_json(settings, out);
    out << ',';
    out << std::quoted("results") << ':';
    out << '{';
    out << std::quoted("time_ns") << ':' << std::chrono::nanoseconds{data.end_time - data.start_time}.count() << ',';
    out << std::quoted("thread_data") << ':';
    out << '[';
    for (auto it = data.thread_data.begin(); it != data.thread_data.end(); ++it) {
        write_thread_data_json(*it, out);
        if (it != std::prev(data.thread_data.end())) {
            out << ',';
        }
    }
    out << ']';
    out << '}';
    out << '}' << '\n';
}

class Context : public thread_coordination::Context {
    handle_type handle_;
    ThreadData thread_data_;
    SharedData* shared_data_;
    Settings const* settings_;

   public:
    explicit Context(thread_coordination::Context ctx, handle_type handle, SharedData& shared_data,
                     Settings const& settings)
        : thread_coordination::Context{std::move(ctx)},
          handle_{std::move(handle)},
          shared_data_{&shared_data},
          settings_{&settings} {
    }
#ifdef LOG_OPERATIONS
    void push(std::pair<key_type, value_type> const& e) {
        handle_.push(e);
        auto tick = std::chrono::high_resolution_clock::now();
        thread_data_.pushes.push_back({tick, e});
        //thread_data_.metrics.push_back({tick, results::dynamic_stickiness, this->id(), 0});
    }

    auto try_pop() {
        auto tick = std::chrono::high_resolution_clock::now();
        auto retval = handle_.try_pop();
        if (retval) {
            thread_data_.pops.push_back({tick, retval->second});
        }


        #ifdef MQ_MODE_STICK_RANDOM_DYNAMIC
        thread_data_.metrics.push_back({
            tick,
            this->handle_.get_dynamic_stickiness(), 
            this->id(), 
            this->thread_data_.iter_count, 
            this->handle_.get_lock_fails(), 
            this->thread_data_.thread_intervals.front().first
        });
        this->handle_.reset_lock_fails();
        #endif
        return retval;
    }
#else
    void push(std::pair<key_type, value_type> const& e) {
        handle_.push(e);
    }

    auto try_pop() {
        return handle_.try_pop();
    }
#endif

    ThreadData& thread_data() noexcept {
        return thread_data_;
    }
    [[nodiscard]] ThreadData const& thread_data() const noexcept {
        return thread_data_;
    }
    SharedData& shared_data() noexcept {
        return *shared_data_;
    }
    [[nodiscard]] SharedData const& shared_data() const noexcept {
        return *shared_data_;
    }

    [[nodiscard]] Settings const& settings() const noexcept {
        return *settings_;
    }
};

// Save the contention for this thread and reset the counter for measuring in the next interval.
// void record_interval(Context& context){
//     long long iterations = context.thread_data().iter_count;
//     context.thread_data().interval_iterations.emplace_back(iterations - context.thread_data().interval_prev_iter);
//     context.thread_data().interval_prev_iter = iterations;
//     context.thread_data().interval_fails.emplace_back(results::lock_fails - context.thread_data().interval_prev_fails);
//     context.thread_data().interval_prev_fails = results::lock_fails;
// }

bool will_hit_timeout(std::chrono::high_resolution_clock::time_point timeout, 
                      std::chrono::high_resolution_clock::time_point interval_end) {
    return (timeout.time_since_epoch().count() != 0 && interval_end > timeout);
}

void print_interval_times(const Context& context, 
    const std::deque<std::chrono::high_resolution_clock::time_point>& interval_times) {
    auto current_thread_id = context.id(); // Use context.id() to get thread ID

    std::clog << "Thread ID " << current_thread_id << " - interval_times: [";
    bool first = true;
    for (const auto& tp : interval_times) {
        if (!first) std::clog << ", ";
        else first = false;

        // Convert time_point to milliseconds since epoch and print
        auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(tp.time_since_epoch()).count();
        std::clog << millis << "ms";
    }
    std::clog << "]\n";
}

void print_thread_intervals(const Context& context, 
    const std::deque<std::pair<int, std::chrono::milliseconds>>& thread_intervals) {
    auto current_thread_id = context.id(); // Use context.id() to get thread ID

    std::clog << "Thread ID " << current_thread_id << " - thread_intervals: [";
    bool first = true;
    for (const auto& [thread_id, duration] : thread_intervals) {
        if (!first) std::clog << ", ";
        else first = false;

        // Convert duration to milliseconds and print
        auto seconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
        std::clog << "(" << thread_id << ", " << seconds << "s)";
    }
    std::clog << "]\n";
}

// Process waiting during intervals and signal if work loop should be exited.
bool process_intervals(Context& context, 
                       std::deque<std::pair<int,std::chrono::milliseconds>>& thread_intervals,
                       std::deque<std::chrono::high_resolution_clock::time_point>& interval_times,
                       std::chrono::high_resolution_clock::time_point& timeout) {
    if (thread_intervals.empty()) {
        return true;
    }
    while (context.id() >= thread_intervals.front().first) {
        if (will_hit_timeout(timeout, interval_times.front())) {
            std::this_thread::sleep_until(timeout);
            return true;
        }

        std::this_thread::sleep_until(interval_times.front());
        interval_times.pop_front();
        thread_intervals.pop_front();
        if (thread_intervals.empty()) {
            return true;
        }
    }
    return false;
}

[[gnu::noinline]] void work_loop(Context& context) {
    auto offset = static_cast<value_type>(context.settings().num_threads * context.settings().prefill_per_thread);
    long long max = context.settings().iterations_per_thread * context.settings().num_threads;

    context.thread_data().thread_intervals = context.settings().thread_intervals;

    auto& thread_intervals = context.thread_data().thread_intervals;
    std::deque<std::chrono::high_resolution_clock::time_point> interval_times;
    std::chrono::high_resolution_clock::time_point interval_end = context.shared_data().start_time;
    // Make timeout a time point.
    std::chrono::high_resolution_clock::time_point timeout{};
    if (context.settings().timeout_s != 0) {
        timeout = context.shared_data().start_time + std::chrono::seconds(context.settings().timeout_s);
    }
    // Calculate time points of interval transitions.
    for (std::pair<int,std::chrono::milliseconds> interval : thread_intervals) {
        interval_end += interval.second;
        interval_times.emplace_back(interval_end);
    }

    // Sleep if this thread should be inactive from the beginning.
    // Exit if intervals are over or if timeout is reached.
    if (process_intervals(context, thread_intervals, interval_times, timeout)) {
        return;
    }
    // Loop through all work in batches.
    for (auto from = context.shared_data().counter.fetch_add(context.settings().batch_size, std::memory_order_relaxed);
         from < max;
         from = context.shared_data().counter.fetch_add(context.settings().batch_size, std::memory_order_relaxed)) {
        
        auto to = std::min(from + context.settings().batch_size, max);
        // Loop through batch of work.
        for (auto i = from; i < to; ++i) {
            if (i >= max) {
                std::cerr << "Error: No more iterations";
                return;
            }

            while (true) {
                if (auto e = context.try_pop(); e) {
                    if (context.settings().sleep_us != 0) {
                        auto sleep_until = std::chrono::high_resolution_clock::now() +
                            std::chrono::microseconds{context.settings().sleep_us};
                        while (std::chrono::high_resolution_clock::now() < sleep_until) {
                            PAUSE;
                        }
                    }
                    context.push({static_cast<key_type>(static_cast<long long>(e->first) +
                                                        context.shared_data().updates[static_cast<std::size_t>(i)]),
                                  offset + static_cast<value_type>(i)});
                    break;
                }
                ++context.thread_data().failed_pop_count;
            }
        }
        context.thread_data().iter_count += to - from;
        if (timeout.time_since_epoch().count() != 0 && std::chrono::high_resolution_clock::now() > timeout) {
            return;
        }
        // Finish if the interval is over.
        if (std::chrono::high_resolution_clock::now() >= interval_times.front()) {
            interval_times.pop_front();
            thread_intervals.pop_front();
        }
        // Sleep if this thread should be inactive.
        // Exit if intervals are over or if timeout is reached.
        if (process_intervals(context, thread_intervals, interval_times, timeout)) {
            return;
        }
    }
}

#ifdef WITH_PAPI
int prepare_papi(Settings const& settings) {
    if (int ret = PAPI_register_thread(); ret != PAPI_OK) {
        throw std::runtime_error{"Failed to register thread for PAPI"};
    }
    int event_set = PAPI_NULL;
    if (int ret = PAPI_create_eventset(&event_set); ret != PAPI_OK) {
        throw std::runtime_error{"Failed to create PAPI event set"};
    }
    for (auto const& name : settings.papi_events) {
        auto event = PAPI_NULL;
        if (PAPI_event_name_to_code(name.c_str(), &event) != PAPI_OK) {
            throw std::runtime_error{"Failed to resolve PAPI event '" + name + '\''};
        }
        if (PAPI_add_event(event_set, event) != PAPI_OK) {
            throw std::runtime_error{"Failed to add PAPI event '" + name + '\''};
        }
    }
    return event_set;
}
#endif

void benchmark_thread(Context context) {
#ifdef WITH_PAPI
    int event_set = PAPI_NULL;
    if (!context.settings().papi_events.empty()) {
        context.thread_data().papi_event_counter.resize(context.settings().papi_events.size());
        try {
            event_set = prepare_papi(context.settings());
        } catch (std::exception const& e) {
            context.write(std::cerr) << e.what() << '\n';
        }
    }
#endif
#ifdef LOG_OPERATIONS
    context.thread_data().pushes.reserve(
        static_cast<std::size_t>(context.settings().prefill_per_thread + 2 * context.settings().iterations_per_thread));
    context.thread_data().pops.reserve(static_cast<std::size_t>(2 * context.settings().iterations_per_thread));
    context.thread_data().metrics.reserve(static_cast<std::size_t>(context.settings().prefill_per_thread + 4 * context.settings().iterations_per_thread));
#endif

    std::vector<key_type> prefill(static_cast<std::size_t>(context.settings().prefill_per_thread));

    if (context.id() == 0) {
        std::clog << "Preparing...\n";
    }
    std::seed_seq seed{context.settings().seed, context.id()};
    std::default_random_engine rng(seed);
    context.synchronize();
    std::generate(prefill.begin(), prefill.end(),
                  [&rng, min = context.settings().min_prefill, max = context.settings().max_prefill]() {
                      return std::uniform_int_distribution<key_type>(min, max)(rng);
                  });
    std::generate_n(context.shared_data().updates.begin() + context.id() * context.settings().iterations_per_thread,
                    context.settings().iterations_per_thread,
                    [&rng, min = context.settings().min_update, max = context.settings().max_update]() {
                        return std::uniform_int_distribution<long>(min, max)(rng);
                    });
    context.synchronize();
    if (context.id() == 0) {
        std::clog << "Prefilling...\n";
    }
    context.synchronize();
    for (auto i = 0LL; i < context.settings().prefill_per_thread; ++i) {
        context.push({prefill[static_cast<std::size_t>(i)],
                      static_cast<value_type>(context.id() * context.settings().prefill_per_thread + i)});
    }
    context.synchronize();
    if (context.id() == 0) {
        std::clog << "Working...\n";
    }
#ifdef WITH_PAPI
    if (!context.settings().papi_events.empty()) {
        if (int ret = PAPI_start(event_set); ret != PAPI_OK) {
            context.write(std::cerr) << "Failed to start performance counters\n";
        }
    }
#endif
    if (context.id() == 0) {
        context.shared_data().start_time = std::chrono::high_resolution_clock::now();
    }
    context.synchronize();
    work_loop(context);
    context.synchronize();
    if (context.id() == 0) {
        context.shared_data().end_time = std::chrono::high_resolution_clock::now();
    }
#ifdef WITH_PAPI
    if (!context.settings().papi_events.empty()) {
        if (int ret = PAPI_stop(event_set, context.thread_data().papi_event_counter.data()); ret != PAPI_OK) {
            context.write(std::cerr) << "Failed to stop performance counters\n";
        }
    }
#endif
    context.shared_data().thread_data[static_cast<std::size_t>(context.id())] = std::move(context.thread_data());
}

void run_benchmark(Settings const& settings) {
    SharedData shared_data;
    shared_data.updates.resize(static_cast<std::size_t>(settings.iterations_per_thread * settings.num_threads));
    shared_data.thread_data.resize(static_cast<std::size_t>(settings.num_threads));

    auto pq =
        pq_type(settings.num_threads, static_cast<std::size_t>(settings.prefill_per_thread * settings.num_threads),
                settings.pq_settings);

    auto dispatch = [&](auto const& affinity) {
        auto dispatcher = thread_coordination::Dispatcher(affinity, settings.num_threads, [&](auto ctx) {
            benchmark_thread(Context(std::move(ctx), pq.get_handle(), shared_data, settings));
        });

        dispatcher.wait();
    };
    switch (settings.affinity) {
        case 0:
            dispatch(thread_coordination::affinity::None{});
            break;
        case 1:
            dispatch(thread_coordination::affinity::ThreadId{});
            break;
        case 2:
            dispatch(thread_coordination::affinity::Same{});
            break;
        case 3:
            dispatch(thread_coordination::affinity::CloseCaches{});
            break;
        case 4:
            dispatch(thread_coordination::affinity::FarCaches{});
            break;
        case 5:
            dispatch(thread_coordination::affinity::CloseL3FarL1{});
            break;
        case 6:
            dispatch(thread_coordination::affinity::FarL1CloseL3{});
            break;
    }

#ifdef LOG_OPERATIONS
    std::clog << "Writing logs...\n";
    std::ofstream log_out(settings.log_file);  // assumed to be valid
    write_log(shared_data.thread_data, log_out);
    log_out.close();
    
    #ifdef MQ_MODE_STICK_RANDOM_DYNAMIC
        std::clog << "Writing metric logs...\n";
        std::ofstream metric_log_out(settings.log_file_metrics);  // assumed to be valid
        write_log_metrics(shared_data.thread_data, metric_log_out);
        metric_log_out.close();
    #endif
#endif


    std::clog << "Done\n";
    std::clog << '\n';
    std::clog << "= Results =\n";
    std::clog << "Time (s): " << std::fixed << std::setprecision(3)
              << std::chrono::duration<double>(shared_data.end_time - shared_data.start_time).count() << '\n';
    write_result_json(settings, shared_data, std::cout);
    std::ofstream output;
    output.open("output/output.json");
    write_result_json(settings, shared_data, output);
    output.close();
}

int main(int argc, char* argv[]) {
    write_build_info(std::clog);
    std::clog << '\n';

    std::clog << "= Priority queue =\n";
    pq_type::write_human_readable(std::clog);
    std::clog << '\n';

    std::clog << "= Command line =\n";
    for (int i = 0; i < argc; ++i) {
        std::clog << argv[i];
        if (i != argc - 1) {
            std::clog << ' ';
        }
    }
    std::clog << '\n' << '\n';

    cxxopts::Options cmd(argv[0]);
    cmd.add_options()("h,help", "Print this help", cxxopts::value<bool>());
    Settings settings{};
    register_cmd_options(settings, cmd);

    try {
        auto args = cmd.parse(argc, argv);
        if (args.count("help") > 0) {
            std::clog << cmd.help() << '\n';
            return EXIT_SUCCESS;
        }
    } catch (std::exception const& e) {
        std::cerr << "Error parsing command line: " << e.what() << '\n';
        std::cerr << "Use --help for usage information" << '\n';
        return EXIT_FAILURE;
    }

    read_thread_intervals(settings);

    std::clog << "= Settings =\n";
    write_settings_human_readable(settings, std::clog);
    std::clog << '\n';

    if (!validate_settings(settings)) {
        return EXIT_FAILURE;
    }

    std::clog << "= Running benchmark =\n";
    run_benchmark(settings);
    return EXIT_SUCCESS;
}
