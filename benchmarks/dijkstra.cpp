#include "util/build_info.hpp"
#include "util/graph.hpp"
#include "util/selector.hpp"
#include "util/termination_detection.hpp"
#include "util/thread_coordination.hpp"

#include <cxxopts.hpp>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <x86intrin.h>
#include <atomic>
#include <cassert>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <thread>
#include <type_traits>
#include <vector>

using pq_type = PQ<true, unsigned long, unsigned long>;
using handle_type = pq_type::handle_type;
using node_type = pq_type::value_type;

struct Settings {
    int num_threads = 4;
    std::filesystem::path graph_file;
    unsigned int seed = 1;
    pq_type::settings_type pq_settings{};

    #if defined(MQ_MODE_STICK_RANDOM_DYNAMIC) && defined(LOG_OPERATIONS)
    std::filesystem::path log_file_metrics = "metrics_log.txt";
    int log_frequency = 50;
    #endif
};

Settings settings{};

void register_cmd_options(cxxopts::Options& cmd) {
    // clang-format off
    cmd.add_options()
        ("j,threads", "The number of threads", cxxopts::value<int>(settings.num_threads), "NUMBER")
        ("graph", "The input graph", cxxopts::value<std::filesystem::path>(settings.graph_file), "PATH")
    // clang-format on
    // optional logging for dynamic relaxation
    #if defined(MQ_MODE_STICK_RANDOM_DYNAMIC) && defined(LOG_OPERATIONS)
        ("m,log-file-metrics", "File to write the metric log to", cxxopts::value<std::filesystem::path>(settings.log_file_metrics), "PATH")
        ("log-frequency", "Time in nanoseconds between logging of operations", cxxopts::value<int>(settings.log_frequency), "NUMBER")
    #endif
        ;
    settings.pq_settings.register_cmd_options(cmd);
    cmd.parse_positional({"graph"});
}

void write_settings_human_readable(std::ostream& out) {
    out << "Threads: " << settings.num_threads << '\n';
    out << "Graph: " << settings.graph_file << '\n';
    settings.pq_settings.write_human_readable(out);
}

void write_settings_json(std::ostream& out) {
    std::ostringstream tmp;
    pq_type::write_human_readable(tmp);

    std::string name = tmp.str();

    // Find position of first newline
    auto pos = name.find('\n');
    if (pos != std::string::npos) {
        name = name.substr(0, pos);
    }


         
    out << '{';
    out << std::quoted("num_threads") << ':' << settings.num_threads << ',';
    out << std::quoted("graph_file") << ':' << settings.graph_file << ',';
    out << std::quoted("seed") << ':' << settings.seed << ',';
    out << std::quoted("pq_name") << ':' << std::quoted(name) << ',';
    out << std::quoted("pq_settings") << ':';
    settings.pq_settings.write_json(out);
    out << '}';
}

struct Counter {
    long long pushed_nodes{0};
    long long ignored_nodes{0};
    long long processed_nodes{0};
#if defined(MQ_MODE_STICK_RANDOM_DYNAMIC) && defined(LOG_OPERATIONS)
    long long pushed_nodes_record{0};
    long long popped_nodes_record{0};
#endif
};

struct alignas(L1_CACHE_LINE_SIZE) AtomicDistance {
    std::atomic<long long> value{std::numeric_limits<long long>::max()};
};

#if defined(MQ_MODE_STICK_RANDOM_DYNAMIC) && defined(LOG_OPERATIONS)
struct ThreadInterval {
    int active_threads;
    std::chrono::nanoseconds delay;
    std::chrono::milliseconds duration;
};

struct ThreadData {
    std::deque<ThreadInterval> thread_intervals;


    struct DynamicLog {
        std::chrono::high_resolution_clock::time_point tick;
        long long pushes;
        long long pops;
        double stickiness;
        int thread_id;
        int lock_fail_count;
        
    };
    std::vector<DynamicLog> metrics;
};

void logging(handle_type& handle, int id, ThreadData& thread_data, Counter& counter) {
    auto tick = std::chrono::high_resolution_clock::now();

    thread_data.metrics.push_back({
        tick,
        counter.pushed_nodes_record,
        counter.popped_nodes_record,
        handle.get_dynamic_stickiness(), 
        id, 
        handle.get_lock_fails(), 
    });
    handle.reset_lock_fails();
    counter.pushed_nodes_record = 0;
    counter.popped_nodes_record = 0;
}

void write_log_metrics(std::vector<ThreadData> const& thread_data, std::ostream& out) {
    std::vector<ThreadData::DynamicLog> metrics;
    metrics.reserve(std::accumulate(thread_data.begin(), thread_data.end(), 0UL,
                                   [](std::size_t sum, auto const& e) { return sum + e.metrics.size(); }));
    for (auto const& e : thread_data) {
        metrics.insert(metrics.end(), e.metrics.begin(), e.metrics.end());
    }
    std::sort(metrics.begin(), metrics.end(), [](auto const& lhs, auto const& rhs) { return lhs.tick < rhs.tick; });
    out << "tick,pushes,pops,stickiness,thread_id,lock_fails\n";
    for (auto const& metric : metrics) {
        out << metric.tick.time_since_epoch().count() << ',' 
            << metric.pushes << ',' 
            << metric.pops << ',' 
            << metric.stickiness << ',' 
            << metric.thread_id << ',' 
            << metric.lock_fail_count
            << '\n';
    }
}

#endif

struct SharedData {
    Graph graph;
    std::vector<AtomicDistance> distances;
    termination_detection::TerminationDetection termination_detection;
    std::atomic_llong missing_nodes{0};
    #if defined(MQ_MODE_STICK_RANDOM_DYNAMIC) && defined(LOG_OPERATIONS)
    std::vector<ThreadData> thread_data;
    #endif

};

void process_node(node_type const& node, handle_type& handle, Counter& counter, SharedData& data) {
    auto current_distance = data.distances[node.second].value.load(std::memory_order_relaxed);
    if (static_cast<long long>(node.first) > current_distance) {
        ++counter.ignored_nodes;
        return;
    }
    for (auto i = data.graph.nodes[node.second]; i < data.graph.nodes[node.second + 1]; ++i) {
        auto target = data.graph.edges[i].target;
        auto d = static_cast<long long>(node.first) + data.graph.edges[i].weight;
        auto old_d = data.distances[target].value.load(std::memory_order_relaxed);
        while (d < old_d) {
            if (data.distances[target].value.compare_exchange_weak(old_d, d, std::memory_order_relaxed)) {
                if (handle.push({d, target})) {
                    ++counter.pushed_nodes;
                    // DOUBLECHECK THIS
                    #if defined(MQ_MODE_STICK_RANDOM_DYNAMIC) && defined(LOG_OPERATIONS)
                    ++counter.pushed_nodes_record;
                    #endif
                }
                break;
            }
        }
    }
    ++counter.processed_nodes;
}

// CHECK THIS PART
#if defined(MQ_MODE_STICK_RANDOM_DYNAMIC) && defined(LOG_OPERATIONS)
// [[gnu::noinline]] Counter benchmark_thread(thread_coordination::Context& thread_context, pq_type& pq,
//                                            SharedData& data) {
//     Counter counter;
//     auto handle = pq.get_handle();

//     auto& thread_data = data.thread_data[thread_context.id()];

//     if (thread_context.id() == 0) {
//         data.distances[0].value = 0;
//         handle.push({0, 0});
//         ++counter.pushed_nodes;
//         ++counter.pushed_nodes_record;
//     }
//     thread_context.synchronize();
//     std::optional<node_type> node;
//     while (data.termination_detection.repeat([&]() {
//         if(counter.pushed_nodes_record + counter.popped_nodes_record >= 200) {
//             node = try_pop_with_logging(handle, thread_context.id(), thread_data, counter);
//             ++counter.popped_nodes_record;
//         }
//         else{
//             node = handle.try_pop();
//             ++counter.popped_nodes_record;
//         }

//         return node.has_value();
//     })) {
//         process_node(*node, handle, counter, data);
//     }
//     thread_context.synchronize();
//     return counter;
// }

[[gnu::noinline]] Counter benchmark_thread(thread_coordination::Context& thread_context, pq_type& pq,
                                           SharedData& data) {
    Counter counter;
    auto handle = pq.get_handle();
    auto& thread_data = data.thread_data[thread_context.id()];
    auto last_log = std::chrono::high_resolution_clock::now();

    if (thread_context.id() == 0) {
        data.distances[0].value = 0;
        handle.push({0, 0});
        ++counter.pushed_nodes;
    }
    thread_context.synchronize();
    while (true) {
        std::optional<node_type> node;
        while (data.termination_detection.repeat([&]() {
            node = handle.try_pop();
            ++counter.popped_nodes_record;
            if (std::chrono::high_resolution_clock::now() - last_log >= std::chrono::nanoseconds(settings.log_frequency)) {
                last_log = std::chrono::high_resolution_clock::now();
                logging(handle, thread_context.id(), thread_data, counter);
            }
            return node.has_value();
        })) {
            process_node(*node, handle, counter, data);
        }
        data.missing_nodes.fetch_add(counter.pushed_nodes - counter.processed_nodes - counter.ignored_nodes,
                                     std::memory_order_relaxed);
        thread_context.synchronize();
        if (data.missing_nodes.load(std::memory_order_relaxed) == 0) {
            break;
        }
        thread_context.synchronize();
        if (thread_context.id() == 0) {
            data.missing_nodes.store(0, std::memory_order_relaxed);
            data.termination_detection.reset(settings.num_threads);     // DOESN'T MATCH COMMIT FROM MARVIN, BUT REQUIRES NUM_THREADS.
        }
        thread_context.synchronize();
    }
    return counter;
}

#else
[[gnu::noinline]] Counter benchmark_thread(thread_coordination::Context& thread_context, pq_type& pq,
                                           SharedData& data) {
    Counter counter;
    auto handle = pq.get_handle();
    if (thread_context.id() == 0) {
        data.distances[0].value = 0;
        handle.push({0, 0});
        ++counter.pushed_nodes;
    }
    thread_context.synchronize();
    while (true) {
        std::optional<node_type> node;
        while (data.termination_detection.repeat([&]() {
            node = handle.try_pop();
            return node.has_value();
        })) {
            process_node(*node, handle, counter, data);
        }
        data.missing_nodes.fetch_add(counter.pushed_nodes - counter.processed_nodes - counter.ignored_nodes,
                                     std::memory_order_relaxed);
        thread_context.synchronize();
        if (data.missing_nodes.load(std::memory_order_relaxed) == 0) {
            break;
        }
        thread_context.synchronize();
        if (thread_context.id() == 0) {
            data.missing_nodes.store(0, std::memory_order_relaxed);
            data.termination_detection.reset(settings.num_threads);     // DOESN'T MATCH COMMIT FROM MARVIN, BUT REQUIRES NUM_THREADS.
        }
        thread_context.synchronize();
    }
    return counter;
}
#endif




void run_benchmark() {
    std::clog << "Reading graph...\n";
    SharedData shared_data{{}, {}, termination_detection::TerminationDetection(settings.num_threads)};
    #if defined(MQ_MODE_STICK_RANDOM_DYNAMIC) && defined(LOG_OPERATIONS)
    shared_data.thread_data.resize(static_cast<std::size_t>(settings.num_threads));
    #endif
    try {
        shared_data.graph = Graph(settings.graph_file);
    } catch (std::runtime_error const& e) {
        std::clog << "Error: " << e.what() << '\n';
        std::exit(EXIT_FAILURE);
    }
    std::clog << "Graph has " << shared_data.graph.num_nodes() << " nodes and " << shared_data.graph.num_edges()
              << " edges\n";
    shared_data.distances = std::vector<AtomicDistance>(shared_data.graph.num_nodes());

    std::vector<Counter> thread_counter(static_cast<std::size_t>(settings.num_threads));
    auto pq = pq_type(settings.num_threads, shared_data.graph.num_nodes(), settings.pq_settings);
    std::clog << "Working...\n";
    auto start_time = std::chrono::steady_clock::now();
    thread_coordination::Dispatcher dispatcher{settings.num_threads, [&](auto ctx) {
                                                   auto t_id = static_cast<std::size_t>(ctx.id());
                                                   thread_counter[t_id] = benchmark_thread(ctx, pq, shared_data);
                                               }};
    dispatcher.wait();
    auto end_time = std::chrono::steady_clock::now();

#if defined(MQ_MODE_STICK_RANDOM_DYNAMIC) && defined(LOG_OPERATIONS)

    std::clog << "Writing metric logs...\n";
    std::ofstream metric_log_out(settings.log_file_metrics);  // assumed to be valid
    write_log_metrics(shared_data.thread_data, metric_log_out);
    metric_log_out.close();
#endif

    std::clog << "Done\n";
    auto total_counts =
        std::accumulate(thread_counter.begin(), thread_counter.end(), Counter{}, [](auto sum, auto const& counter) {
            sum.pushed_nodes += counter.pushed_nodes;
            sum.processed_nodes += counter.processed_nodes;
            sum.ignored_nodes += counter.ignored_nodes;
            return sum;
        });
    std::clog << '\n';
    auto furthest_node =
        std::max_element(shared_data.distances.begin(), shared_data.distances.end(), [](auto const& a, auto const& b) {
            auto a_val = a.value.load(std::memory_order_relaxed);
            auto b_val = b.value.load(std::memory_order_relaxed);
            if (b_val == std::numeric_limits<long long>::max()) {
                return false;
            }
            if (a_val == std::numeric_limits<long long>::max()) {
                return true;
            }
            return a_val < b_val;
        });
    std::clog << "= Results =\n";
    std::clog << "Time (s): " << std::fixed << std::setprecision(3)
              << std::chrono::duration<double>(end_time - start_time).count() << '\n';
    std::clog << "Furthest node: " << furthest_node - shared_data.distances.begin() << '\n';
    std::clog << "Longest distance: " << furthest_node->value.load(std::memory_order_relaxed) << '\n';
    std::clog << "Processed nodes: " << total_counts.processed_nodes << '\n';
    std::clog << "Ignored nodes: " << total_counts.ignored_nodes << '\n';
    if (total_counts.processed_nodes + total_counts.ignored_nodes != total_counts.pushed_nodes) {
        std::cerr << "Warning: Not all nodes were popped\n";
        std::cerr << "Probably the priority queue discards duplicate keys\n";
    }
    std::cout << '{';
    std::cout << std::quoted("settings") << ':';
    write_settings_json(std::cout);
    std::cout << ',';
    std::cout << std::quoted("graph") << ':';
    std::cout << '{';
    std::cout << std::quoted("num_nodes") << ':' << shared_data.graph.num_nodes() << ',';
    std::cout << std::quoted("num_edges") << ':' << shared_data.graph.num_edges();
    std::cout << '}' << ',';
    std::cout << std::quoted("results") << ':';
    std::cout << '{';
    std::cout << std::quoted("time_ns") << ':' << std::chrono::nanoseconds{end_time - start_time}.count() << ',';
    std::cout << std::quoted("furthest_node") << ':' << furthest_node - shared_data.distances.begin() << ',';
    std::cout << std::quoted("longest_distance") << ':' << furthest_node->value.load(std::memory_order_relaxed) << ',';
    std::cout << std::quoted("processed_nodes") << ':' << total_counts.processed_nodes << ',';
    std::cout << std::quoted("ignored_nodes") << ':' << total_counts.ignored_nodes;
    std::cout << '}';
    std::cout << '}' << '\n';
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
    cmd.add_options()("h,help", "Print this help");
    Settings settings{};
    register_cmd_options(cmd);

    try {
        auto args = cmd.parse(argc, argv);
        if (args.count("help") > 0) {
            std::cerr << cmd.help() << '\n';
            return EXIT_SUCCESS;
        }
    } catch (cxxopts::OptionParseException const& e) {
        std::cerr << "Error parsing command line: " << e.what() << '\n';
        std::cerr << "Use --help for usage information" << '\n';
        return EXIT_FAILURE;
    }

    std::clog << "= Settings =\n";
    write_settings_human_readable(std::clog);
    std::clog << '\n';

    std::clog << "= Running benchmark =\n";
    run_benchmark();
    return EXIT_SUCCESS;
}
