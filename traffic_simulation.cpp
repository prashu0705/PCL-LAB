#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <random>
#include <cmath>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <queue>
#include <cstring>

// Hybrid MPI + OpenMP Traffic Simulator
// Modes supported implicitly by runner:
// - Serial: run with mpirun -np 1, OMP_NUM_THREADS=1
// - OpenMP-only: run with mpirun -np 1, OMP_NUM_THREADS>1
// - MPI-only: run with mpirun -np P, OMP_NUM_THREADS=1
// - Hybrid MPI+OpenMP: run with mpirun -np P, OMP_NUM_THREADS>1

struct Road {
    int from_node;
    int to_node;
    double length;           // in km
    double speed_limit;      // in km/h
    int max_capacity;        // max cars on this road
    int current_cars;        // current number of cars

    Road() : from_node(0), to_node(0), length(1.0), speed_limit(50.0),
             max_capacity(10), current_cars(0) {}

    double get_congestion() const {
        return (double)current_cars / std::max(1, max_capacity);
    }

    double get_travel_time() const {
        double base_time = length / speed_limit;
        double congestion_factor = 1.0 + 2.0 * get_congestion();
        return base_time * congestion_factor;
    }
};

struct Car {
    int id;
    int current_node;
    int destination_node;
    std::vector<int> route;
    int route_index;
    double position_on_road;  // 0.0 to 1.0
    double speed;
    bool completed;

    Car() : id(0), current_node(0), destination_node(0), route_index(0), position_on_road(0.0), speed(0.0), completed(false) {}
};

class CityGraph {
private:
    int grid_width;
    int grid_height;
    std::unordered_map<int, std::vector<Road>> adjacency_list; // stable storage per node
    std::mt19937 rng;

public:
    CityGraph(int width, int height, int seed = 42)
        : grid_width(width), grid_height(height), rng(seed) {
        build_grid_graph();
    }

    int get_node_id(int x, int y) const {
        return y * grid_width + x;
    }

    void build_grid_graph() {
        std::uniform_real_distribution<> speed_dist(40.0, 80.0);
        std::uniform_real_distribution<> length_dist(0.5, 2.0);
        std::uniform_int_distribution<> capacity_dist(8, 15);

        // Create grid connections (bidirectional stored separately)
        for (int y = 0; y < grid_height; ++y) {
            for (int x = 0; x < grid_width; ++x) {
                int node = get_node_id(x, y);

                // Right neighbor
                if (x < grid_width - 1) {
                    int neighbor = get_node_id(x + 1, y);
                    Road road;
                    road.from_node = node;
                    road.to_node = neighbor;
                    road.length = length_dist(rng);
                    road.speed_limit = speed_dist(rng);
                    road.max_capacity = capacity_dist(rng);
                    road.current_cars = 0;
                    adjacency_list[node].push_back(road);

                    // back edge
                    Road back = road;
                    back.from_node = neighbor;
                    back.to_node = node;
                    adjacency_list[neighbor].push_back(back);
                }

                // Bottom neighbor
                if (y < grid_height - 1) {
                    int neighbor = get_node_id(x, y + 1);
                    Road road;
                    road.from_node = node;
                    road.to_node = neighbor;
                    road.length = length_dist(rng);
                    road.speed_limit = speed_dist(rng);
                    road.max_capacity = capacity_dist(rng);
                    road.current_cars = 0;
                    adjacency_list[node].push_back(road);

                    // back edge
                    Road back = road;
                    back.from_node = neighbor;
                    back.to_node = node;
                    adjacency_list[neighbor].push_back(back);
                }
            }
        }
    }

    std::vector<int> find_shortest_path(int start, int goal) {
        int n = get_total_nodes();
        const double INF = 1e18;
        std::vector<double> dist(n, INF);
        std::vector<int> prev(n, -1);
        using PDI = std::pair<double,int>;
        std::priority_queue<PDI, std::vector<PDI>, std::greater<PDI>> pq;

        dist[start] = 0.0;
        pq.push({0.0, start});

        while (!pq.empty()) {
            auto [d, u] = pq.top(); pq.pop();
            if (d > dist[u]) continue;
            if (u == goal) break;

            auto it = adjacency_list.find(u);
            if (it == adjacency_list.end()) continue;
            for (const Road &road : it->second) {
                int v = road.to_node;
                double w = road.get_travel_time();
                if (dist[u] + w < dist[v]) {
                    dist[v] = dist[u] + w;
                    prev[v] = u;
                    pq.push({dist[v], v});
                }
            }
        }

        std::vector<int> path;
        if (dist[goal] == INF) {
            // no path
            return path;
        }

        int cur = goal;
        while (cur != -1) {
            path.push_back(cur);
            if (cur == start) break;
            cur = prev[cur];
        }
        std::reverse(path.begin(), path.end());
        return path;
    }

    std::vector<Car> generate_random_cars(int num_cars, int seed_offset = 0) {
        std::vector<Car> cars;
        std::mt19937 local_rng(rng());
        local_rng.seed(rng() + seed_offset + 1);
        std::uniform_int_distribution<> node_dist(0, grid_width * grid_height - 1);

        for (int i = 0; i < num_cars; ++i) {
            Car car;
            car.id = i;
            car.current_node = node_dist(local_rng);
            car.destination_node = node_dist(local_rng);

            while (car.destination_node == car.current_node) {
                car.destination_node = node_dist(local_rng);
            }

            car.route = find_shortest_path(car.current_node, car.destination_node);
            car.route_index = 0;
            car.position_on_road = 0.0;
            car.speed = 50.0;
            car.completed = false;

            if (!car.route.empty()) cars.push_back(car);
        }

        return cars;
    }

    Road* get_road(int from, int to) {
        auto it = adjacency_list.find(from);
        if (it == adjacency_list.end()) return nullptr;
        for (auto &road : it->second) {
            if (road.to_node == to) return &road;
        }
        return nullptr;
    }

    const std::unordered_map<int, std::vector<Road>>& get_adjacency() const { return adjacency_list; }

    int get_total_nodes() const { return grid_width * grid_height; }
    int get_width() const { return grid_width; }
    int get_height() const { return grid_height; }
};

// Serialization helpers: pack Car into vector<double> for MPI send/recv
static std::vector<double> serialize_car(const Car &c) {
    std::vector<double> out;
    out.reserve(8 + c.route.size());
    out.push_back((double)c.id);
    out.push_back((double)c.current_node);
    out.push_back((double)c.destination_node);
    out.push_back((double)c.route_index);
    out.push_back(c.position_on_road);
    out.push_back(c.speed);
    out.push_back(c.completed ? 1.0 : 0.0);
    out.push_back((double)c.route.size());
    for (int v : c.route) out.push_back((double)v);
    return out;
}

static Car deserialize_car(const std::vector<double> &buf) {
    Car c;
    if (buf.size() < 8) return c;
    size_t idx = 0;
    c.id = (int)buf[idx++];
    c.current_node = (int)buf[idx++];
    c.destination_node = (int)buf[idx++];
    c.route_index = (int)buf[idx++];
    c.position_on_road = buf[idx++];
    c.speed = buf[idx++];
    c.completed = (buf[idx++] > 0.5);
    int route_len = (int)buf[idx++];
    c.route.clear();
    for (int i = 0; i < route_len && idx < buf.size(); ++i) c.route.push_back((int)buf[idx++]);
    return c;
}

class MPISimulator {
private:
    int rank, size;
    CityGraph* graph;
    std::vector<Car> local_cars;
    int start_node, end_node;

public:
    MPISimulator(CityGraph* g) : graph(g) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        int total_nodes = graph->get_total_nodes();
        int nodes_per_process = (total_nodes + size - 1) / size; // ceiling
        start_node = rank * nodes_per_process;
        end_node = std::min(total_nodes, (rank + 1) * nodes_per_process);
    }

    void distribute_cars_from_root(std::vector<Car>& all_cars) {
        local_cars.clear();
        int total_nodes = graph->get_total_nodes();
        int nodes_per_process = (total_nodes + size - 1) / size;

        if (rank == 0) {
            // bucket cars for each rank
            std::vector<std::vector<double>> buckets(size);
            for (auto &car : all_cars) {
                int target_rank = car.current_node / nodes_per_process;
                if (target_rank >= size) target_rank = size - 1;
                auto packed = serialize_car(car);
                // prefix length then append
                buckets[target_rank].push_back((double)packed.size());
                buckets[target_rank].insert(buckets[target_rank].end(), packed.begin(), packed.end());
            }

            // send sizes first
            for (int r = 1; r < size; ++r) {
                int bytes = (int)buckets[r].size();
                MPI_Send(&bytes, 1, MPI_INT, r, 0, MPI_COMM_WORLD);
                if (bytes > 0) MPI_Send(buckets[r].data(), bytes, MPI_DOUBLE, r, 1, MPI_COMM_WORLD);
            }

            // local copy for rank 0
            if (!buckets[0].empty()) {
                // unpack
                size_t idx = 0; auto &buf = buckets[0];
                while (idx < buf.size()) {
                    int len = (int)buf[idx++];
                    std::vector<double> packed;
                    packed.insert(packed.end(), buf.begin() + idx, buf.begin() + idx + len);
                    idx += len;
                    local_cars.push_back(deserialize_car(packed));
                }
            }

        } else {
            int recv_bytes = 0;
            MPI_Recv(&recv_bytes, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (recv_bytes > 0) {
                std::vector<double> buf(recv_bytes);
                MPI_Recv(buf.data(), recv_bytes, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                size_t idx = 0;
                while (idx < buf.size()) {
                    int len = (int)buf[idx++];
                    std::vector<double> packed;
                    packed.insert(packed.end(), buf.begin() + idx, buf.begin() + idx + len);
                    idx += len;
                    local_cars.push_back(deserialize_car(packed));
                }
            }
        }

        if (rank == 0) {
            std::cout << "Distributed cars to " << size << " ranks.\n";
        }
    }

    // Exchange cars that moved across process boundaries (full serialization)
    void exchange_boundary_cars() {
        int total_nodes = graph->get_total_nodes();
        int nodes_per_process = (total_nodes + size - 1) / size;

        std::vector<std::vector<double>> send_bufs(size);
        std::vector<Car> remaining;

        for (auto &car : local_cars) {
            // determine owner based on current_node
            int owner = car.current_node / nodes_per_process;
            if (owner < 0) owner = 0;
            if (owner >= size) owner = size - 1;
            if (owner == rank) remaining.push_back(car);
            else {
                auto packed = serialize_car(car);
                send_bufs[owner].push_back((double)packed.size());
                send_bufs[owner].insert(send_bufs[owner].end(), packed.begin(), packed.end());
            }
        }

        local_cars.swap(remaining);

        // send counts
        for (int r = 0; r < size; ++r) {
            if (r == rank) continue;
            int bytes = (int)send_bufs[r].size();
            MPI_Send(&bytes, 1, MPI_INT, r, 10, MPI_COMM_WORLD);
        }

        // recv counts
        std::vector<int> recv_bytes(size,0);
        for (int r = 0; r < size; ++r) {
            if (r == rank) continue;
            MPI_Recv(&recv_bytes[r], 1, MPI_INT, r, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // send actual buffers
        for (int r = 0; r < size; ++r) {
            if (r == rank) continue;
            if (!send_bufs[r].empty()) MPI_Send(send_bufs[r].data(), (int)send_bufs[r].size(), MPI_DOUBLE, r, 11, MPI_COMM_WORLD);
        }

        // receive actual buffers
        for (int r = 0; r < size; ++r) {
            if (r == rank) continue;
            int b = recv_bytes[r];
            if (b > 0) {
                std::vector<double> buf(b);
                MPI_Recv(buf.data(), b, MPI_DOUBLE, r, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                size_t idx = 0;
                while (idx < buf.size()) {
                    int len = (int)buf[idx++];
                    std::vector<double> packed;
                    packed.insert(packed.end(), buf.begin() + idx, buf.begin() + idx + len);
                    idx += len;
                    local_cars.push_back(deserialize_car(packed));
                }
            }
        }
    }

    double compute_local_congestion() {
        double total_congestion = 0.0;
        int count = 0;
        for (int node = start_node; node < end_node; ++node) {
            auto it = graph->get_adjacency().find(node);
            if (it == graph->get_adjacency().end()) continue;
            for (const Road &road : it->second) {
                total_congestion += road.get_congestion();
                count++;
            }
        }
        return count > 0 ? total_congestion / count : 0.0;
    }

    double gather_global_congestion() {
        double local_congestion = compute_local_congestion();
        double global_congestion = 0.0;
        MPI_Reduce(&local_congestion, &global_congestion, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) global_congestion /= size;
        return global_congestion;
    }

    int get_rank() const { return rank; }
    int get_size() const { return size; }
    std::vector<Car>& get_local_cars() { return local_cars; }
};

class OpenMPEngine {
public:
    static void reset_road_counts(CityGraph* graph) {
        // single-threaded reset to avoid races
        for (auto &kv : const_cast<std::unordered_map<int, std::vector<Road>>&>(graph->get_adjacency())) {
            for (auto &road : kv.second) road.current_cars = 0;
        }
    }

    static void count_cars_on_roads(CityGraph* graph, const std::vector<Car>& cars) {
        // increment counts atomically
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < (int)cars.size(); ++i) {
            const Car &c = cars[i];
            if (c.completed) continue;
            if (c.route_index >= (int)c.route.size() - 1) continue;
            int u = c.route[c.route_index];
            int v = c.route[c.route_index + 1];
            Road* road = graph->get_road(u, v);
            if (road) {
                #pragma omp atomic
                road->current_cars++;
            }
        }
    }

    static void update_cars_parallel(std::vector<Car>& cars, CityGraph* graph, double dt) {
        int num_cars = cars.size();

        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < num_cars; ++i) {
            if (cars[i].completed) continue;
            if (cars[i].route_index >= (int)cars[i].route.size() - 1) {
                cars[i].completed = true;
                continue;
            }

            int current = cars[i].route[cars[i].route_index];
            int next = cars[i].route[cars[i].route_index + 1];
            Road* road = graph->get_road(current, next);
            if (!road) {
                cars[i].completed = true;
                continue;
            }

            double congestion = road->get_congestion();
            double effective_speed = road->speed_limit * (1.0 - 0.5 * congestion);
            double distance_traveled_fraction = (effective_speed * dt) / road->length; // dt in hours
            cars[i].position_on_road += distance_traveled_fraction;

            if (cars[i].position_on_road >= 1.0) {
                cars[i].position_on_road = 0.0;
                cars[i].route_index++;
                cars[i].current_node = next;
                if (cars[i].route_index >= (int)cars[i].route.size() - 1) {
                    cars[i].completed = true;
                }
            }
        }
    }
};

void export_congestion_data(CityGraph* graph, int timestep, int rank) {
    std::ofstream file("congestion_data_rank" + std::to_string(rank) + "_t" + std::to_string(timestep) + ".csv");
    file << "from_x,from_y,to_x,to_y,congestion\n";

    for (int y = 0; y < graph->get_height(); ++y) {
        for (int x = 0; x < graph->get_width(); ++x) {
            int node = graph->get_node_id(x, y);
            // right
            if (x < graph->get_width() - 1) {
                int neighbor = graph->get_node_id(x + 1, y);
                Road* road = graph->get_road(node, neighbor);
                if (road) file << x << "," << y << "," << (x+1) << "," << y << "," << road->get_congestion() << "\n";
            }
            // bottom
            if (y < graph->get_height() - 1) {
                int neighbor = graph->get_node_id(x, y + 1);
                Road* road = graph->get_road(node, neighbor);
                if (road) file << x << "," << y << "," << x << "," << (y+1) << "," << road->get_congestion() << "\n";
            }
        }
    }

    file.close();
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Simulation parameters (you can change via CLI/env if desired)
    const int GRID_SIZE = 10;
    const int NUM_CARS = 200;
    const int NUM_TIMESTEPS = 100;
    const double DT = 0.05;  // hours (~3 minutes)

    CityGraph graph(GRID_SIZE, GRID_SIZE, 42);

    if (rank == 0) {
        int omp_threads = omp_get_max_threads();
        std::cout << "=== Traffic Simulation Started ===\n";
        std::cout << "Grid: " << GRID_SIZE << "x" << GRID_SIZE << "\n";
        std::cout << "Cars: " << NUM_CARS << "\n";
        std::cout << "MPI Processes: " << size << "\n";
        std::cout << "OpenMP Threads (max): " << omp_threads << "\n\n";
    }

    MPISimulator mpi_sim(&graph);

    std::vector<Car> all_cars;
    if (rank == 0) {
        all_cars = graph.generate_random_cars(NUM_CARS);
    }

    // Distribute cars from rank 0 to owners
    mpi_sim.distribute_cars_from_root(all_cars);

    double start_time = MPI_Wtime();

    for (int t = 0; t < NUM_TIMESTEPS; ++t) {
        // Reset road counts first (single-threaded)
        OpenMPEngine::reset_road_counts(&graph);

        // Count cars per road (atomic increments)
        OpenMPEngine::count_cars_on_roads(&graph, mpi_sim.get_local_cars());

        // Update cars using OpenMP
        OpenMPEngine::update_cars_parallel(mpi_sim.get_local_cars(), &graph, DT);

        // Exchange cars that moved across MPI boundaries
        mpi_sim.exchange_boundary_cars();

        if (t % 10 == 0) {
            double global_congestion = mpi_sim.gather_global_congestion();
            if (rank == 0) {
                std::cout << "Timestep " << t << " - Global Congestion: " << global_congestion << "\n";
            }

            export_congestion_data(&graph, t, rank);
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    double end_time = MPI_Wtime();

    // Gather some final stats
    int local_completed = 0;
    for (auto &c : mpi_sim.get_local_cars()) if (c.completed) local_completed++;
    int global_completed = 0;
    MPI_Reduce(&local_completed, &global_completed, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "\n=== Simulation Complete ===\n";
        std::cout << "Total time: " << (end_time - start_time) << " seconds\n";
        std::cout << "Total completed cars: " << global_completed << " / " << NUM_CARS << "\n";
    }

    MPI_Finalize();
    return 0;
}

