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

// ============================================================================
// 1. CITY MODELER - Data Structures & Graph Representation
// ============================================================================

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
        return (double)current_cars / max_capacity;
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
    
    Car() : id(0), current_node(0), destination_node(0), 
            route_index(0), position_on_road(0.0), speed(0.0), completed(false) {}
};

class CityGraph {
private:
    int grid_width;
    int grid_height;
    std::unordered_map<int, std::vector<std::pair<int, Road>>> adjacency_list;
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
        
        // Create grid connections
        for (int y = 0; y < grid_height; ++y) {
            for (int x = 0; x < grid_width; ++x) {
                int node = get_node_id(x, y);
                
                // Connect to right neighbor
                if (x < grid_width - 1) {
                    int neighbor = get_node_id(x + 1, y);
                    Road road;
                    road.from_node = node;
                    road.to_node = neighbor;
                    road.length = length_dist(rng);
                    road.speed_limit = speed_dist(rng);
                    road.max_capacity = capacity_dist(rng);
                    road.current_cars = 0;
                    adjacency_list[node].push_back({neighbor, road});
                }
                
                // Connect to bottom neighbor
                if (y < grid_height - 1) {
                    int neighbor = get_node_id(x, y + 1);
                    Road road;
                    road.from_node = node;
                    road.to_node = neighbor;
                    road.length = length_dist(rng);
                    road.speed_limit = speed_dist(rng);
                    road.max_capacity = capacity_dist(rng);
                    road.current_cars = 0;
                    adjacency_list[node].push_back({neighbor, road});
                }
                
                // Connect to left neighbor (bidirectional)
                if (x > 0) {
                    int neighbor = get_node_id(x - 1, y);
                    Road road;
                    road.from_node = node;
                    road.to_node = neighbor;
                    road.length = length_dist(rng);
                    road.speed_limit = speed_dist(rng);
                    road.max_capacity = capacity_dist(rng);
                    road.current_cars = 0;
                    adjacency_list[node].push_back({neighbor, road});
                }
                
                // Connect to top neighbor (bidirectional)
                if (y > 0) {
                    int neighbor = get_node_id(x, y - 1);
                    Road road;
                    road.from_node = node;
                    road.to_node = neighbor;
                    road.length = length_dist(rng);
                    road.speed_limit = speed_dist(rng);
                    road.max_capacity = capacity_dist(rng);
                    road.current_cars = 0;
                    adjacency_list[node].push_back({neighbor, road});
                }
            }
        }
    }
    
    std::vector<int> find_shortest_path(int start, int goal) {
        std::unordered_map<int, double> distance;
        std::unordered_map<int, int> previous;
        
        auto cmp = [&](int a, int b) { return distance[a] > distance[b]; };
        std::priority_queue<int, std::vector<int>, decltype(cmp)> pq(cmp);
        
        distance[start] = 0.0;
        pq.push(start);
        
        while (!pq.empty()) {
            int current = pq.top();
            pq.pop();
            
            if (current == goal) break;
            
            if (adjacency_list.find(current) != adjacency_list.end()) {
                for (auto& [neighbor, road] : adjacency_list[current]) {
                    double new_dist = distance[current] + road.get_travel_time();
                    
                    if (distance.find(neighbor) == distance.end() || new_dist < distance[neighbor]) {
                        distance[neighbor] = new_dist;
                        previous[neighbor] = current;
                        pq.push(neighbor);
                    }
                }
            }
        }
        
        // Reconstruct path
        std::vector<int> path;
        int current = goal;
        while (current != start && previous.find(current) != previous.end()) {
            path.push_back(current);
            current = previous[current];
        }
        path.push_back(start);
        std::reverse(path.begin(), path.end());
        
        return path;
    }
    
    std::vector<Car> generate_random_cars(int num_cars) {
        std::vector<Car> cars;
        std::uniform_int_distribution<> node_dist(0, grid_width * grid_height - 1);
        
        for (int i = 0; i < num_cars; ++i) {
            Car car;
            car.id = i;
            car.current_node = node_dist(rng);
            car.destination_node = node_dist(rng);
            
            while (car.destination_node == car.current_node) {
                car.destination_node = node_dist(rng);
            }
            
            car.route = find_shortest_path(car.current_node, car.destination_node);
            car.route_index = 0;
            car.position_on_road = 0.0;
            car.speed = 50.0;
            car.completed = false;
            
            cars.push_back(car);
        }
        
        return cars;
    }
    
    Road* get_road(int from, int to) {
        if (adjacency_list.find(from) != adjacency_list.end()) {
            for (auto& [neighbor, road] : adjacency_list[from]) {
                if (neighbor == to) return &road;
            }
        }
        return nullptr;
    }
    
    int get_total_nodes() const { return grid_width * grid_height; }
    int get_width() const { return grid_width; }
    int get_height() const { return grid_height; }
};

// ============================================================================
// 2. MPI ENGINEER - Distributed Computing Logic
// ============================================================================

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
        
        // Divide nodes among processes
        int total_nodes = graph->get_total_nodes();
        int nodes_per_process = total_nodes / size;
        start_node = rank * nodes_per_process;
        end_node = (rank == size - 1) ? total_nodes : (rank + 1) * nodes_per_process;
    }
    
    void distribute_cars(const std::vector<Car>& all_cars) {
        local_cars.clear();
        for (const auto& car : all_cars) {
            if (car.current_node >= start_node && car.current_node < end_node) {
                local_cars.push_back(car);
            }
        }
        
        if (rank == 0) {
            std::cout << "Distributed " << all_cars.size() << " cars among " 
                      << size << " processes\n";
        }
    }
    
    void exchange_boundary_cars(std::vector<Car>& outgoing_cars) {
        // Cars that moved to another process's region
        std::vector<std::vector<Car>> send_buffers(size);
        std::vector<Car> remaining_cars;
        
        for (auto& car : local_cars) {
            if (car.current_node < start_node || car.current_node >= end_node) {
                int target_rank = car.current_node / ((graph->get_total_nodes() + size - 1) / size);
                target_rank = std::min(target_rank, size - 1);
                send_buffers[target_rank].push_back(car);
                outgoing_cars.push_back(car);
            } else {
                remaining_cars.push_back(car);
            }
        }
        
        local_cars = remaining_cars;
        
        // MPI communication for car exchange
        for (int i = 0; i < size; ++i) {
            if (i == rank) continue;
            
            int send_count = send_buffers[i].size();
            MPI_Send(&send_count, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            
            if (send_count > 0) {
                // In real implementation, serialize Car data properly
                // This is simplified for demonstration
            }
        }
        
        for (int i = 0; i < size; ++i) {
            if (i == rank) continue;
            
            int recv_count;
            MPI_Recv(&recv_count, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            // Receive and add cars
        }
    }
    
    double compute_local_congestion() {
        double total_congestion = 0.0;
        int count = 0;
        
        for (int node = start_node; node < end_node; ++node) {
            for (int neighbor = 0; neighbor < graph->get_total_nodes(); ++neighbor) {
                Road* road = graph->get_road(node, neighbor);
                if (road) {
                    total_congestion += road->get_congestion();
                    count++;
                }
            }
        }
        
        return count > 0 ? total_congestion / count : 0.0;
    }
    
    double gather_global_congestion() {
        double local_congestion = compute_local_congestion();
        double global_congestion;
        
        MPI_Reduce(&local_congestion, &global_congestion, 1, MPI_DOUBLE, 
                   MPI_SUM, 0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            global_congestion /= size;
        }
        
        return global_congestion;
    }
    
    int get_rank() const { return rank; }
    int get_size() const { return size; }
    std::vector<Car>& get_local_cars() { return local_cars; }
};

// ============================================================================
// 3. OPENMP ENGINEER - Parallel Car Updates
// ============================================================================

class OpenMPEngine {
public:
    static void update_cars_parallel(std::vector<Car>& cars, CityGraph* graph, double dt) {
        int num_cars = cars.size();
        
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < num_cars; ++i) {
            if (cars[i].completed) continue;
            
            if (cars[i].route_index >= cars[i].route.size() - 1) {
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
            
            // Update position based on speed and congestion
            double effective_speed = road->speed_limit * (1.0 - 0.5 * road->get_congestion());
            double distance_traveled = effective_speed * dt / road->length;
            
            cars[i].position_on_road += distance_traveled;
            
            if (cars[i].position_on_road >= 1.0) {
                cars[i].position_on_road = 0.0;
                cars[i].route_index++;
                cars[i].current_node = next;
                
                if (cars[i].route_index >= cars[i].route.size() - 1) {
                    cars[i].completed = true;
                }
            }
        }
    }
    
    static void update_road_congestion_parallel(CityGraph* graph, const std::vector<Car>& cars) {
        // Reset all road counts
        #pragma omp parallel
        {
            // This is simplified - in real implementation, use proper synchronization
        }
        
        // Count cars on each road
        #pragma omp parallel for
        for (int i = 0; i < cars.size(); ++i) {
            if (cars[i].completed || cars[i].route_index >= cars[i].route.size() - 1) 
                continue;
            
            int current = cars[i].route[cars[i].route_index];
            int next = cars[i].route[cars[i].route_index + 1];
            
            Road* road = graph->get_road(current, next);
            if (road) {
                #pragma omp atomic
                road->current_cars++;
            }
        }
    }
};

// ============================================================================
// 4. MAIN SIMULATION LOOP
// ============================================================================

void export_congestion_data(CityGraph* graph, int timestep, int rank) {
    std::ofstream file("congestion_data_rank" + std::to_string(rank) + 
                       "_t" + std::to_string(timestep) + ".csv");
    file << "from_x,from_y,to_x,to_y,congestion\n";
    
    for (int y = 0; y < graph->get_height(); ++y) {
        for (int x = 0; x < graph->get_width(); ++x) {
            int node = graph->get_node_id(x, y);
            
            // Check right neighbor
            if (x < graph->get_width() - 1) {
                int neighbor = graph->get_node_id(x + 1, y);
                Road* road = graph->get_road(node, neighbor);
                if (road) {
                    file << x << "," << y << "," << (x+1) << "," << y << "," 
                         << road->get_congestion() << "\n";
                }
            }
            
            // Check bottom neighbor
            if (y < graph->get_height() - 1) {
                int neighbor = graph->get_node_id(x, y + 1);
                Road* road = graph->get_road(node, neighbor);
                if (road) {
                    file << x << "," << y << "," << x << "," << (y+1) << "," 
                         << road->get_congestion() << "\n";
                }
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
    
    // Simulation parameters
    const int GRID_SIZE = 10;
    const int NUM_CARS = 100;
    const int NUM_TIMESTEPS = 50;
    const double DT = 0.1;  // time step in hours
    
    // Initialize city graph (same on all processes)
    CityGraph graph(GRID_SIZE, GRID_SIZE, 42);
    
    if (rank == 0) {
        std::cout << "=== Traffic Simulation Started ===\n";
        std::cout << "Grid: " << GRID_SIZE << "x" << GRID_SIZE << "\n";
        std::cout << "Cars: " << NUM_CARS << "\n";
        std::cout << "MPI Processes: " << size << "\n";
        std::cout << "OpenMP Threads: " << omp_get_max_threads() << "\n\n";
    }
    
    // Generate cars (only rank 0)
    std::vector<Car> all_cars;
    if (rank == 0) {
        all_cars = graph.generate_random_cars(NUM_CARS);
    }
    
    // Distribute cars using MPI
    MPISimulator mpi_sim(&graph);
    mpi_sim.distribute_cars(all_cars);
    
    // Timing
    double start_time = MPI_Wtime();
    
    // Main simulation loop
    for (int t = 0; t < NUM_TIMESTEPS; ++t) {
        // Update cars using OpenMP
        OpenMPEngine::update_cars_parallel(mpi_sim.get_local_cars(), &graph, DT);
        
        // Update road congestion
        OpenMPEngine::update_road_congestion_parallel(&graph, mpi_sim.get_local_cars());
        
        // Exchange boundary cars
        std::vector<Car> outgoing;
        mpi_sim.exchange_boundary_cars(outgoing);
        
        // Compute and gather congestion metrics
        if (t % 10 == 0) {
            double congestion = mpi_sim.gather_global_congestion();
            if (rank == 0) {
                std::cout << "Timestep " << t << " - Global Congestion: " 
                          << congestion << "\n";
            }
            
            // Export data for visualization
            export_congestion_data(&graph, t, rank);
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    double end_time = MPI_Wtime();
    
    if (rank == 0) {
        std::cout << "\n=== Simulation Complete ===\n";
        std::cout << "Total time: " << (end_time - start_time) << " seconds\n";
    }
    
    MPI_Finalize();
    return 0;
}
