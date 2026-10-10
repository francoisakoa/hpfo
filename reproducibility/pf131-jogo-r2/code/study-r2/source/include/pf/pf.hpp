#pragma once

#include <Eigen/Core>
#include <chrono>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <type_traits>
#include <climits>

#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ > 0)
#error "PF requires IEEE binary64 semantics: disable fast-math and finite-math-only"
#endif

namespace pf {
using Real = double;
static_assert(sizeof(Real)*CHAR_BIT==64 && std::numeric_limits<Real>::is_iec559 &&
              std::numeric_limits<Real>::radix==2 && std::numeric_limits<Real>::digits==53 &&
              std::numeric_limits<Real>::max_exponent==1024,
              "PF requires IEEE 754 binary64 (NumPy float64 equivalent)");
using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;
static_assert(std::is_same_v<Vector::Scalar,Real> && std::is_same_v<Matrix::Scalar,Real>);
using Population = std::vector<Vector>;
constexpr double infinity = std::numeric_limits<double>::infinity();
constexpr const char* version = "1.3.1-article-study-r1";

struct Bounds {
    Vector lower, upper;
    void validate(Eigen::Index dimension) const;
    Vector project(const Vector& x) const;
    bool contains(const Vector& x, double tolerance = 0.0) const;
};

// A problem must be reentrant when batch_threads > 1. The supplied LJ problem is.
class Problem {
public:
    virtual ~Problem() = default;
    virtual Eigen::Index dimension() const = 0;
    virtual double value(const Vector& x) const = 0;
    virtual Vector gradient(const Vector& x) const = 0;
    virtual bool has_hessian() const { return false; }
    virtual Matrix hessian(const Vector&) const;
};

class LennardJones final : public Problem {
public:
    explicit LennardJones(int atoms, int spatial_dimension = 3);
    Eigen::Index dimension() const override;
    double value(const Vector& x) const override;
    Vector gradient(const Vector& x) const override;
    bool has_hessian() const override { return true; }
    Matrix hessian(const Vector& x) const override;
    int atoms() const { return atoms_; }
    int spatial_dimension() const { return spatial_dimension_; }
private:
    int atoms_, spatial_dimension_;
    void check(const Vector& x) const;
};

// NumPy Generator(PCG64(SeedSequence(seed))) for scalar nonnegative uint64 seeds.
// Draws stay on the coordinator thread, independent of worker scheduling.
class Random {
public:
    explicit Random(std::uint64_t seed);
    std::uint64_t raw();
    double uniform();
    double normal();
    std::size_t index(std::size_t size);
    std::vector<std::size_t> choice(std::size_t size, std::size_t count, bool replacement = false);
    // Skip raw PCG64 draws in O(log count); preserves the buffered uint32 half.
    void advance(std::uint64_t count);
    // CPU sklearn's random_state=int uses legacy RandomState MT19937, not PCG64.
    static Random sklearn(std::uint32_t seed);
private:
    std::uint64_t state_low_ = 0, state_high_ = 0, inc_low_ = 0, inc_high_ = 0;
    std::uint32_t next32();
    std::uint32_t buffered_ = 0;
    bool has_buffered_ = false, legacy_ = false;
    std::mt19937 legacy_engine_;
};

struct RunSeeds {
    std::uint64_t base;
    int atoms, run;
    std::uint32_t init, pf;
};
struct PythonInitialization {
    RunSeeds seeds;
    Vector x0; // Python main() point; the supplied PF loop does not use it.
    Population population; // Actual PF starts generated with seed_pf, full box.
};

enum class Flow { invertible, fast, mutation };
enum class Mapping { linear, residual, affine, cnf };
enum class Clustering { kmeans, dbscan, gmm };
enum class LocalMethod { ipopt, projected_lbfgs, none, lj_newton };

struct Options {
    int max_iterations = 1000;
    int no_improve_stop = 400; // 0 disables global stagnation stopping
    int particles = 50;
    int min_particles = 5;
    int refine_top_k = 5;
    bool refine_all = false;
    bool final_refinement = true;
    Flow flow = Flow::invertible;
    Mapping mapping = Mapping::linear;
    Clustering clustering = Clustering::kmeans;
    // Additive study controls. Defaults preserve the original PF131 trajectory.
    bool ipf_enabled = true;
    bool ipf_endpoint_schedule = false;
    double ipf_alpha_initial = 0.98, ipf_alpha_final = 0.05;
    bool adaptive_annealing = true;
    bool island_exchange = true; // isolates the every-10 exchange from every-20 injection
    bool ess_rejuvenation = true;
    bool covariance_mutation = false;
    bool kicks = true;
    bool islands = true;
    bool diversity_injection = true;
    double radius = 5.0;
    double mutation_sigma = 0.05;
    double beta_initial = 1e-4;
    double beta_max = 5.0;
    double lambda_share = 0.01;
    double lambda_novelty = 0.01;
    double target = -infinity;
    double target_tolerance = 1e-4;
    double max_seconds = 20000.0;
    std::uint64_t max_objective_evaluations = 0; // zero = unlimited
    std::uint64_t seed = 2024;
    std::optional<RunSeeds> run_seeds; // CLI provenance; seed above is the effective PF seed
    int batch_threads = 1;
    int local_workers = 1;
    std::string worker_executable; // empty: pf_local_worker beside the current executable
    LocalMethod local_method = LocalMethod::ipopt;
    int local_max_iterations = 5000;
    double local_tolerance = 1e-8;
    std::string newton_linear_solver = "cg"; // cg | ldl (Eigen LDLT) | ma27 | mumps
    std::string newton_linear_ordering = "auto"; // auto | metis (ldl or mumps)
    std::string hsl_library; // explicit HSL shared library, used by Ipopt and native MA27
    int newton_cg_max_iterations = 0; // 0: dimension
    double newton_initial_radius = 0.2; // trust radius per sqrt(atom count)
    std::string ipopt_linear_solver = "mumps";
    bool ipopt_exact_hessian = true;
    bool ipopt_ball_constraints = true; // LJ only: one ball per atom, local refinement only
    int ipopt_print_level = 0;
    int kmeans_restarts = 1;
    int cluster_max_iterations = 300;
    int gmm_max_iterations = 100;
    double dbscan_epsilon = 0.5;
    int dbscan_min_samples = 2;
    void validate(const Problem& problem, const Bounds& bounds) const;
};

struct Counters {
    std::uint64_t objective_pf = 0, objective_local = 0;
    std::uint64_t gradient = 0, hessian = 0, invalid_objective = 0;
    std::uint64_t local_calls = 0, local_converged = 0;
    std::uint64_t local_failed = 0, local_limited = 0;
    std::uint64_t local_improvements = 0;
    std::uint64_t linear_analyses = 0, linear_factorizations = 0, linear_solves = 0, linear_cg_fallbacks = 0;
    std::uint64_t linear_metis_analyses = 0, mumps_user_ordering_analyses = 0;
    std::uint64_t hessian_products = 0, newton_iterations = 0;
    std::uint64_t newton_accepted = 0, newton_rejected = 0, negative_curvature = 0;
    std::uint64_t objective_total() const { return objective_pf + objective_local; }
};

struct Iteration {
    int iteration = 0, particles = 0;
    double best = infinity, current_best = infinity;
    double beta_used = 0, beta_next = 0, ess = 0;
    double diversity_before = 0, diversity_after = 0;
    double mutation_sigma = 0, radius = 0;
    double ipf_alpha = 0;
    bool resampled = false, kick = false, island_exchange = false;
    int injected = 0;
    std::uint64_t objective_evaluations = 0;
    double seconds = 0;
    std::vector<double> objectives, weights;
};

struct Result {
    Vector x;
    double objective = infinity;
    std::string status = "not_started";
    int iterations = 0;
    bool target_reached = false;
    double seconds = 0, local_seconds = 0, local_work_seconds = 0;
    double projected_gradient_inf = infinity;
    double max_atom_radius = 0, active_local_radius = 0;
    std::uint64_t audit_objective_evaluations = 0;
    Counters counters;
    std::vector<Iteration> history;
};

struct WeightResult { std::vector<double> weights; double ess; };
double ipf_step(const Options& options, int iteration);
Vector map_particle(const Vector& x, const Vector& best, double alpha, Mapping mapping);
std::vector<double> sharing_factors(const Population& particles, double sigma, int threads = 1);
std::vector<double> novelty_scores(const Population& particles, const Population& archive, int threads = 1);
WeightResult compute_weights(const std::vector<double>& objectives, double beta,
                             const Population& particles, const Population& archive,
                             double sigma_share, double lambda_share, double lambda_novelty, int threads = 1);
double diversity(const Population& particles, int threads = 1);
double next_beta(int iteration, int max_iterations, int initial_particles,
                 double ess, double beta, double factor, double beta_initial, double beta_max);
std::vector<std::size_t> stratified_indices(const std::vector<double>& weights,
                                          const std::vector<double>& uniform_draws);
std::vector<int> cluster(const Population& particles, Clustering method, Random& rng,
                         const Options& options, int requested_clusters = 0);
Population uniform_population(const Bounds& bounds, int count, Random& rng);
PythonInitialization python_initialization(const Bounds& bounds, int count,
                                            std::uint64_t base, int atoms, int run);
void write_initialization_json(const std::string& path, const PythonInitialization& initial,
                                bool external_population = false);
void save_population(const std::string& path, const Population& population);
Population load_population(const std::string& path, Eigen::Index dimension);
bool ipopt_available();
bool mumps_available();
bool metis_available();
int metis_api_version();
std::string ipopt_version();

class Solver {
public:
    Solver(const Problem& problem, Bounds bounds, Options options = {});
    // Empty input generates a population. Supplied populations are copied.
    // Both paths start the PF stream AFTER particles*dimension uniform draws,
    // so freezing/reloading a generated population preserves the continuation.
    Result solve(const Population& initial = {},
                 const std::function<void(const Iteration&)>& observer = {}) const;
    // Fair multistart baseline: same local solver, original bounds and input population.
    Result multistart(const Population& initial) const;
private:
    const Problem& problem_;
    Bounds bounds_;
    Options options_;
};

void write_result_json(const std::string& path, const Result& result, const Options& options,
                       int atoms, int run, const std::string& algorithm);
void write_history_csv(const std::string& path, const Result& result);
void write_xyz(const std::string& path, const Result& result, int atoms);
// int(hashlib.md5(f"{base}_{atoms}_{run}_{offset}".encode()).hexdigest()[:8], 16)
std::uint32_t derive_seed(std::uint64_t base, int atoms, int run, int offset = 0);
} // namespace pf
