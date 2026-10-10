#pragma once
#include "pf/pf.hpp"
#include <memory>

namespace pf::detail {
struct Stop final : std::exception {
    std::string reason;
    explicit Stop(std::string value) : reason(std::move(value)) {}
    const char* what() const noexcept override { return reason.c_str(); }
};

struct LocalCandidate {
    const Bounds& bounds;
    Vector x;
    double objective = infinity;
};

struct Context {
    const Problem& problem;
    const Options& options;
    const Bounds& original_bounds;
    Counters counters;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    double local_seconds = 0, local_work_seconds = 0, best = infinity;
    Vector best_x;
    double active_ball_radius;
    bool require_local_feasibility = false; // constrained multistart incumbent
    LocalCandidate* local_candidate = nullptr; // scoped to one refinement, never shared by workers
    Context(const Problem& p, const Options& o, const Bounds& b)
        : problem(p), options(o), original_bounds(b), active_ball_radius(o.radius) {}
    int ball_dimension() const;
    bool local_feasible(const Vector& x) const;
    double seconds() const;
    void check(bool objective_request = false) const;
    void observe(const Vector& x, double f);
    double evaluate(const Vector& x, bool local = false);
    std::vector<double> batch(const Population& population);
    Vector gradient(const Vector& x);
    Matrix hessian(const Vector& x);
};

struct LocalResult { Vector x; double objective; int status; };
LocalResult refine(Context& context, const Vector& x, const Bounds& bounds, LocalCandidate* retained = nullptr);
LocalResult refine_ipopt(Context& context, const Vector& x, const Bounds& bounds, double initial_value);
LocalResult refine_lj_newton(Context& context, const Vector& x, const Bounds& bounds, double initial_value);
Vector local_projection(const Vector& x, const Bounds& bounds, int ball_dimension, double radius);
Vector projected_gradient(const Vector& x, const Vector& gradient, const Bounds& bounds);
class LocalPool {
public:
    explicit LocalPool(Context& context);
    ~LocalPool();
    void shutdown();
    LocalPool(const LocalPool&) = delete;
    LocalPool& operator=(const LocalPool&) = delete;
    void refine_selected(Population& particles, const std::vector<std::size_t>& indices, const Bounds& bounds,
                         std::vector<double>* known_values = nullptr);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
int local_worker_main(int fd);
} // namespace pf::detail
