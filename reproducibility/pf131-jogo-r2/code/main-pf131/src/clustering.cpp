#include "pf/pf.hpp"
#include <Eigen/Cholesky>
#include <algorithm>
#include <cmath>
#include <numeric>

namespace pf {
namespace {
std::vector<int> kmeans(const Population& p,int k,Random& rng,const Options& o) {
    const int n=static_cast<int>(p.size());
    std::vector<int> best_labels(n,0);
    double best_loss=infinity;
    Vector mean=Vector::Zero(p[0].size());for(const auto& x:p) mean+=x;mean/=n;
    double variance=0;for(const auto& x:p) variance+=(x-mean).squaredNorm();
    const double tolerance=1e-4*variance/(n*p[0].size());
    const int trials=2+static_cast<int>(std::log(k));
    for(int restart=0;restart<o.kmeans_restarts;++restart) {
        // sklearn KMeans++ picks the first center with uniform sample weights.
        const double draw=rng.uniform(),weight=1.0/n;
        std::size_t first=0;double cumulative=weight;
        while(first+1<p.size()&&cumulative<=draw) {++first;cumulative+=weight;}
        Population centers{p[first]};
        std::vector<double> distances(n,infinity);
        for(int c=1;c<k;++c) {
            double sum=0;
            for(int i=0;i<n;++i) { distances[i]=std::min(distances[i],(p[i]-centers.back()).squaredNorm());sum+=distances[i]; }
            std::size_t next=0;
            if(sum>0 && std::isfinite(sum)) {
                double best_potential=infinity;
                for(int trial=0;trial<trials;++trial) {
                    std::size_t candidate=0;double u=rng.uniform()*sum,cum=distances[0];
                    while(candidate+1<p.size()&&cum<u) cum+=distances[++candidate];
                    double potential=0;
                    for(int i=0;i<n;++i) potential+=std::min(distances[i],(p[i]-p[candidate]).squaredNorm());
                    if(potential<best_potential) {best_potential=potential;next=candidate;}
                }
            } else next=rng.index(p.size());
            centers.push_back(p[next]);
        }
        std::vector<int> labels(n,-1);
        for(int step=0;step<o.cluster_max_iterations;++step) {
            std::vector<int> next(n);
            #pragma omp parallel for schedule(static) num_threads(o.batch_threads) if(n>32)
            for(int i=0;i<n;++i) {
                int chosen=0;double d=(p[i]-centers[0]).squaredNorm();
                for(int c=1;c<k;++c) { double v=(p[i]-centers[c]).squaredNorm();if(v<d) {d=v;chosen=c;} }
                next[i]=chosen;
            }
            if(next==labels) break;
            labels=std::move(next);
            Population means(k,Vector::Zero(p[0].size()));std::vector<int> counts(k,0);
            for(int i=0;i<n;++i) { means[labels[i]]+=p[i];++counts[labels[i]]; }
            double shift=0;
            for(int c=0;c<k;++c) {
                Vector previous_center=centers[c];
                if(counts[c]) centers[c]=means[c]/counts[c];
                else {
                    // Deterministic farthest-point repair; identical points may retain fewer groups.
                    int far=0;double far_d=-1;
                    for(int i=0;i<n;++i) {double d=(p[i]-centers[labels[i]]).squaredNorm();if(d>far_d){far=i;far_d=d;}}
                    centers[c]=p[far];
                }
                shift+=(centers[c]-previous_center).squaredNorm();
            }
            if(shift<=tolerance) break;
        }
        double loss=0;
        for(int i=0;i<n;++i) {
            int chosen=0;double d=(p[i]-centers[0]).squaredNorm();
            for(int c=1;c<k;++c) {double v=(p[i]-centers[c]).squaredNorm();if(v<d){d=v;chosen=c;}}
            labels[i]=chosen;loss+=d;
        }
        if(loss<best_loss) {best_loss=loss;best_labels=labels;}
    }
    return best_labels;
}
std::vector<int> dbscan(const Population& p,const Options& o) {
    const int n=static_cast<int>(p.size());
    std::vector<std::vector<int>> neighbors(n);
    #pragma omp parallel for schedule(static) num_threads(o.batch_threads) if(n>32)
    for(int i=0;i<n;++i) for(int j=0;j<n;++j)
        if((p[i]-p[j]).squaredNorm()<=o.dbscan_epsilon*o.dbscan_epsilon) neighbors[i].push_back(j);
    std::vector<int> labels(n,-2);int label=0;
    for(int i=0;i<n;++i) if(labels[i]==-2) {
        if(static_cast<int>(neighbors[i].size())<o.dbscan_min_samples) {labels[i]=-1;continue;}
        std::vector<int> queue{i};labels[i]=label;
        for(std::size_t at=0;at<queue.size();++at) {
            int j=queue[at];
            if(static_cast<int>(neighbors[j].size())<o.dbscan_min_samples) continue;
            for(int v:neighbors[j]) {
                if(labels[v]==-2) {labels[v]=label;queue.push_back(v);}
                else if(labels[v]==-1) labels[v]=label;
            }
        }
        ++label;
    }
    return labels; // source groups the -1 noise label together too
}
std::vector<int> gmm(const Population& p,int k,Random& rng,const Options& o) {
    const int n=static_cast<int>(p.size());const Eigen::Index d=p[0].size();
    auto labels=kmeans(p,k,rng,o);
    Matrix responsibilities=Matrix::Zero(n,k);
    for(int i=0;i<n;++i) responsibilities(i,labels[i])=1;
    double previous=-infinity;
    for(int step=0;step<o.gmm_max_iterations;++step) {
        Population means(k,Vector::Zero(d));std::vector<Matrix> lower(k);Vector mass=responsibilities.colwise().sum();
        Vector log_constants(k);
        for(int c=0;c<k;++c) {
            double count=std::max(mass[c],1e-12);
            for(int i=0;i<n;++i) means[c]+=responsibilities(i,c)*p[i];
            means[c]/=count;
            Matrix covariance=Matrix::Zero(d,d);
            for(int i=0;i<n;++i) {Vector delta=p[i]-means[c];covariance.noalias()+=(responsibilities(i,c)/count)*(delta*delta.transpose());}
            covariance.diagonal().array()+=1e-6;
            Eigen::LLT<Matrix> decomposition(covariance);
            if(decomposition.info()!=Eigen::Success) throw std::runtime_error("GMM covariance factorization failed");
            lower[c]=decomposition.matrixL();
            log_constants[c]=std::log(count/n)-lower[c].diagonal().array().log().sum()-0.5*d*std::log(6.2831853071795864769);
        }
        std::vector<double> log_likelihood(n);
        #pragma omp parallel for schedule(static) num_threads(o.batch_threads) if(n>32)
        for(int i=0;i<n;++i) {
            Vector scores(k);
            for(int c=0;c<k;++c) {
                Vector v=lower[c].triangularView<Eigen::Lower>().solve(p[i]-means[c]);
                scores[c]=log_constants[c]-0.5*v.squaredNorm();
            }
            double maximum=scores.maxCoeff();Vector probs=(scores.array()-maximum).exp();
            double sum=probs.sum();responsibilities.row(i)=(probs/sum).transpose();
            log_likelihood[i]=maximum+std::log(sum);
        }
        double score=std::accumulate(log_likelihood.begin(),log_likelihood.end(),0.0)/n;
        if(std::abs(score-previous)<1e-3) break;
        previous=score;
    }
    for(int i=0;i<n;++i) {Eigen::Index c;responsibilities.row(i).maxCoeff(&c);labels[i]=static_cast<int>(c);}
    return labels;
}
}
std::vector<int> cluster(const Population& p,Clustering method,Random& rng,const Options& o,int requested) {
    if(p.empty()) throw std::invalid_argument("Cannot cluster an empty population");
    const auto d=p[0].size();
    for(const auto& x:p) if(x.size()!=d||!x.allFinite()) throw std::invalid_argument("Invalid cluster data");
    int k=std::min(static_cast<int>(p.size()),requested>0?requested:std::max(2,int(std::sqrt(p.size()))));
    if(method==Clustering::kmeans) return kmeans(p,k,rng,o);
    if(method==Clustering::dbscan) return dbscan(p,o);
    return gmm(p,k,rng,o);
}
} // namespace pf
