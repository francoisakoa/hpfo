#include "../src/linear.hpp"
#include <iostream>
#include <iomanip>

int main(int argc,char** argv) {
    try {
        pf::Options o;o.local_method=pf::LocalMethod::lj_newton;o.local_tolerance=1e-8;
        std::string backend="ldl";
        for(int i=1;i<argc;++i) {
            std::string key=argv[i];
            if(key=="--help") {std::cout<<"pf_linear_check --backend ldl|ma27|mumps|ipopt-ma27|ipopt-mumps [--ordering auto|metis] [--hsl-library FILE]\n";return 0;}
            if(i+1==argc)throw std::invalid_argument("Missing option value");
            if(key=="--backend")backend=argv[++i];
            else if(key=="--ordering")o.newton_linear_ordering=argv[++i];
            else if(key=="--hsl-library")o.hsl_library=argv[++i];
            else throw std::invalid_argument("Unknown argument: "+key);
        }
        const bool ipopt=backend.rfind("ipopt-",0)==0;
        o.newton_linear_solver=ipopt?"cg":backend;
        if(ipopt){o.local_method=pf::LocalMethod::ipopt;o.ipopt_linear_solver=backend.substr(6);}
        if(ipopt&&o.newton_linear_ordering!="auto")throw std::invalid_argument("--ordering applies to native LDL/MUMPS, not to Ipopt's internal linear solver");
        double worst_residual=0;
        int mumps_ordering=-1;std::uint64_t analyses=0,permutation_hash=0;
        if(!ipopt) {
            const int n=9;
            pf::detail::SymmetricFactor factor(n,o);
            pf::Matrix a(n,n);
            for(int i=0;i<n;++i)for(int j=0;j<n;++j)a(i,j)=i==j?4+.1*i:1.0/(i+j+2);
            pf::Vector rhs(n);for(int i=0;i<n;++i)rhs[i]=(i%2?-1.:1.)*(i+1);
            for(int k=0;k<3;++k) {
                a.diagonal().array()+=k;
                a(0,n-1)=a(n-1,0)=k==0?0:.015;
                if(!factor.factor(a))throw std::runtime_error("SPD matrix rejected");
                auto x=factor.solve(rhs);double residual=(a*x-rhs).norm()/rhs.norm();
                if(!std::isfinite(residual)||residual>1e-11)throw std::runtime_error("Linear residual exceeds 1e-11");
                worst_residual=std::max(worst_residual,residual);
            }
            pf::Matrix indefinite=pf::Matrix::Identity(n,n);indefinite(0,0)=-1;
            if(factor.factor(indefinite))throw std::runtime_error("Indefinite inertia not detected");
            if(factor.factor(pf::Matrix::Zero(n,n)))throw std::runtime_error("Singular matrix accepted");
            if((backend!="ldl"||o.newton_linear_ordering=="metis")&&factor.analyses()!=1)throw std::runtime_error("Symbolic analysis not reused");
            if(!factor.factor(a))throw std::runtime_error("Recovery after singular factorization failed");
            auto x=factor.solve(rhs);
            if((a*x-rhs).norm()>1e-10)throw std::runtime_error("Recovery residual failed");
            analyses=factor.analyses();mumps_ordering=factor.mumps_ordering();
            if(o.newton_linear_ordering=="metis") {
                if(!factor.uses_metis()||factor.permutation().size()!=std::size_t(n))throw std::runtime_error("METIS permutation absent");
                std::vector<bool> seen(n,false);permutation_hash=1469598103934665603ULL;
                for(int p:factor.permutation()) {
                    if(p<0||p>=n||seen[p])throw std::runtime_error("Invalid permutation");
                    seen[p]=true;permutation_hash=(permutation_hash^std::uint64_t(p+1))*1099511628211ULL;
                }
                if(backend=="mumps"&&mumps_ordering!=1)throw std::runtime_error("MUMPS ignored user ordering");
            }
        }
        pf::LennardJones problem(2);
        pf::Bounds bounds{pf::Vector::Constant(6,-5),pf::Vector::Constant(6,5)};
        pf::Vector initial(6);initial<<.3,.2,.2,-.1,-.3,-.1;
        o.validate(problem,bounds);
        for(double radius:{5.,.35}) {
            o.radius=radius;pf::detail::Context c(problem,o,bounds);
            auto result=pf::detail::refine(c,initial,bounds);
            const double distance=std::min(2*radius,std::sqrt(1-1e-8));
            const double q=1/(distance*distance+1e-8),q3=q*q*q,expected=q3*(q3-2);
            if(!c.local_feasible(result.x)||std::abs(result.objective-expected)>1e-6*std::max(1.,std::abs(expected)))
                throw std::runtime_error("Exact-derivative LJ pair minimization failed");
            if(!ipopt&&c.counters.linear_factorizations==0)throw std::runtime_error("Direct backend not used");
            if(o.newton_linear_ordering=="metis"&&c.counters.linear_metis_analyses!=1)throw std::runtime_error("METIS local-path counter missing");
        }
        std::cout<<std::setprecision(17)<<"PASS backend="<<backend<<" ordering="<<o.newton_linear_ordering<<" metis_api="<<pf::metis_api_version()<<" mumps_infog7="<<mumps_ordering<<" analyses="<<analyses<<" permutation_hash="<<permutation_hash<<" worst_relative_residual="<<worst_residual<<" pair_minima=2/2\n";
    }catch(const std::exception& e){std::cerr<<"ERROR: "<<e.what()<<'\n';return 2;}
}
