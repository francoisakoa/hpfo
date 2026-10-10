#include "linear.hpp"
#include "ordering.hpp"
#include <Eigen/Cholesky>
#include <Eigen/SparseCholesky>
#include <algorithm>
#include <climits>
#include <cmath>
#include <dlfcn.h>
#ifdef PF_WITH_MUMPS
#include <dmumps_c.h>
static_assert(std::is_same_v<DMUMPS_REAL,double> && std::is_same_v<DMUMPS_COMPLEX,double>,"DMUMPS must use binary64 double precision");
#endif

namespace pf {
bool mumps_available() {
#ifdef PF_WITH_MUMPS
    return true;
#else
    return false;
#endif
}
}
namespace pf::detail {
bool newton_uses_threads(const Options& o) {
    return o.local_method==LocalMethod::lj_newton&&(o.newton_linear_solver=="cg"||o.newton_linear_solver=="ldl");
}
struct SymmetricFactor::Impl {
    int n,nz;std::string backend;Eigen::LDLT<Matrix> ldl;
    bool metis=false;
    MetisPermutation ordering;
    std::vector<int> mumps_permutation;
    using Sparse=Eigen::SparseMatrix<double,Eigen::ColMajor,int>;
    Sparse permuted;
    Eigen::SimplicialLDLT<Sparse,Eigen::Lower,Eigen::NaturalOrdering<int>> sparse_ldl;
    std::vector<int> row,col;std::vector<double> values;
    bool ready=false;std::uint64_t analyses=0;
    void* hsl=nullptr;
    using Init=void(*)(int*,double*);
    using Analysis=void(*)(int*,int*,const int*,const int*,int*,int*,int*,int*,int*,int*,int*,double*,int*,double*);
    using Factor=void(*)(int*,int*,const int*,const int*,double*,int*,int*,int*,int*,int*,int*,int*,int*,double*,int*);
    using Solve=void(*)(int*,double*,int*,int*,int*,double*,int*,double*,int*,int*,int*,double*);
    Init ma27i=nullptr;Analysis ma27a=nullptr;Factor ma27b=nullptr;Solve ma27c=nullptr;
    int icntl[30]{},info[20]{},liw=0,la=0,nsteps=0,maxfrt=0;double cntl[5]{};
    std::vector<int> iw,ikeep,iw1;std::vector<double> ma_values,work;
#ifdef PF_WITH_MUMPS
    DMUMPS_STRUC_C mumps{};bool mumps_initialized=false;
#endif
    void cleanup()noexcept {
#ifdef PF_WITH_MUMPS
        if(mumps_initialized){mumps.job=-2;dmumps_c(&mumps);mumps_initialized=false;}
#endif
        if(hsl){dlclose(hsl);hsl=nullptr;}
    }
    template<class T>T symbol(const char* name) {
        auto pointer=dlsym(hsl,name);
        if(!pointer)throw std::runtime_error(std::string("HSL library missing ")+name+" (32-bit Fortran INTEGER / double MA27 required)");
        return reinterpret_cast<T>(pointer);
    }
    Impl(int dimension,const Options& options):n(dimension),backend(options.newton_linear_solver) {
        if(n<1||std::int64_t(n)*(n+1)/2>INT_MAX/8)throw std::invalid_argument("Direct Newton dimension exceeds 32-bit workspace capacity");
        nz=n*(n+1)/2;
        if(options.newton_linear_ordering!="auto"&&options.newton_linear_ordering!="metis")throw std::invalid_argument("Unknown Newton ordering");
        metis=options.newton_linear_ordering=="metis";
        if(metis) {
            if(backend!="ldl"&&backend!="mumps")throw std::invalid_argument("METIS requires ldl or mumps");
            ordering=metis_complete_ordering(n);
        }
        if(backend=="ldl") {
            if(metis) {
                std::vector<Eigen::Triplet<double,int>> entries;entries.reserve(nz);
                for(int j=0;j<n;++j)for(int i=j;i<n;++i)entries.emplace_back(i,j,1.0);
                permuted.resize(n,n);permuted.setFromTriplets(entries.begin(),entries.end());
                sparse_ldl.analyzePattern(permuted);++analyses;
            }
            return;
        }
        row.reserve(nz);col.reserve(nz);values.resize(nz);
        for(int i=0;i<n;++i)for(int j=0;j<=i;++j){row.push_back(i+1);col.push_back(j+1);}
        try {
            if(backend=="ma27") {
                std::vector<std::string> names=options.hsl_library.empty()?std::vector<std::string>{"libcoinhsl.so","libhsl.so"}:std::vector<std::string>{options.hsl_library};
                std::string errors;
                for(const auto& name:names) {hsl=dlopen(name.c_str(),RTLD_NOW|RTLD_LOCAL);if(hsl)break;const char* error=dlerror();errors+=name+": "+(error?error:"load failure")+"; ";}
                if(!hsl)throw std::runtime_error("MA27 unavailable. Use --hsl-library /absolute/path/libcoinhsl.so. "+errors);
                ma27i=symbol<Init>("ma27id_");ma27a=symbol<Analysis>("ma27ad_");ma27b=symbol<Factor>("ma27bd_");ma27c=symbol<Solve>("ma27cd_");
                ma27i(icntl,cntl);icntl[0]=0;icntl[1]=0;cntl[0]=1e-8;
                liw=4*nz+6*n+100;iw.resize(liw);ikeep.resize(3*n);iw1.resize(2*n);
                int flag=0;double operations=0;
                ma27a(&n,&nz,row.data(),col.data(),iw.data(),&liw,ikeep.data(),iw1.data(),&nsteps,&flag,icntl,cntl,info,&operations);
                if(info[0]!=0)throw std::runtime_error("MA27 symbolic analysis failed: "+std::to_string(info[0]));
                ++analyses;la=std::max(nz,2*info[4]+100);liw=std::max(liw,2*info[5]+100);
                iw.resize(liw);ma_values.resize(la);iw1.resize(std::max(2*n,nsteps));
            } else if(backend=="mumps") {
#ifdef PF_WITH_MUMPS
                static_assert(sizeof(MUMPS_INT)==sizeof(int),"32-bit MUMPS interface required");
                mumps.sym=2;mumps.par=1;mumps.comm_fortran=-987654;mumps.job=-1;dmumps_c(&mumps);
                if(mumps.infog[0]<0)throw std::runtime_error("MUMPS initialization failed");
                mumps_initialized=true;mumps.n=n;mumps.nz=nz;mumps.irn=row.data();mumps.jcn=col.data();mumps.a=values.data();
                mumps.icntl[0]=-1;mumps.icntl[1]=-1;mumps.icntl[2]=-1;mumps.icntl[3]=0;
                mumps.icntl[9]=0;mumps.icntl[12]=1;mumps.icntl[13]=100;
                if(metis) {
                    mumps_permutation.resize(n);
                    for(int i=0;i<n;++i)mumps_permutation[i]=ordering.inverse[i]+1;
                    mumps.perm_in=mumps_permutation.data();
                    mumps.icntl[6]=1; // externally computed METIS, not automatic MUMPS ordering
                    mumps.icntl[27]=1; // sequential symbolic analysis accepts PERM_IN
                }
                mumps.job=1;dmumps_c(&mumps);
                if(mumps.infog[0]<0)throw std::runtime_error("MUMPS symbolic analysis failed: "+std::to_string(mumps.infog[0]));
                if(metis&&mumps.infog[6]!=1)throw std::runtime_error("MUMPS did not use supplied METIS ordering; INFOG(7)="+std::to_string(mumps.infog[6]));
                ++analyses;
#else
                throw std::runtime_error("Native MUMPS unavailable; rebuild with PF_WITH_MUMPS=ON and sequential MUMPS headers/library");
#endif
            } else throw std::invalid_argument("Direct linear backend must be ldl, ma27 or mumps");
        }catch(...){cleanup();throw;}
    }
    ~Impl(){cleanup();}
    bool factor(const Matrix& matrix) {
        ready=false;
        if(matrix.rows()!=n||matrix.cols()!=n||!matrix.allFinite())return false;
        if(backend=="ldl") {
            if(metis) {
                for(int j=0;j<n;++j)for(Sparse::InnerIterator it(permuted,j);it;++it)
                    it.valueRef()=matrix(ordering.perm[it.row()],ordering.perm[j]);
                sparse_ldl.factorize(permuted);
                ready=sparse_ldl.info()==Eigen::Success&&sparse_ldl.vectorD().allFinite()&&(sparse_ldl.vectorD().array()>0).all();
                return ready;
            }
            ldl.compute(matrix);
            ready=ldl.info()==Eigen::Success&&(ldl.vectorD().array()>0).all();return ready;
        }
        std::size_t k=0;for(int i=0;i<n;++i)for(int j=0;j<=i;++j)values[k++]=matrix(i,j);
        if(backend=="ma27") {
            for(int retry=0;retry<4;++retry) {
                std::copy(values.begin(),values.end(),ma_values.begin());
                ma27b(&n,&nz,row.data(),col.data(),ma_values.data(),&la,iw.data(),&liw,ikeep.data(),&nsteps,&maxfrt,iw1.data(),icntl,cntl,info);
                if(info[0]==-3){liw=std::max(2*liw,2*info[1]);iw.resize(liw);continue;}
                if(info[0]==-4){la=std::max(2*la,2*info[1]);ma_values.resize(la);continue;}
                if(info[0]==-5||info[0]==3)return false;
                if(info[0]!=0)throw std::runtime_error("MA27 factorization failed: "+std::to_string(info[0]));
                ready=info[14]==0;return ready;
            }
            throw std::runtime_error("MA27 workspace retries exhausted");
        }
#ifdef PF_WITH_MUMPS
        for(int retry=0;retry<5;++retry) {
            mumps.job=2;dmumps_c(&mumps);
            if(mumps.infog[0]==-8||mumps.infog[0]==-9){mumps.icntl[13]*=2;continue;}
            if(mumps.infog[0]==-10)return false;
            if(mumps.infog[0]<0)throw std::runtime_error("MUMPS factorization failed: "+std::to_string(mumps.infog[0]));
            ready=mumps.infog[11]==0;return ready;
        }
        throw std::runtime_error("MUMPS workspace retries exhausted");
#else
        return false;
#endif
    }
    Vector solve(const Vector& rhs) {
        if(!ready||rhs.size()!=n)throw std::logic_error("Linear solve requires a positive definite factorization");
        if(backend=="ldl") {
            if(!metis)return ldl.solve(rhs);
            Vector reordered(n);for(int i=0;i<n;++i)reordered[i]=rhs[ordering.perm[i]];
            Vector solved=sparse_ldl.solve(reordered),result(n);
            for(int i=0;i<n;++i)result[ordering.perm[i]]=solved[i];
            return result;
        }
        Vector result=rhs;
        if(backend=="ma27") {
            work.resize(maxfrt);ma27c(&n,ma_values.data(),&la,iw.data(),&liw,work.data(),&maxfrt,result.data(),iw1.data(),&nsteps,icntl,cntl);
        }
#ifdef PF_WITH_MUMPS
        else {mumps.rhs=result.data();mumps.nrhs=1;mumps.lrhs=n;mumps.job=3;dmumps_c(&mumps);mumps.rhs=nullptr;if(mumps.infog[0]<0)throw std::runtime_error("MUMPS triangular solve failed");}
#endif
        return result;
    }
};
SymmetricFactor::SymmetricFactor(int n,const Options& options):impl_(std::make_unique<Impl>(n,options)){}
SymmetricFactor::~SymmetricFactor()=default;
bool SymmetricFactor::factor(const Matrix& matrix){return impl_->factor(matrix);}
Vector SymmetricFactor::solve(const Vector& rhs){return impl_->solve(rhs);}
std::uint64_t SymmetricFactor::analyses()const{return impl_->analyses;}
bool SymmetricFactor::uses_metis()const{return impl_->metis;}
const std::vector<int>& SymmetricFactor::permutation()const{return impl_->ordering.perm;}
int SymmetricFactor::mumps_ordering()const {
#ifdef PF_WITH_MUMPS
    if(impl_->backend=="mumps")return impl_->mumps.infog[6];
#endif
    return -1;
}
}
