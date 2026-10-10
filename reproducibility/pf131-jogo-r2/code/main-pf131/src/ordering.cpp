#include "ordering.hpp"
#include <cstdint>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#ifdef PF_WITH_METIS
extern "C" {
#include <metis.h>
}
#endif

namespace pf {
bool metis_available() {
#ifdef PF_WITH_METIS
    return true;
#else
    return false;
#endif
}
int metis_api_version() {
#ifdef PF_WITH_METIS
#ifdef METIS_VER_MAJOR
    return METIS_VER_MAJOR;
#else
    return 4;
#endif
#else
    return 0;
#endif
}
}
namespace pf::detail {
MetisPermutation metis_complete_ordering(int dimension) {
#ifndef PF_WITH_METIS
    (void)dimension;
    throw std::runtime_error("METIS unavailable; rebuild with PF_WITH_METIS=ON");
#else
    // METIS 4 has global RNG state. Serialize ordering only; all numerical LDL
    // factorizations remain independent and parallel. Cache this fixed graph.
    static std::mutex mutex;
    static std::map<int,MetisPermutation> cache;
    std::lock_guard<std::mutex> lock(mutex);
    auto previous=cache.find(dimension);
    if(previous!=cache.end())return previous->second;
#ifdef METIS_VER_MAJOR
    using Index=idx_t;
#else
    using Index=idxtype;
#endif
    const std::int64_t edges=std::int64_t(dimension)*(dimension-1);
    if(dimension<1||edges>std::numeric_limits<Index>::max())throw std::invalid_argument("METIS graph exceeds index capacity");
    std::vector<Index> ptr(dimension+1),adj;
    adj.reserve(static_cast<std::size_t>(edges));
    for(int i=0;i<dimension;++i) {
        ptr[i]=static_cast<Index>(adj.size());
        for(int j=0;j<dimension;++j)if(i!=j)adj.push_back(j);
    }
    ptr[dimension]=static_cast<Index>(adj.size());
    std::vector<Index> perm(dimension),inverse(dimension);
    if(dimension==1) {perm[0]=0;inverse[0]=0;}
    else {
#ifdef METIS_VER_MAJOR
        Index n=dimension;
        Index options[METIS_NOPTIONS];
        if(METIS_SetDefaultOptions(options)!=METIS_OK)throw std::runtime_error("METIS default options failed");
        options[METIS_OPTION_NUMBERING]=0;options[METIS_OPTION_SEED]=0;
        const int status=METIS_NodeND(&n,ptr.data(),adj.data(),nullptr,options,perm.data(),inverse.data());
        if(status!=METIS_OK)throw std::runtime_error("METIS_NodeND failed: "+std::to_string(status));
#else
        static_assert(sizeof(idxtype)==sizeof(int),"METIS 4 INTEGER32 required");
        int count=dimension,numbering=0,options[8]{};
        METIS_NodeND(&count,ptr.data(),adj.data(),&numbering,options,perm.data(),inverse.data());
#endif
    }
    MetisPermutation result;
    result.perm.assign(perm.begin(),perm.end());result.inverse.assign(inverse.begin(),inverse.end());
    std::vector<bool> seen(dimension,false);
    for(int i=0;i<dimension;++i) {
        const int old=result.perm[i];
        if(old<0||old>=dimension||seen[old]||result.inverse[old]!=i)throw std::runtime_error("Invalid METIS permutation");
        seen[old]=true;
    }
    cache.emplace(dimension,result);
    return result;
#endif
}
}
