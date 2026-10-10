// Algorithms adapted from NumPy 2.3.5; see THIRD_PARTY_NOTICES.md.
#include "pf/pf.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <unordered_set>

#ifndef __SIZEOF_INT128__
#error "NumPy-compatible PCG64 currently requires GCC/Clang unsigned 128-bit arithmetic"
#endif

namespace pf {
namespace {
__extension__ typedef unsigned __int128 U128;
constexpr U128 multiplier=(U128(2549297995355413924ULL)<<64)|4865540595714422341ULL;
U128 join(std::uint64_t high,std::uint64_t low) {return (U128(high)<<64)|low;}
#include "numpy_rng/ziggurat_normal.hpp"

std::array<std::uint64_t,4> seed_sequence(std::uint64_t seed) {
    std::array<std::uint32_t,4> pool{};
    std::uint32_t hash=0x43b0d7e5U;
    auto hashmix=[&](std::uint32_t v) {
        v^=hash;hash*=0x931e8875U;v*=hash;return v^(v>>16);
    };
    for(std::size_t i=0;i<4;++i) pool[i]=hashmix(i==0?std::uint32_t(seed):i==1?std::uint32_t(seed>>32):0);
    for(std::size_t src=0;src<4;++src) for(std::size_t dst=0;dst<4;++dst) if(src!=dst) {
        std::uint32_t v=0xca01f9ddU*pool[dst]-0x4973f715U*hashmix(pool[src]);
        pool[dst]=v^(v>>16);
    }
    hash=0x8b51f9ddU;
    std::array<std::uint64_t,4> result{};
    for(std::size_t i=0;i<8;++i) {
        std::uint32_t v=pool[i%4]^hash;hash*=0x58f38dedU;v*=hash;v^=v>>16;
        result[i/2]|=std::uint64_t(v)<<(32*(i%2));
    }
    return result;
}
}

Random::Random(std::uint64_t seed) {
    const auto words=seed_sequence(seed);
    const U128 increment=(join(words[2],words[3])<<1)|1;
    const U128 state=(increment+join(words[0],words[1]))*multiplier+increment;
    state_low_=std::uint64_t(state);state_high_=std::uint64_t(state>>64);
    inc_low_=std::uint64_t(increment);inc_high_=std::uint64_t(increment>>64);
}

Random Random::sklearn(std::uint32_t seed) {
    Random rng(seed);rng.legacy_=true;rng.legacy_engine_.seed(seed);return rng;
}

std::uint64_t Random::raw() {
    if(legacy_) throw std::logic_error("raw() requires PCG64");
    const U128 state=join(state_high_,state_low_)*multiplier+join(inc_high_,inc_low_);
    state_low_=std::uint64_t(state);state_high_=std::uint64_t(state>>64);
    const std::uint64_t x=state_low_^state_high_;
    const unsigned rotation=unsigned(state>>122);
    return (x>>rotation)|(x<<((0U-rotation)&63));
}

std::uint32_t Random::next32() {
    if(legacy_) return legacy_engine_();
    if(has_buffered_) {has_buffered_=false;return buffered_;}
    const auto value=raw();buffered_=std::uint32_t(value>>32);has_buffered_=true;
    return std::uint32_t(value);
}

double Random::uniform() {
    if(legacy_) {
        const auto a=legacy_engine_()>>5,b=legacy_engine_()>>6;
        return (double(a)*67108864.0+double(b))*0x1.0p-53;
    }
    return double(raw()>>11)*0x1.0p-53;
}

double Random::normal() {
    if(legacy_) throw std::logic_error("normal() requires PCG64; legacy mode is for clustering only");
    for(;;) {
        auto r=raw();const auto idx=r&0xff;r>>=8;
        const std::uint64_t sign=r&1,rabs=(r>>1)&0x000fffffffffffffULL;
        double x=double(rabs)*wi_double[idx];if(sign)x=-x;
        if(rabs<ki_double[idx]) return x;
        if(idx==0) {
            for(;;) {
                const double xx=-ziggurat_nor_inv_r*std::log1p(-uniform());
                const double yy=-std::log1p(-uniform());
                if(yy+yy>xx*xx) return ((rabs>>8)&1)?-(ziggurat_nor_r+xx):ziggurat_nor_r+xx;
            }
        }
        else if((fi_double[idx-1]-fi_double[idx])*uniform()+fi_double[idx]<std::exp(-0.5*x*x)) return x;
    }
}

std::size_t Random::index(std::size_t size) {
    if(!size) throw std::invalid_argument("Cannot sample from an empty set");
    if(size==1) return 0; // NumPy does not consume a draw for a degenerate range.
    const std::uint64_t bound=size;
    if(legacy_) {
        if(bound-1>UINT32_MAX) throw std::invalid_argument("Legacy clustering index exceeds uint32");
        std::uint32_t mask=std::uint32_t(bound-1);
        mask|=mask>>1;mask|=mask>>2;mask|=mask>>4;mask|=mask>>8;mask|=mask>>16;
        for(;;) {auto value=next32()&mask;if(value<bound)return value;}
    }
    if(bound-1<=UINT32_MAX) {
        if(bound-1==UINT32_MAX) return next32();
        const auto b=std::uint32_t(bound),threshold=std::uint32_t(0U-b)%b;
        for(;;) {
            const std::uint64_t product=std::uint64_t(next32())*b;
            if(std::uint32_t(product)>=threshold) return std::size_t(product>>32);
        }
    }
    const auto threshold=(std::uint64_t(0)-bound)%bound;
    for(;;) {
        const U128 product=U128(raw())*bound;
        if(std::uint64_t(product)>=threshold) return std::size_t(product>>64);
    }
}

std::vector<std::size_t> Random::choice(std::size_t size,std::size_t count,bool replacement) {
    if(legacy_) throw std::logic_error("choice() requires PCG64");
    if((!size&&count)||(!replacement&&count>size)) throw std::invalid_argument("Invalid choice size");
    std::vector<std::size_t> result;result.reserve(count);
    if(replacement) {for(std::size_t i=0;i<count;++i)result.push_back(index(size));return result;}
    // Generator.choice(..., replace=False, shuffle=True), including full selections.
    if(size>10000&&count>size/50) {
        std::vector<std::size_t> order(size);std::iota(order.begin(),order.end(),0);
        const auto first=std::max(size-count,std::size_t(1));
        for(std::size_t i=size;i>first;) {--i;std::swap(order[i],order[index(i+1)]);}
        return {order.end()-count,order.end()};
    }
    std::unordered_set<std::size_t> selected;selected.reserve(count);
    for(std::size_t j=size-count;j<size;++j) {
        const auto value=index(j+1);
        const auto chosen=selected.count(value)?j:value;
        selected.insert(chosen);result.push_back(chosen);
    }
    for(std::size_t i=count;i>1;) {--i;std::swap(result[i],result[index(i+1)]);}
    return result;
}

void Random::advance(std::uint64_t count) {
    if(legacy_) throw std::logic_error("advance() requires PCG64");
    U128 acc_mult=1,acc_plus=0,current_mult=multiplier,current_plus=join(inc_high_,inc_low_);
    while(count) {
        if(count&1) {acc_mult*=current_mult;acc_plus=acc_plus*current_mult+current_plus;}
        current_plus*=(current_mult+1);current_mult*=current_mult;count>>=1;
    }
    const U128 state=acc_mult*join(state_high_,state_low_)+acc_plus;
    state_low_=std::uint64_t(state);state_high_=std::uint64_t(state>>64);
}
} // namespace pf
