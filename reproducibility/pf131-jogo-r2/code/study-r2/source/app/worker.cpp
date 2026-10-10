#include "internal.hpp"
#include <iostream>
#ifdef __linux__
#include <sys/prctl.h>
#include <signal.h>
#include <unistd.h>
#endif
int main(int argc,char** argv) {
    try {
        if(argc!=2||std::string(argv[1])!="--fd=3") {std::cerr<<"Internal PF worker; launch pf_solver instead.\n";return 2;}
#ifdef __linux__
        const auto parent=getppid();
        if(prctl(PR_SET_PDEATHSIG,SIGTERM)!=0||parent==1||getppid()!=parent) return 2;
#endif
        return pf::detail::local_worker_main(3);
    } catch(const std::exception& e) {std::cerr<<"PF worker: "<<e.what()<<'\n';return 2;}
}
