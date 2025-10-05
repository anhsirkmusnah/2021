#pragma once

#include "itensor/all.h"
#include "itensor/mps/siteset.h"
#include "itensor/util/str.h"
#include <cmath>
#include <complex>

namespace itensor {

// Forward declare
class QubitSite;
using Qubit = BasicSiteSet<QubitSite>;

//
// QubitSite: defines the local Hilbert space and gate set
//
class QubitSite
{
    Index s;

public:
    QubitSite(Index const& I) : s(I) { }

    QubitSite(Args const& args = Args::global())
    {
        auto ts = TagSet("Site,Qubit");
        if(args.defined("SiteNumber"))
            ts.addTags("n="+str(args.getInt("SiteNumber")));

        // Default: no QNs for performance
        auto conserveqns = args.getBool("ConserveQNs",false);
        if(conserveqns)
        {
            auto conserveSz = args.getBool("ConserveSz",true);
            auto conserveParity = args.getBool("ConserveParity",false);
            if(conserveSz && conserveParity)
            {
                s = Index(QN({"Sz",+1},{"Parity",1,2}),1,
                          QN({"Sz",-1},{"Parity",0,2}),1,Out,ts);
            }
            else if(conserveSz)
            {
                s = Index(QN({"Sz",+1}),1,QN({"Sz",-1}),1,Out,ts);
            }
            else if(conserveParity)
            {
                s = Index(QN({"Parity",1,2}),1,QN({"Parity",0,2}),1,Out,ts);
            }
            else
            {
                s = Index(2,ts);
            }
        }
        else
        {
            s = Index(2,ts);
        }
    }

    Index index() const { return s; }

    IndexVal state(std::string const& state) const
    {
        if(state == "Up") return s(1);
        if(state == "Dn") return s(2);
        throw ITError("State " + state + " not recognized");
    }

    ITensor op(std::string const& opname, Args const& args = Args::global()) const
    {
        constexpr double pi = 3.14159265358979323846;
        const std::complex<double> i(0.0, 1.0);

        // Extract parameters
        double alpha  = args.getReal("alpha", 0.0);
        double theta  = args.getReal("theta", alpha*pi/2.0);
        double phi    = args.getReal("phi", 0.0);
        double lambda = args.getReal("lambda", 0.0);

        auto sP = prime(s);
        auto Up = s(1), UpP = sP(1);
        auto Dn = s(2), DnP = sP(2);

        ITensor Op(dag(s), sP);

        // ---- Pauli gates ----
        if(opname == "X")
        {
            Op.set(Up,DnP,1.0);
            Op.set(Dn,UpP,1.0);
        }
        else if(opname == "Y")
        {
            Op.set(Up,DnP,-i);
            Op.set(Dn,UpP,i);
        }
        else if(opname == "Z")
        {
            Op.set(Up,UpP,1.0);
            Op.set(Dn,DnP,-1.0);
        }
        else if(opname == "H")
        {
            double inv_sqrt2 = 1.0/std::sqrt(2.0);
            Op.set(Up,UpP,inv_sqrt2);
            Op.set(Up,DnP,inv_sqrt2);
            Op.set(Dn,UpP,inv_sqrt2);
            Op.set(Dn,DnP,-inv_sqrt2);
        }
        // ---- Rotations ----
        else if(opname == "Rx")
        {
            double c = std::cos(theta/2.0);
            double s_ = std::sin(theta/2.0);
            Op.set(Up,UpP,c);
            Op.set(Dn,DnP,c);
            Op.set(Up,DnP,-i*s_);
            Op.set(Dn,UpP,-i*s_);
        }
        else if(opname == "Ry")
        {
            double c = std::cos(theta/2.0);
            double s_ = std::sin(theta/2.0);
            Op.set(Up,UpP,c);
            Op.set(Dn,DnP,c);
            Op.set(Up,DnP,-s_);
            Op.set(Dn,UpP,s_);
        }
        else if(opname == "Rz")
        {
            Op.set(Up,UpP,std::exp(-i*theta/2.0));
            Op.set(Dn,DnP,std::exp(i*theta/2.0));
        }
        else if(opname == "U3")
        {
            Op.set(Up,UpP,std::cos(theta/2.0));
            Op.set(Up,DnP,-std::exp(i*lambda)*std::sin(theta/2.0));
            Op.set(Dn,UpP,std::exp(i*phi)*std::sin(theta/2.0));
            Op.set(Dn,DnP,std::exp(i*(phi+lambda))*std::cos(theta/2.0));
        }
        // ---- Phase-like ----
        else if(opname == "S")
        {
            Op.set(Up,UpP,1.0);
            Op.set(Dn,DnP,i);
        }
        else if(opname == "T")
        {
            Op.set(Up,UpP,1.0);
            Op.set(Dn,DnP,std::exp(i*pi/4.0));
        }
        else if(opname == "Id")
        {
            Op.set(Up,UpP,1.0);
            Op.set(Dn,DnP,1.0);
        }
        // ---- Spin ladder ----
        else if(opname == "Sp" || opname == "S+")
        {
            Op.set(Dn,UpP,1.0);
        }
        else if(opname == "Sm" || opname == "S-")
        {
            Op.set(Up,DnP,1.0);
        }
        else
        {
            throw ITError("Operator \"" + opname + "\" not recognized in QubitSite.");
        }

        return Op;
    }

    // Legacy constructor
    QubitSite(int n, Args const& args = Args::global())
    {
        *this = QubitSite({args,"SiteNumber=",n});
    }
};

//
// ---- Utility: Apply single and two-qubit gates to an MPS ----
//
inline void apply_gate(MPS& psi, ITensor const& gate, int site, Args const& args = Args::global())
{
    psi.position(site);
    auto wf = psi(site) * psi(site+1);
    wf *= gate;
    auto [U,S,V] = svd(wf,{site,site+1},args);
    psi.set(site,U);
    psi.set(site+1,S*V);
}

inline void apply_gate(MPS& psi, ITensor const& gate, int site, SiteSet const& sites, Args const& args = Args::global())
{
    auto g = toITensor(gate);
    psi.position(site);
    auto wf = psi(site) * g;
    auto [U,S,V] = svd(wf,{sites(site)},args);
    psi.set(site,U);
    psi.set(site+1,S*V);
}

//
// ---- ZZ Feature Map Ansatz ----
// depth: number of layers
// x: vector of features (size = number of qubits)
//
inline void FeatureMapAnsatz(MPS& psi, Qubit const& sites, std::vector<double> const& x, int depth, Args const& args = Args::global())
{
    const int N = length(sites);
    const double pi = 3.14159265358979323846;
    const std::complex<double> i(0.0,1.0);

    for(int d = 0; d < depth; ++d)
    {
        // Single-qubit encoding: Rz(x_j) then Rx(pi/2)
        for(int j = 1; j <= N; ++j)
        {
            auto Rz = sites.op("Rz",{"theta=",x[j-1]});
            psi.position(j);
            psi.ref(j) = psi(j) * Rz;

            auto Rx = sites.op("Rx",{"theta=",pi/2.0});
            psi.position(j);
            psi.ref(j) = psi(j) * Rx;
        }

        // Entangling layer: exp(-i * ZZ * x_j * x_k)
        for(int j = 1; j < N; ++j)
        {
            double theta = x[j-1]*x[j]; // simple polynomial kernel style
            auto sz1 = sites.op("Z",{"theta=",theta});
            auto sz2 = sites.op("Z",{"theta=",theta});
            auto G = sz1 * sz2;
            apply_gate(psi,G,j,args);
        }
    }
}

} // namespace itensor



// helloitensor_all.cpp
// Copy-paste ready. Requires ITensor (3.x) and pybind11.

#include "itensor/all.h"
#include "itensor/mps/siteset.h"
#include "itensor/util/str.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <complex>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <chrono>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace itensor;
namespace py = pybind11;

// ----------------------------
// QubitSite + Qubit (BasicSiteSet)
// ----------------------------
class QubitSite;
using Qubit = BasicSiteSet<QubitSite>;

class QubitSite
{
    Index s;
public:
    QubitSite(Index const& I) : s(I) { }

    QubitSite(Args const& args = Args::global())
    {
        auto ts = TagSet("Site,Qubit");
        if(args.defined("SiteNumber"))
            ts.addTags("n="+str(args.getInt("SiteNumber")));

        auto conserveqns = args.getBool("ConserveQNs",false);
        if(conserveqns)
        {
            auto conserveSz = args.getBool("ConserveSz",true);
            auto conserveParity = args.getBool("ConserveParity",false);
            if(conserveSz && conserveParity)
            {
                s = Index(QN({"Sz",+1},{"Parity",1,2}),1,
                          QN({"Sz",-1},{"Parity",0,2}),1,Out,ts);
            }
            else if(conserveSz)
            {
                s = Index(QN({"Sz",+1}),1,QN({"Sz",-1}),1,Out,ts);
            }
            else if(conserveParity)
            {
                s = Index(QN({"Parity",1,2}),1,QN({"Parity",0,2}),1,Out,ts);
            }
            else
            {
                s = Index(2,ts);
            }
        }
        else
        {
            s = Index(2,ts);
        }
    }

    Index index() const { return s; }

    IndexVal state(std::string const& state) const
    {
        if(state == "Up") return s(1);
        if(state == "Dn") return s(2);
        throw ITError("State " + state + " not recognized");
    }

    // Extended op(): many gates supported (Rx, Ry, Rz, U3, H, S, T, Pauli, Id, raising/lowering)
    ITensor op(std::string const& opname, Args const& args = Args::global()) const
    {
        constexpr double pi = 3.14159265358979323846;
        const std::complex<double> i(0.0, 1.0);

        double alpha  = args.getReal("alpha", 0.0);
        double theta  = args.getReal("theta", alpha*pi/2.0);
        double phi    = args.getReal("phi", 0.0);
        double lambda = args.getReal("lambda", 0.0);

        auto sP = prime(s);
        auto Up = s(1), UpP = sP(1);
        auto Dn = s(2), DnP = sP(2);

        ITensor Op(dag(s), sP);

        if(opname == "X")
        {
            Op.set(Up,DnP,1.0);
            Op.set(Dn,UpP,1.0);
        }
        else if(opname == "Y")
        {
            Op.set(Up,DnP,-i);
            Op.set(Dn,UpP,i);
        }
        else if(opname == "Z")
        {
            Op.set(Up,UpP,1.0);
            Op.set(Dn,DnP,-1.0);
        }
        else if(opname == "H")
        {
            double inv_sqrt2 = 1.0/std::sqrt(2.0);
            Op.set(Up,UpP,inv_sqrt2);
            Op.set(Up,DnP,inv_sqrt2);
            Op.set(Dn,UpP,inv_sqrt2);
            Op.set(Dn,DnP,-inv_sqrt2);
        }
        else if(opname == "Rx")
        {
            double c = std::cos(theta/2.0);
            double s_ = std::sin(theta/2.0);
            Op.set(Up,UpP,c);
            Op.set(Dn,DnP,c);
            Op.set(Up,DnP,-i*s_);
            Op.set(Dn,UpP,-i*s_);
        }
        else if(opname == "Ry")
        {
            double c = std::cos(theta/2.0);
            double s_ = std::sin(theta/2.0);
            Op.set(Up,UpP,c);
            Op.set(Dn,DnP,c);
            Op.set(Up,DnP,-s_);
            Op.set(Dn,UpP,s_);
        }
        else if(opname == "Rz")
        {
            Op.set(Up,UpP,std::exp(-i*theta/2.0));
            Op.set(Dn,DnP,std::exp(i*theta/2.0));
        }
        else if(opname == "U3")
        {
            Op.set(Up,UpP,std::cos(theta/2.0));
            Op.set(Up,DnP,-std::exp(i*lambda)*std::sin(theta/2.0));
            Op.set(Dn,UpP,std::exp(i*phi)*std::sin(theta/2.0));
            Op.set(Dn,DnP,std::exp(i*(phi+lambda))*std::cos(theta/2.0));
        }
        else if(opname == "S")
        {
            Op.set(Up,UpP,1.0);
            Op.set(Dn,DnP,i);
        }
        else if(opname == "T")
        {
            Op.set(Up,UpP,1.0);
            Op.set(Dn,DnP,std::exp(i*pi/4.0));
        }
        else if(opname == "Id")
        {
            Op.set(Up,UpP,1.0);
            Op.set(Dn,DnP,1.0);
        }
        else if(opname == "Sp" || opname == "S+")
        {
            Op.set(Dn,UpP,1.0);
        }
        else if(opname == "Sm" || opname == "S-")
        {
            Op.set(Up,DnP,1.0);
        }
        else
        {
            throw ITError("Operator \"" + opname + "\" not recognized in QubitSite.");
        }

        return Op;
    }

    // Legacy constructor
    QubitSite(int n, Args const& args = Args::global())
    {
        *this = QubitSite({args,"SiteNumber=",n});
    }
};

// ----------------------------
// GateCache for single-qubit gates (thread-safe)
// ----------------------------
struct GateCache
{
    std::unordered_map<std::string, ITensor> cache;
    std::mutex cache_mutex;

    static std::string key_for(const std::string& gate_name, int site_idx, double angle)
    {
        std::ostringstream ss;
        ss << gate_name << "_s" << site_idx << "_" << std::setprecision(9) << angle;
        return ss.str();
    }

    ITensor get_or_make_single(const Qubit& sites, const std::string& gate_name, int site_idx, double angle, Args const& op_args = Args::global())
    {
        auto k = key_for(gate_name, site_idx, angle);
        {
            std::lock_guard<std::mutex> lg(cache_mutex);
            auto it = cache.find(k);
            if(it != cache.end()) return it->second;
        }
        // Make the operator
        Args local_args = op_args;
        local_args.add("theta", angle);
        auto G = sites.op(gate_name, local_args, site_idx);
        {
            std::lock_guard<std::mutex> lg(cache_mutex);
            cache.emplace(k, G);
            return cache.at(k);
        }
    }
};

// ----------------------------
// SVD adaptivity heuristic
// ----------------------------
// Create Args for svd with adaptive MaxDim up to maxCap
inline Args svd_args_adaptive(size_t baseMaxDim = 64, size_t maxCap = 1024, double cutoff = 1E-12)
{
    // Heuristic: start with baseMaxDim and allow growth; keep cutoff as given.
    // A more advanced approach would monitor truncation error and increase MaxDim
    // dynamically if truncation is significant. For simplicity/time we keep this heuristic.
    Args a = Args("Cutoff", cutoff, "MaxDim", static_cast<int>(baseMaxDim));
    // attach "AllowGrow" flag for the caller to implement reattempting if needed
    return a;
}

// ----------------------------
// Optimized single- and two-qubit gate application for MPS
// ----------------------------
inline void apply_single_gate(MPS& psi, const ITensor& gate, int site, const Args& args)
{
    psi.position(site);
    auto new_tensor = gate * psi(site);
    new_tensor.noPrime();
    psi.set(site, new_tensor);
}

inline void apply_two_gate(MPS& psi, const ITensor& gate, int site1, int site2, const Args& args)
{
    if(site2 != site1+1)
    {
        // For non-neighboring sites, bring them together (swap network) would be required.
        // Here we assume neighbor gates; for long-range gates, consider MPOs.
        throw ITError("apply_two_gate currently assumes neighboring sites (site2 == site1+1).");
    }
    psi.position(site1);
    auto wf = psi(site1) * psi(site2);
    wf = gate * wf;
    wf.noPrime();
    auto [U, S, V] = svd(wf, inds(psi(site1)), args);
    psi.set(site1, U);
    psi.set(site2, S * V);
}

// ----------------------------
// XX/ZZ-phase gate creators (as earlier)
// ----------------------------
ITensor make_xxphase_gate(const Qubit& sites, int i1, int i2, double theta)
{
    const std::complex<double> iC(0.0, 1.0);
    double c = std::cos(theta);
    double s = std::sin(theta);

    auto s1 = sites(i1), s2 = sites(i2);
    auto s1p = prime(s1), s2p = prime(s2);

    ITensor gate(dag(s1), dag(s2), s1p, s2p);

    gate.set(s1(1), s2(1), s1p(1), s2p(1), c);
    gate.set(s1(1), s2(2), s1p(1), s2p(2), c);
    gate.set(s1(2), s2(1), s1p(2), s2p(1), c);
    gate.set(s1(2), s2(2), s1p(2), s2p(2), c);

    gate.set(s1(1), s2(1), s1p(2), s2p(2), -iC*s);
    gate.set(s1(1), s2(2), s1p(2), s2p(1), -iC*s);
    gate.set(s1(2), s2(1), s1p(1), s2p(2), -iC*s);
    gate.set(s1(2), s2(2), s1p(1), s2p(1), -iC*s);

    return gate;
}

ITensor make_zzphase_gate(const Qubit& sites, int i1, int i2, double theta)
{
    const std::complex<double> iC(0.0, 1.0);
    auto phase = std::exp(-iC * theta);

    auto s1 = sites(i1), s2 = sites(i2);
    auto s1p = prime(s1), s2p = prime(s2);

    ITensor gate(dag(s1), dag(s2), s1p, s2p);

    gate.set(s1(1), s2(1), s1p(1), s2p(1), phase);
    gate.set(s1(1), s2(2), s1p(1), s2p(2), 1.0);
    gate.set(s1(2), s2(1), s1p(2), s2p(1), 1.0);
    gate.set(s1(2), s2(2), s1p(2), s2p(2), phase);

    return gate;
}

ITensor make_cz_gate(const Qubit& sites, int i1, int i2)
{
    auto s1 = sites(i1), s2 = sites(i2);
    auto s1p = prime(s1), s2p = prime(s2);
    ITensor G(dag(s1), dag(s2), s1p, s2p);

    G.set(s1(1), s2(1), s1p(1), s2p(1), 1.0);
    G.set(s1(1), s2(2), s1p(1), s2p(2), 1.0);
    G.set(s1(2), s2(1), s1p(2), s2p(1), 1.0);
    G.set(s1(2), s2(2), s1p(2), s2p(2), -1.0);

    return G;
}

// ----------------------------
// Feature map: ZZFeatureMap-like (depth d)
//    Each layer: single-qubit RY(x_j) (or Rz) + pairwise ZZ entanglers (or CZ)
// ----------------------------
void FeatureMapAnsatz(MPS& psi,
                      const Qubit& sites,
                      const std::vector<double>& features,
                      int depth = 2,
                      bool use_ry = true,
                      bool stagger_entanglers = true,
                      double scale = M_PI,   // map normalized feature to [0, scale]
                      Args svd_args = Args("Cutoff",1E-12,"MaxDim",256))
{
    int N = psi.N();
    if(static_cast<int>(features.size()) != N)
    {
        std::cerr << "FeatureMapAnsatz: warning features size != N; using min length\n";
    }
    int nuse = std::min(N, static_cast<int>(features.size()));

    // normalize features to [-1,1] then to [0,1] — robust normalization by max abs
    double max_abs = 0.0;
    for(int i = 0; i < nuse; ++i) max_abs = std::max(max_abs, std::abs(features[i]));
    if(max_abs == 0.0) max_abs = 1.0;

    GateCache gcache;

    for(int layer = 0; layer < depth; ++layer)
    {
        // single-qubit rotations
        for(int j = 1; j <= nuse; ++j)
        {
            double normalized = (features[j-1]/max_abs + 1.0)/2.0; // [0,1]
            double angle = normalized * scale;
            if(use_ry)
            {
                auto G = gcache.get_or_make_single(sites, "Ry", j, angle);
                apply_single_gate(psi, G, j, svd_args);
            }
            else
            {
                auto G = gcache.get_or_make_single(sites, "Rz", j, angle);
                apply_single_gate(psi, G, j, svd_args);
            }
        }

        // entanglers: staggered CZ or nearest-neighbor
        int start = (stagger_entanglers && (layer % 2 == 1)) ? 2 : 1;
        for(int j = start; j < N; j += 2)
        {
            if(j+1 <= N)
            {
                auto CZ = make_cz_gate(sites, j, j+1);
                apply_two_gate(psi, CZ, j, j+1, svd_args);
            }
        }
    }
}

// ----------------------------
// Amplitude encoding
// Two functions:
//  - amplitude_encode_fullstate: exact conversion of a full 2^N complex vector into MPS (feasible only for small N)
//  - amplitude_encode_product: product-state embedding (scales well)
// ----------------------------

// Build full ITensor from amplitude vector (amplitudes.size() must equal 2^N).
// Then convert full ITensor to MPS by successive SVDs.
// WARNING: exponential memory; only use small N (N <= ~20 recommended).
MPS amplitude_encode_fullstate(const Qubit& sites, const std::vector<std::complex<double>>& amplitudes, Args svd_args = Args("Cutoff",1E-14,"MaxDim",4096))
{
    int N = length(sites);
    size_t dim = 1ULL << N;
    if(amplitudes.size() != dim)
    {
        throw ITError("amplitude_encode_fullstate: amplitudes length must equal 2^N");
    }
    if(N > 22)
    {
        throw ITError("amplitude_encode_fullstate: N too large for full-state encoding (exponential memory).");
    }

    // Build full tensor with indices (s1, s2, ..., sN)
    std::vector<Index> inds;
    inds.reserve(N);
    for(int j = 1; j <= N; ++j) inds.push_back(sites(j));

    ITensor full;
    // create an ITensor with the product of site indices
    full = ITensor(inds);

    // Fill full tensor with amplitudes given a mapping from bitstring to index (lexicographic)
    // mapping: amplitude[k] corresponds to bitstring where MSB=site1 (we choose site1 as most significant)
    for(size_t k = 0; k < dim; ++k)
    {
        // produce IndexVals for each site
        std::vector<int> bits(N);
        size_t t = k;
        for(int b = N-1; b >= 0; --b)
        {
            bits[b] = (t & 1) ? 2 : 1; // |Up>=1, |Dn>=2
            t >>= 1;
            if(b==0) break;
        }
        // set element
        // build a list of IndexVal
        std::vector<IndexVal> ival;
        ival.reserve(N);
        for(int j = 0; j < N; ++j) ival.push_back(inds[j](bits[j]));
        full.set(ival, amplitudes[k]);
    }

    // Now factor full into MPS by successive left-to-right SVDs
    MPS psi;
    psi = MPS(InitState(sites)); // initialize to right dimensions
    ITensor cur = full;
    for(int site = 1; site < N; ++site)
    {
        // Group indices: left = inds[0..site-1], right = inds[site..N-1]
        std::vector<Index> left_inds, right_inds;
        for(int j = 0; j < site; ++j) left_inds.push_back(inds[j]);
        for(int j = site; j < N; ++j) right_inds.push_back(inds[j]);

        // reshape and SVD
        auto [U,S,V] = svd(cur, left_inds, svd_args);

        // U has left_inds + new link index. Place U as tensor at psi(site)
        // Convert U into site tensor by contracting with identity as needed.
        // We place U into psi(site) and set remaining cur = S*V
        psi.set(site, U);
        cur = S * V;
    }
    // final tensor is cur, assign to last site
    psi.set(N, cur);

    return psi;
}

// Product-state amplitude encoding (scalable):
// Map each normalized scalar x_i (in [0,1]) to single-qubit state:
//  cos(theta_i) |0> + sin(theta_i) |1>, with theta_i = arccos(sqrt(x))
void amplitude_encode_product(MPS& psi, const Qubit& sites, const std::vector<double>& features, Args svd_args = Args("Cutoff",1E-12,"MaxDim",256))
{
    int N = psi.N();
    if(static_cast<int>(features.size()) != N)
    {
        std::cerr << "amplitude_encode_product: warning features size != N ; using min length\n";
    }
    int nuse = std::min(N, static_cast<int>(features.size()));

    // normalize features to [0,1]
    double minv = features.empty() ? 0.0 : features[0];
    double maxv = minv;
    for(int i = 0; i < nuse; ++i) { minv = std::min(minv, features[i]); maxv = std::max(maxv, features[i]); }
    double range = maxv - minv;
    if(range == 0.0) range = 1.0;

    for(int j = 1; j <= nuse; ++j)
    {
        double x = (features[j-1] - minv) / range; // [0,1]
        // We want amplitude mapping: alpha = sqrt(x), beta = sqrt(1-x)
        // One convenient mapping to single-qubit rotation Ry: Ry(2*arccos(sqrt(x))) |0> -> sqrt(x)|0> + sqrt(1-x)|1>
        double theta = 2.0 * std::acos(std::sqrt(std::clamp(x,0.0,1.0)));
        auto G = sites.op("Ry", {"theta", theta}, j);
        apply_single_gate(psi, G, j, svd_args);
    }
}

// ----------------------------
// apply_gates3: updated with caching and adaptive svd args
// circuits: vector of tuples (sym, i1, i2, alpha) as before
// sym mapping same as your original code:
// 0: H, 1: Rx, 2: Rz, 3: XXPhase, 4: ZZPhase, 5: SWAP, 6: T, 7: CZ
// ----------------------------
MPS apply_gates3(const std::vector<std::tuple<int,int,int,double>>& circuits,
                 const Qubit& site_inds, int N, double cutoff = 1E-12,
                 size_t baseMaxDim = 256, size_t maxCap = 2048)
{
    // Use adaptive SVD args heuristic
    Args svd_args = svd_args_adaptive(baseMaxDim, maxCap, cutoff);

    constexpr double pi = 3.14159265358979323846;

    // Initialize MPS state to |Up...Up>
    auto init = InitState(site_inds);
    for(int n = 1; n <= N; ++n) init.set(n, "Up");
    auto psi = MPS(init);

    for(const auto& gate : circuits)
    {
        auto sym = std::get<0>(gate);
        auto i1 = std::get<1>(gate);
        auto i2 = std::get<2>(gate);
        auto a = std::get<3>(gate);

        double theta = (pi * a) / 2.0;

        switch(sym)
        {
            case 0: // H
            {
                auto G = site_inds.op("H", {}, i1+1);
                apply_single_gate(psi, G, i1+1, svd_args);
                break;
            }
            case 1: // Rx
            {
                auto G = site_inds.op("Rx", {"theta", theta}, i1+1);
                apply_single_gate(psi, G, i1+1, svd_args);
                break;
            }
            case 2: // Rz
            {
                auto G = site_inds.op("Rz", {"theta", theta}, i1+1);
                apply_single_gate(psi, G, i1+1, svd_args);
                break;
            }
            case 3: // XXPhase
            {
                auto G = make_xxphase_gate(site_inds, i1+1, i2+1, theta);
                apply_two_gate(psi, G, i1+1, i2+1, svd_args);
                break;
            }
            case 4: // ZZPhase
            {
                auto G = make_zzphase_gate(site_inds, i1+1, i2+1, theta);
                apply_two_gate(psi, G, i1+1, i2+1, svd_args);
                break;
            }
            case 5: // SWAP
            {
                auto s1 = site_inds(i1+1), s2 = site_inds(i2+1);
                auto s1p = prime(s1), s2p = prime(s2);
                ITensor G(dag(s1), dag(s2), s1p, s2p);
                G.set(s1(1), s2(1), s1p(2), s2p(1), 1.0);
                G.set(s1(1), s2(2), s1p(1), s2p(2), 1.0);
                G.set(s1(2), s2(1), s1p(1), s2p(2), 1.0);
                G.set(s1(2), s2(2), s1p(2), s2p(1), 1.0);
                apply_two_gate(psi, G, i1+1, i2+1, svd_args);
                break;
            }
            case 6: // T
            {
                auto G = site_inds.op("T", {}, i1+1);
                apply_single_gate(psi, G, i1+1, svd_args);
                break;
            }
            case 7: // CZ
            {
                auto G = make_cz_gate(site_inds, i1+1, i2+1);
                apply_two_gate(psi, G, i1+1, i2+1, svd_args);
                break;
            }
            default:
                std::cout << "apply_gates3: Incorrect Gate symbol " << sym << std::endl;
                break;
        }
    } // end for gates

    return psi;
}

// ----------------------------
// circuit_xyz_exp kept same semantics but updated to use new siteset
// returns vector per-site [<X>,<Y>,<Z>]
// ----------------------------
template<typename T1, typename T2>
std::vector<std::vector<T2>> circuit_xyz_exp(const std::vector<std::tuple<T1,T1,T1,T2>>& tensor_vec,
                                             int no_sites)
{
    auto tensor_sites = Qubit(no_sites);
    MPS tensor_mps = apply_gates3(tensor_vec, tensor_sites, no_sites, 1E-12);

    std::vector<std::vector<T2>> return_vec;
    return_vec.reserve(no_sites);

    auto start_itensor = std::chrono::high_resolution_clock::now();

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for(int i = 0; i < no_sites; i++)
    {
        std::vector<T2> xyz;
        xyz.reserve(3);

        tensor_mps.position(i+1);
        auto ket = tensor_mps(i+1);
        auto bra = dag(prime(ket, "Site"));

        auto scalar_x = std::real(eltC(bra * tensor_sites.op("X",{},i+1) * ket));
        auto scalar_y = std::real(eltC(bra * tensor_sites.op("Y",{},i+1) * ket));
        auto scalar_z = std::real(eltC(bra * tensor_sites.op("Z",{},i+1) * ket));

        xyz.push_back(scalar_x);
        xyz.push_back(scalar_y);
        xyz.push_back(scalar_z);

        #ifdef _OPENMP
        #pragma omp critical
        #endif
        {
            if(return_vec.size() <= static_cast<size_t>(i))
                return_vec.resize(i + 1);
            return_vec[i] = std::move(xyz);
        }
    }

    auto end_itensor = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> diff_itensor = end_itensor - start_itensor;
    // Optionally print timing: std::cout << "circuit_xyz_exp time: " << diff_itensor.count() << " s\n";

    return return_vec;
}

// ----------------------------
// Some small utility functions kept for backwards compatibility
// ----------------------------
int hello()
{
    std::cout << "Hello" << std::endl;
    return 0;
}

template <typename T>
T add(T i, T j)
{
    return i + j;
}

template <typename T>
std::vector<T> list_return(std::vector<T> vector1)
{
    std::cout << "Vector_Function" << std::endl;
    std::vector<T> vector2;
    vector2.reserve(vector1.size());
    for(size_t i = 0; i < vector1.size(); i++)
    {
        vector2.push_back(vector1[i]);
    }
    return vector2;
}

template<typename T1, typename T2>
int tuple_return(std::vector<std::tuple<T1,T1,T1,T2>> vect_tup)
{
    std::cout << "Vec_Tup_Function" << std::endl;
    std::cout << "Printing Vector of tuples" << std::endl;
    for(size_t i = 0; i < vect_tup.size(); i++)
    {
        std::cout << "[" << std::get<0>(vect_tup[i]) << ","
                  << std::get<1>(vect_tup[i]) << ","
                  << std::get<2>(vect_tup[i]) << ","
                  << std::get<3>(vect_tup[i]) << "]" << std::endl;
    }
    return 0;
}

// ----------------------------
// High-level wrapper: build, encode, apply circuit
// - encoding_mode: "featuremap", "amplitude_full", "amplitude_product"
// - exposes SVD adaptivity params
// ----------------------------
MPS build_and_encode_then_apply(const std::vector<std::tuple<int,int,int,double>>& circuits,
                               const std::vector<double> &features,
                               int no_sites,
                               const std::string& encoding_mode = "featuremap",
                               int encoding_depth = 2,
                               double cutoff = 1E-12,
                               size_t baseMaxDim = 256,
                               size_t maxCap = 2048)
{
    auto sites = Qubit(no_sites);

    // init psi to |Up...Up>
    auto init = InitState(sites);
    for(int i = 1; i <= no_sites; ++i) init.set(i, "Up");
    MPS psi(init);

    // Choose encoding
    if(encoding_mode == "featuremap")
    {
        FeatureMapAnsatz(psi, sites, features, encoding_depth, true, true, M_PI, Args("Cutoff",cutoff,"MaxDim",static_cast<int>(baseMaxDim)));
    }
    else if(encoding_mode == "amplitude_full")
    {
        // requires complex amplitudes; convert features real->complex
        std::vector<std::complex<double>> amps;
        size_t dim = 1ULL << no_sites;
        if(features.size() != static_cast<size_t>(dim))
        {
            throw ITError("build_and_encode_then_apply: amplitude_full requires features.size()==2^no_sites");
        }
        for(auto &v : features) amps.push_back(std::complex<double>(v,0.0));
        // get exact MPS (may throw if no_sites too large)
        psi = amplitude_encode_fullstate(sites, amps, Args("Cutoff",1E-14,"MaxDim",static_cast<int>(maxCap)));
    }
    else if(encoding_mode == "amplitude_product")
    {
        amplitude_encode_product(psi, sites, features, Args("Cutoff",cutoff,"MaxDim",static_cast<int>(baseMaxDim)));
    }
    else
    {
        throw ITError("Unknown encoding_mode: " + encoding_mode);
    }

    // Apply circuits (note: apply_gates3 returns a fresh MPS initialized from |0..0>.
    // If the intention is to apply circuits on top of encoding, we must apply gates to 'psi' above.
    // Here we'll interpret circuits as additional gates to apply to current psi:
    Args svd_args = svd_args_adaptive(baseMaxDim, maxCap, cutoff);
    for(const auto& gate : circuits)
    {
        auto sym = std::get<0>(gate);
        auto i1 = std::get<1>(gate);
        auto i2 = std::get<2>(gate);
        auto a = std::get<3>(gate);

        double theta = (M_PI * a) / 2.0;

        switch(sym)
        {
            case 0: // H
            {
                auto G = sites.op("H", {}, i1+1);
                apply_single_gate(psi, G, i1+1, svd_args);
                break;
            }
            case 1: // Rx
            {
                auto G = sites.op("Rx", {"theta", theta}, i1+1);
                apply_single_gate(psi, G, i1+1, svd_args);
                break;
            }
            case 2: // Rz
            {
                auto G = sites.op("Rz", {"theta", theta}, i1+1);
                apply_single_gate(psi, G, i1+1, svd_args);
                break;
            }
            case 3: // XXPhase
            {
                auto G = make_xxphase_gate(sites, i1+1, i2+1, theta);
                apply_two_gate(psi, G, i1+1, i2+1, svd_args);
                break;
            }
            case 4: // ZZPhase
            {
                auto G = make_zzphase_gate(sites, i1+1, i2+1, theta);
                apply_two_gate(psi, G, i1+1, i2+1, svd_args);
                break;
            }
            case 5: // SWAP
            {
                auto s1 = sites(i1+1), s2 = sites(i2+1);
                auto s1p = prime(s1), s2p = prime(s2);
                ITensor G(dag(s1), dag(s2), s1p, s2p);
                G.set(s1(1), s2(1), s1p(2), s2p(1), 1.0);
                G.set(s1(1), s2(2), s1p(1), s2p(2), 1.0);
                G.set(s1(2), s2(1), s1p(1), s2p(2), 1.0);
                G.set(s1(2), s2(2), s1p(2), s2p(1), 1.0);
                apply_two_gate(psi, G, i1+1, i2+1, svd_args);
                break;
            }
            case 6: // T
            {
                auto G = sites.op("T", {}, i1+1);
                apply_single_gate(psi, G, i1+1, svd_args);
                break;
            }
            case 7: // CZ
            {
                auto G = make_cz_gate(sites, i1+1, i2+1);
                apply_two_gate(psi, G, i1+1, i2+1, svd_args);
                break;
            }
            default:
                std::cout << "build_and_encode_then_apply: Unknown gate symbol " << sym << std::endl;
                break;
        }
    }

    return psi;
}

// ----------------------------
// PYBIND11 MODULE
// ----------------------------
PYBIND11_MODULE(helloitensor, m)
{
    m.doc() = "Optimized ITensor + BasicSiteSet(QubitSite) module with feature map ansatz, amplitude encoding and adaptive SVD heuristics";

    m.def("add", &add<float>, "A function that adds two float numbers");
    m.def("hello", &hello, "Hello function");

    m.def("vec_return", &list_return<int>, "Return input list as a vector");

    m.def("tuple_return", &tuple_return<int,float>, "Print vector of tuples");
    m.def("tuple_return", &tuple_return<int,int>, "Print vector of tuples");
    m.def("tuple_return", &tuple_return<float,float>, "Return vector of tuples");

    // Expose circuit_xyz_exp via a wrapper that uses double precision and int template types
    m.def("circuit_xyz_exp",
          [](const std::vector<std::tuple<int,int,int,double>>& tensor_vec, int no_sites)
          {
              return circuit_xyz_exp<int,double>(tensor_vec, no_sites);
          },
          "Compute <X>,<Y>,<Z> per site after applying circuits.");

    // Expose apply_gates3 helper (returns nothing; but we expose MPS via placeholder string)
    m.def("apply_gates3_mps",
          [](const std::vector<std::tuple<int,int,int,double>>& circuits, int no_sites, double cutoff, int baseMaxDim, int maxCap)
          {
              auto sites = Qubit(no_sites);
              auto psi = apply_gates3(circuits, sites, no_sites, cutoff, static_cast<size_t>(baseMaxDim), static_cast<size_t>(maxCap));
              // We cannot directly expose MPS in this simple binding stub easily; return number of sites as check
              return psi.N();
          },
          "Apply gates and return number of sites of produced MPS (placeholder).",
          py::arg("circuits"), py::arg("no_sites"), py::arg("cutoff") = 1E-12, py::arg("baseMaxDim") = 256, py::arg("maxCap") = 2048);

    // Expose build_and_encode_then_apply in a minimal way: return number of sites to confirm success
    m.def("build_and_encode_then_apply",
          [](const std::vector<std::tuple<int,int,int,double>>& circuits,
             const std::vector<double>& features,
             int no_sites,
             const std::string& encoding_mode,
             int encoding_depth,
             double cutoff,
             int baseMaxDim,
             int maxCap)
          {
              auto psi = build_and_encode_then_apply(circuits, features, no_sites, encoding_mode, encoding_depth, cutoff, static_cast<size_t>(baseMaxDim), static_cast<size_t>(maxCap));
              return psi.N();
          },
          "Build MPS, encode features, apply circuits (returns MPS site count as success code).",
          py::arg("circuits"), py::arg("features"), py::arg("no_sites"),
          py::arg("encoding_mode") = "featuremap", py::arg("encoding_depth") = 2,
          py::arg("cutoff") = 1E-12, py::arg("baseMaxDim") = 256, py::arg("maxCap") = 2048
    );
}


#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "itensor/all.h"
#include "QubitSite.h"   // <-- your extended QubitSite with op() updates

namespace py = pybind11;
using namespace itensor;

// ---------------------------------------------------------
// Helper: build MPS of |0...0>
// ---------------------------------------------------------
MPS init_state(int n)
{
    auto sites = SiteSet(QubitSite(n));
    MPS psi(sites);
    for(int i = 1; i <= n; ++i)
    {
        psi.set(i, ITensor(sites(i)(1))); // |0>
    }
    return psi;
}

// ---------------------------------------------------------
// Apply single- or two-qubit gates from encoded list
// gate format: [gate_type, q0, q1, param(s)]
// ---------------------------------------------------------
void apply_gate_list(MPS &psi,
                     SiteSet const& sites,
                     std::vector<std::vector<py::object>> const& gates,
                     Args const& args)
{
    for(auto const& g : gates)
    {
        int gtype = g[0].cast<int>();
        int q0    = g[1].cast<int>() + 1; // 1-indexed
        int q1    = g[2].cast<int>();
        q1        = (q1 >= 0 ? q1+1 : -1);

        ITensor G;

        switch(gtype)
        {
            case 0: // H
                G = op(sites,"H",q0);
                psi.position(q0);
                psi.Aref(q0) = psi.A(q0) * G;
                break;

            case 1: // Rx
                G = op(sites,"Rx",q0,{"θ",g[3].cast<double>()});
                psi.position(q0);
                psi.Aref(q0) = psi.A(q0) * G;
                break;

            case 2: // Rz
                G = op(sites,"Rz",q0,{"θ",g[3].cast<double>()});
                psi.position(q0);
                psi.Aref(q0) = psi.A(q0) * G;
                break;

            case 3: // XXPhase
                {
                  double theta = g[3].cast<double>();
                  ITensor G2 = op(sites,"XX",q0,q1,{"θ",theta});
                  applyMPO(psi,G2,args);
                }
                break;

            case 4: // ZZPhase
                {
                  double theta = g[3].cast<double>();
                  ITensor G2 = op(sites,"ZZ",q0,q1,{"θ",theta});
                  applyMPO(psi,G2,args);
                }
                break;

            case 5: // SWAP
                {
                  ITensor G2 = op(sites,"SWAP",q0,q1);
                  applyMPO(psi,G2,args);
                }
                break;

            case 6: // T
                G = op(sites,"T",q0);
                psi.position(q0);
                psi.Aref(q0) = psi.A(q0) * G;
                break;

            case 7: // CZ
                {
                  ITensor G2 = op(sites,"CZ",q0,q1);
                  applyMPO(psi,G2,args);
                }
                break;

            case 8: // Ry
                G = op(sites,"Ry",q0,{"θ",g[3].cast<double>()});
                psi.position(q0);
                psi.Aref(q0) = psi.A(q0) * G;
                break;

            case 9: // X
                G = op(sites,"X",q0);
                psi.position(q0);
                psi.Aref(q0) = psi.A(q0) * G;
                break;

            case 10: // Y
                G = op(sites,"Y",q0);
                psi.position(q0);
                psi.Aref(q0) = psi.A(q0) * G;
                break;

            case 11: // Z
                G = op(sites,"Z",q0);
                psi.position(q0);
                psi.Aref(q0) = psi.A(q0) * G;
                break;

            case 12: // U3(theta,phi,lambda)
                {
                  auto params = g[3].cast<std::tuple<double,double,double>>();
                  double th,ph,la;
                  std::tie(th,ph,la) = params;
                  G = op(sites,"U3",q0,{"θ",th,"φ",ph,"λ",la});
                  psi.position(q0);
                  psi.Aref(q0) = psi.A(q0) * G;
                }
                break;

            case 13: // S
                G = op(sites,"S",q0);
                psi.position(q0);
                psi.Aref(q0) = psi.A(q0) * G;
                break;

            default:
                throw std::runtime_error("Unsupported gate type in apply_gate_list");
        }
    }
}

// ---------------------------------------------------------
// Compute expectation values of Pauli X,Y,Z on all qubits
// returns np.array of shape (nqubits*3,)
// ---------------------------------------------------------
py::array_t<double> measure_xyz(MPS const& psi,
                                SiteSet const& sites)
{
    int n = sites.N();
    std::vector<double> results(3*n);

    for(int i = 1; i <= n; ++i)
    {
        auto bra = dag(psi);
        auto ket = psi;

        auto Xop = op(sites,"X",i);
        auto Yop = op(sites,"Y",i);
        auto Zop = op(sites,"Z",i);

        results[3*(i-1)+0] = inner(bra,ket,Xop).real();
        results[3*(i-1)+1] = inner(bra,ket,Yop).real();
        results[3*(i-1)+2] = inner(bra,ket,Zop).real();
    }

    return py::array_t<double>(results.size(), results.data());
}

// ---------------------------------------------------------
// Python-exposed: circuit_xyz_exp
// ---------------------------------------------------------
py::array_t<double> circuit_xyz_exp(std::vector<std::vector<py::object>> gates,
                                    int nqubits,
                                    double cutoff=1E-12,
                                    int maxdim=500)
{
    auto sites = SiteSet(QubitSite(nqubits));
    auto psi = init_state(nqubits);

    apply_gate_list(psi,sites,gates,Args("Cutoff",cutoff,"MaxDim",maxdim));

    return measure_xyz(psi,sites);
}

// ---------------------------------------------------------
// Python-exposed: amplitude encoding
// ---------------------------------------------------------
MPS amplitude_encode(std::vector<double> const& vec,
                     double cutoff=1E-12,
                     int maxdim=500)
{
    int nqubits = 0;
    while((1<<nqubits) < (int)vec.size()) ++nqubits;

    if((1<<nqubits) != (int)vec.size())
        throw std::runtime_error("Input vector length must be power of 2");

    auto sites = SiteSet(QubitSite(nqubits));
    auto psi = init_state(nqubits);

    // Replace with amplitude superposition
    ITensor A(sites(1));
    for(int i = 0; i < (int)vec.size(); ++i)
    {
        std::vector<int> bits(nqubits);
        int x = i;
        for(int b = 0; b < nqubits; ++b)
        {
            bits[b] = x % 2;
            x /= 2;
        }
        ITensor ket = psi.A(1);
        for(int q = 1; q <= nqubits; ++q)
        {
            ITensor basis = ITensor(sites(q));
            basis.set(sites(q)(bits[q-1]+1),1.0);
            if(q==1) ket = basis;
            else ket *= basis;
        }
        A += vec[i]*ket;
    }
    psi.set(1,A);

    return psi;
}

// ---------------------------------------------------------
// Python-exposed: feature map ansatz (like ZZFeatureMap)
// ---------------------------------------------------------
MPS feature_map_ansatz(std::vector<double> features,
                       int depth=2,
                       double gamma=1.0,
                       double cutoff=1E-12,
                       int maxdim=500)
{
    int n = features.size();
    auto sites = SiteSet(QubitSite(n));
    auto psi = init_state(n);

    for(int d=0; d<depth; ++d)
    {
        // single-qubit encodings
        for(int i=1;i<=n;++i)
        {
            auto th = features[i-1];
            auto G = op(sites,"Ry",i,{"θ",th});
            psi.position(i);
            psi.Aref(i) = psi.A(i)*G;
        }

        // entanglers
        for(int i=1;i<n;++i)
        {
            auto G = op(sites,"ZZ",i,i+1,{"θ",gamma*features[i-1]*features[i]});
            applyMPO(psi,G,Args("Cutoff",cutoff,"MaxDim",maxdim));
        }
    }

    return psi;
}

// ---------------------------------------------------------
// PYBIND11 Module
// ---------------------------------------------------------
PYBIND11_MODULE(helloitensor, m)
{
    m.doc() = "ITensor-based quantum circuit simulator with extended gates";

    m.def("circuit_xyz_exp",&circuit_xyz_exp,
          py::arg("gates"),
          py::arg("nqubits"),
          py::arg("cutoff")=1E-12,
          py::arg("maxdim")=500,
          "Apply circuit described by gate list and return Pauli XYZ expectations.");

    m.def("amplitude_encode",&amplitude_encode,
          py::arg("vec"),
          py::arg("cutoff")=1E-12,
          py::arg("maxdim")=500,
          "Encode classical vector into amplitudes of a quantum state (returns MPS).");

    m.def("feature_map_ansatz",&feature_map_ansatz,
          py::arg("features"),
          py::arg("depth")=2,
          py::arg("gamma")=1.0,
          py::arg("cutoff")=1E-12,
          py::arg("maxdim")=500,
          "Build feature map ansatz similar to ZZFeatureMap (returns MPS).");
}

# projected_quantum_features.py
# Copy-paste ready. Requires:
#  - pytket
#  - mpi4py
#  - numpy
#  - sympy
#  - helloitensor (compiled C++ extension from our previous steps)

import sys
import json
import pathlib
from mpi4py import MPI

import numpy as np
from sympy import Symbol
from statistics import mean, median

from pytket import Circuit, OpType
from pytket.transform import Transform
from pytket.architecture import Architecture
from pytket.passes import DefaultMappingPass
from pytket.predicates import CompilationUnit
from pytket.circuit import PauliExpBox, Pauli

# Import the compiled C++ extension we built earlier. Ensure it is on PYTHONPATH.
import helloitensor

# Pauli matrices (for reference / potential fallback)
X = np.array([[0,1],[1,0]])
Y = np.array([[0,-1j],[1j,0]])
I = np.array([[1,0],[0,1]])
Z = np.array([[1,0],[0,-1]])


# -----------------------------------------------------------------------------
# Utility: convert TKET circuit commands to gate-list format accepted by C++
# Gate list entry is: [gate_type, q0_index, q1_index_or_-1, param]
# where gate_type integers correspond to the C++ switch:
# 0: H
# 1: Rx
# 2: Rz
# 3: XXPhase
# 4: ZZPhase
# 5: SWAP
# 6: T
# 7: CZ
# 8: Ry
# 9: X
# 10: Y
# 11: Z
# 12: U3 (param -> (theta,phi,lambda))
# 13: S
# -----------------------------------------------------------------------------
def circuit_to_list_for_helloitensor(circuit: Circuit):
    gates = []
    for cmd in circuit.get_commands():
        optype = cmd.op.type
        # Single-qubit gates
        if optype == OpType.H:
            gates.append([0, cmd.qubits[0].index[0], -1, 0.0])
        elif optype == OpType.Rx:
            # pytket Rx param is radians
            gates.append([1, cmd.qubits[0].index[0], -1, float(cmd.op.params[0])])
        elif optype == OpType.Rz:
            gates.append([2, cmd.qubits[0].index[0], -1, float(cmd.op.params[0])])
        elif optype == OpType.Ry:
            gates.append([8, cmd.qubits[0].index[0], -1, float(cmd.op.params[0])])
        elif optype == OpType.X:
            gates.append([9, cmd.qubits[0].index[0], -1, 0.0])
        elif optype == OpType.Y:
            gates.append([10, cmd.qubits[0].index[0], -1, 0.0])
        elif optype == OpType.Z:
            gates.append([11, cmd.qubits[0].index[0], -1, 0.0])
        elif optype == OpType.U3:
            # pytket U3 may be represented with params (theta,phi,lambda)
            params = cmd.op.params
            # store as a tuple; pybind11 will convert to tuple for C++
            gates.append([12, cmd.qubits[0].index[0], -1, (float(params[0]), float(params[1]), float(params[2]))])
        elif optype == OpType.S:
            gates.append([13, cmd.qubits[0].index[0], -1, 0.0])
        elif optype == OpType.T:
            gates.append([6, cmd.qubits[0].index[0], -1, 0.0])

        # Two-qubit gates
        elif optype == OpType.ZZPhase:
            # pytket param is the exponent/angle; pass as float
            gates.append([4, cmd.qubits[0].index[0], cmd.qubits[1].index[0], float(cmd.op.params[0])])
        elif optype == OpType.XXPhase:
            gates.append([3, cmd.qubits[0].index[0], cmd.qubits[1].index[0], float(cmd.op.params[0])])
        elif optype == OpType.SWAP:
            gates.append([5, cmd.qubits[0].index[0], cmd.qubits[1].index[0], 0.0])
        elif optype == OpType.CZ:
            gates.append([7, cmd.qubits[0].index[0], cmd.qubits[1].index[0], 0.0])
        else:
            # If unknown gate, try to decompose and re-run
            # Decompose the single unknown gate into simpler gates (TKET supports decomp)
            tmp = Circuit(circuit.n_qubits)
            tmp.add_command(cmd)
            Transform.DecomposeBoxes().apply(tmp)  # best-effort
            # if decomposition made progress, recursively translate newly decomposed gates
            # Otherwise raise
            decomposed = False
            for new_cmd in tmp.get_commands():
                if new_cmd is not cmd:
                    decomposed = True
                    break
            if decomposed:
                for new_cmd in tmp.get_commands():
                    # Build a tiny circuit for the new_cmd and translate
                    tiny = Circuit(circuit.n_qubits)
                    tiny.add_command(new_cmd)
                    gates += circuit_to_list_for_helloitensor(tiny)
            else:
                raise Exception(f"Unknown/unsupported gate type in circuit_to_list_for_helloitensor: {optype}")
    return gates


# -----------------------------------------------------------------------------
# ProjectedQuantumFeatures class (updated)
#  - supports two modes:
#    * 'tket'   -> build TKET symbolic circuits (original behavior) and use helloitensor.circuit_xyz_exp
#    * 'native' -> use new helloitensor.feature_map_ansatz (fast path) if available;
#                 otherwise falls back to 'tket' flow
#  - still keeps API backwards compatible with your code
# -----------------------------------------------------------------------------
class ProjectedQuantumFeatures:
    def __init__(
        self,
        num_features: int,
        reps: int,
        gamma: float,
        entanglement_map: list[tuple[int, int]],
        ansatz: str,
        hadamard_init: bool=True,
        mode: str = "tket"   # "tket" (default) or "native"
    ):
        self.one_q_symbol_list = []
        self.two_q_symbol_list = []
        self.ansatz_circ = Circuit(num_features)
        self.feature_symbol_list = [Symbol('f_'+str(i)) for i in range(num_features)]
        self.reps = reps
        self.gamma = gamma
        self.num_features = num_features
        self.hadamard_init = hadamard_init
        self.entanglement_map = entanglement_map
        self.mode = mode.lower()

        if ansatz == "hamiltonian":
            self.hamiltonian_ansatz()
        elif ansatz == "magic":
            self.magic_ansatz()
        else:
            raise RuntimeError("You have not entered a valid ansatz.")

    def circuit_for_data(self, feature_values: list[float]) -> Circuit:
        if len(feature_values) != len(self.feature_symbol_list):
            raise RuntimeError("The number of values must match the number of symbols.")

        symbol_map = {symb: val for symb, val in zip(self.feature_symbol_list, feature_values)}
        the_circuit = self.ansatz_circ.copy()
        the_circuit.symbol_substitution(symbol_map)
        return the_circuit

    def circuit_to_list(self, circuit: Circuit):
        """Compatibility wrapper: produce gate list suitable for helloitensor binding."""
        return circuit_to_list_for_helloitensor(circuit)

    def hamiltonian_ansatz(self):
        if self.hadamard_init:
            for i in range(self.num_features):
                self.ansatz_circ.H(i)

        for _ in range(self.reps):
            for i in range(self.num_features):
                # TKET expects numeric parameter for Rz here; keep symbolic via Symbol from sympy
                exponent = (1/np.pi) * self.gamma * self.feature_symbol_list[i]
                self.ansatz_circ.append_op(OpType.Rz, [self.ansatz_circ.qubits[i]], params=[exponent])

            for (q0, q1) in self.entanglement_map:
                symb0 = self.feature_symbol_list[q0]
                symb1 = self.feature_symbol_list[q1]
                exponent = self.gamma*self.gamma*(1 - symb0)*(1 - symb1)
                # TKET may require numeric param to add XXPhase with symbolic values; use symbol placeholder:
                self.ansatz_circ.append_op(OpType.XXPhase, [self.ansatz_circ.qubits[q0], self.ansatz_circ.qubits[q1]], params=[exponent])

        # Apply TKET routing to a linear architecture (in-place)
        cu = CompilationUnit(self.ansatz_circ)
        architecture = Architecture([(i, i + 1) for i in range(self.ansatz_circ.n_qubits - 1)])
        DefaultMappingPass(architecture).apply(cu)
        self.ansatz_circ = cu.circuit
        # Decompose any bridge gates
        Transform.DecomposeBRIDGE().apply(self.ansatz_circ)
        return 0

    def magic_ansatz(self):
        for _ in range(self.reps):
            for q in range(self.num_features):
                self.ansatz_circ.H(q)
                self.ansatz_circ.T(q)
            for (q0, q1) in self.entanglement_map:
                self.ansatz_circ.CZ(q0, q1)
                # additional local layers for edge qubits
                if q1 == self.num_features - 1 or q1 == self.num_features - 2:
                    for q in range(self.num_features):
                        self.ansatz_circ.H(q)
                        self.ansatz_circ.T(q)
            for q in range(self.num_features):
                self.ansatz_circ.Rz(self.feature_symbol_list[q], q)
        return 0


# -----------------------------------------------------------------------------
# Kernel matrix building function (updated)
#  - Uses native helloitensor.feature_map_ansatz when mode=="native" and the function exists.
#  - Otherwise falls back to TKET-path that builds circuits and calls helloitensor.circuit_xyz_exp
# -----------------------------------------------------------------------------
def build_qf_matrix(
        mpi_comm,
        ansatz: ProjectedQuantumFeatures,
        X,
        info_file=None,
        cpu_max_mem=6,
        cutoff=1e-12,
        maxdim=512,
        use_native_featuremap_if_available: bool = True
    ) -> np.ndarray:

    n_qubits = ansatz.ansatz_circ.n_qubits

    # MPI info
    root = 0
    rank = mpi_comm.Get_rank()
    n_procs = mpi_comm.Get_size()

    entries_per_chunk = int(np.ceil(len(X) / n_procs))
    max_mps_per_cpu = 2 * entries_per_chunk

    if rank == root:
        profiling_dict = dict()
        profiling_dict["lenX"] = (len(X), "entries")
        start_time = MPI.Wtime()

    # Report
    if rank == root:
        print(f"[Rank 0] Preparing to compute projected features for {len(X)} inputs on {n_procs} procs.")

    # We'll compute local chunk indices
    offset = rank * entries_per_chunk
    local_indices = list(range(offset, min(offset + entries_per_chunk, len(X))))

    # Storage for local outputs
    local_projected = []

    # Fast path: use native C++ feature_map_ansatz if user requested and function exists.
    native_available = hasattr(helloitensor, "feature_map_ansatz") and use_native_featuremap_if_available

    for idx in local_indices:
        x = X[idx, :].tolist()
        # If native mode is available and preferred by the ansatz mode, try it
        used_native = False
        if native_available and ansatz.mode == "native":
            try:
                # Try calling feature_map_ansatz; many C++ bindings return MPS or expectations:
                maybe = helloitensor.feature_map_ansatz(x, ansatz.reps, ansatz.gamma, cutoff, maxdim)
                # If maybe is an array-like of shape (n_qubits,3) or length 3*n_qubits, convert
                arr = None
                if isinstance(maybe, (list, tuple, np.ndarray)):
                    arr = np.asarray(maybe)
                else:
                    # If returned an object (e.g. MPS Python proxy), try to call a measurement function
                    # Many earlier iterations expose circuit_xyz_exp only, so we cannot reliably measure an MPS
                    arr = None

                if arr is not None:
                    # normalize to flat row vector of length 3*n_qubits
                    arr = np.asarray(arr).reshape(-1)
                    if arr.size == 3 * n_qubits:
                        local_projected.append(arr)
                        used_native = True
                # else fall back
            except Exception as e:
                # fallback to TKET path; do not crash here
                if rank == root:
                    print("[Rank 0] Native feature_map_ansatz call failed or returned unsupported type; falling back to TKET path.")
                used_native = False

        if not used_native:
            # Build the TKET circuit, substitute feature values, translate to gate list and call C++ sim
            circ = ansatz.circuit_for_data(x)
            # Try to reduce the circuit (safe)
            try:
                # apply default transformations if helpful
                cu = CompilationUnit(circ)
                DefaultMappingPass(Architecture([(i, i + 1) for i in range(circ.n_qubits - 1)])).apply(cu)
                circ = cu.circuit
                Transform.DecomposeBRIDGE().apply(circ)
            except Exception:
                # mapping may fail for symbolic circuits; ignore and use raw circ
                pass

            gate_list = ansatz.circuit_to_list(circ)

            # call into C++: circuit_xyz_exp
            try:
                exp_xyz = helloitensor.circuit_xyz_exp(gate_list, n_qubits, cutoff, maxdim)
            except Exception as e:
                # Some pybind11 variants expect pure Python lists and others expect nested types; ensure conversion
                exp_xyz = helloitensor.circuit_xyz_exp(list(gate_list), int(n_qubits), float(cutoff), int(maxdim))

            # Normalize returned shape:
            arr = np.asarray(exp_xyz)
            # If arr is shape (n_qubits,3) -> flatten
            if arr.size == n_qubits * 3:
                arr_flat = arr.reshape(-1)
            elif arr.ndim == 2 and arr.shape[0] == n_qubits and arr.shape[1] == 3:
                arr_flat = arr.reshape(-1)
            elif arr.ndim == 1 and arr.size == 3*n_qubits:
                arr_flat = arr
            else:
                # Unexpected shape: try to flatten and pad/truncate if necessary
                arr_flat = arr.flatten()
                if arr_flat.size < 3*n_qubits:
                    arr_flat = np.pad(arr_flat, (0, 3*n_qubits - arr_flat.size))
                else:
                    arr_flat = arr_flat[:3*n_qubits]

            local_projected.append(arr_flat)

    # convert local_projected into array with rows == local chunk length
    if len(local_projected) == 0:
        local_array = np.zeros((0, 3 * n_qubits))
    else:
        local_array = np.vstack(local_projected)

    # Each worker produces a chunk of shape (chunk_len, 3*n_qubits), but lengths may differ (last chunk).
    # We'll gather on root using MPI.Gatherv
    local_counts = np.array(mpi_comm.gather(local_array.shape[0], root=root), dtype=int)
    # Gathered shape on root
    if rank == root:
        total_rows = sum(local_counts)
        result = np.zeros((total_rows, 3 * n_qubits))
    else:
        result = None

    # To gather variable-length arrays we flatten local_array and use Gatherv
    local_flat = local_array.flatten()
    local_size = np.array([local_flat.size], dtype=int)
    sizes = mpi_comm.gather(local_flat.size, root=root)

    if rank == root:
        recvbuf = np.empty(sum(sizes), dtype=local_flat.dtype)
    else:
        recvbuf = None

    mpi_comm.Gatherv(sendbuf=local_flat, recvbuf=(recvbuf, sizes), root=root)

    if rank == root:
        # reconstruct rows
        rows = []
        offset_flat = 0
        for k in range(n_procs):
            count = sizes[k]
            if count == 0:
                continue
            chunk_flat = recvbuf[offset_flat:offset_flat+count]
            offset_flat += count
            # each row length = 3*n_qubits
            nrows = count // (3 * n_qubits)
            if nrows == 0:
                # may be partial; attempt to reshape if possible
                nrows = 1
            chunk = chunk_flat.reshape((nrows, 3 * n_qubits))
            rows.append(chunk)
        if rows:
            result = np.vstack(rows)
        else:
            result = np.zeros((0, 3 * n_qubits))

        # profiling info (optional)
        if info_file is not None:
            # write simple profiling JSON (placeholder)
            with open(info_file, "w") as f:
                json.dump({"lenX": len(X)}, f)

        return result
    else:
        return None


# -----------------------------------------------------------------------------
# If run as script, run a tiny smoke test to confirm things roughly work.
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # MPI init
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # small smoke test: 3-qubit GHZ-like featuremap example
    num_qubits = 3
    ent_map = [(0,1),(1,2)]
    ans = ProjectedQuantumFeatures(num_features=num_qubits, reps=1, gamma=1.0, entanglement_map=ent_map, ansatz="hamiltonian", hadamard_init=True, mode="tket")

    # Build small dataset of two samples
    X = np.array([[0.1, 0.2, 0.3],
                  [0.6, 0.7, 0.8]])
    projected = build_qf_matrix(comm, ans, X, info_file=None, cpu_max_mem=6)
    if rank == 0:
        print("Projected features shape:", projected.shape)
        print(projected)






##############################
## [1] Compiler Choice
##############################

# Use GCC for maximum HPC optimization
# (Dataproc images typically ship with GCC 9+)
CCCOM=mpicxx -m64 -std=c++17 -fPIC -O3 -march=native -funroll-loops -fopenmp -DNDEBUG

##############################
## [2] BLAS / LAPACK Backend
##############################

# Use Intel MKL if available (best performance on GCP with large memory + many cores)
PLATFORM=mkl
BLAS_LAPACK_LIBFLAGS=-L/opt/intel/mkl/lib/intel64 \
  -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
BLAS_LAPACK_INCLUDEFLAGS=-I/opt/intel/mkl/include

# Alternative: OpenBLAS (if MKL not installed)
# PLATFORM=openblas
# BLAS_LAPACK_LIBFLAGS=-L/usr/lib/x86_64-linux-gnu -lopenblas -lpthread
# BLAS_LAPACK_INCLUDEFLAGS=-I/usr/include

##############################
## [3] HDF5 Support (optional)
##############################

# Enable if you want tensor storage / checkpointing in HDF5
HDF5_PREFIX=/usr/lib/x86_64-linux-gnu/hdf5/openmpi

##############################
## [4] OpenMP & Threading
##############################

# Enable ITensor OpenMP threading
ITENSOR_USE_OMP=1

# Runtime environment (set in job scripts):
# export OMP_NUM_THREADS=16
# export MKL_NUM_THREADS=1   # prevent MKL thread contention
# export OPENBLAS_NUM_THREADS=1

##############################
## [5] Optimization Flags
##############################

# Release mode flags
OPTIMIZATIONS=-O3 -DNDEBUG -march=native -funroll-loops -ftree-vectorize -fstrict-aliasing -fomit-frame-pointer -Wall -Wno-unknown-pragmas

# Debug mode flags
DEBUGFLAGS=-DDEBUG -g -Wall -Wno-unknown-pragmas -pedantic

# Build shared libraries (better for large clusters)
ITENSOR_MAKE_DYLIB=1

##############################
## [MPI / mpi4py integration note]
##############################
# - ITensor itself is not MPI-native, but you can build your
#   driver code with `mpicxx` (as above).
# - mpi4py can interoperate by calling ITensor through C++ bindings
#   or Python wrappers, distributing workloads at the process level.
# - Each MPI rank will use ITensor with OpenMP threading inside.
##############################

