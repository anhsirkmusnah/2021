#pragma once

#include "itensor/all.h"
#include "itensor/mps/siteset.h"
#include "itensor/util/str.h"
#include <complex>
#include <cmath>

namespace itensor {

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
        if( args.defined("SiteNumber") )
            ts.addTags("n="+str(args.getInt("SiteNumber")));
        auto conserveqns = args.getBool("ConserveQNs",false);
        auto conserveSz = args.getBool("ConserveSz",conserveqns);
        auto conserveParity = args.getBool("ConserveParity",false);
        if(conserveSz && conserveParity)
            {
            s = Index(QN({"Sz",+1},{"Parity",1,2}),1,
                     QN({"Sz",-1},{"Parity",0,2}),1,Out,ts);
            }
        else if(conserveSz)
            {
            s = Index(QN({"Sz",+1}),1,
                     QN({"Sz",-1}),1,Out,ts);
            }
        else if(conserveParity)
            {
            s = Index(QN({"Parity",1,2}),1,
                     QN({"Parity",0,2}),1,Out,ts);
            }
        else
            {
            s = Index(2,ts);
            }
        }

    Index
    index() const { return s; }

    IndexVal
    state(std::string const& state)
        {
        if(state == "Up") 
            {
            return s(1);
            }
        else
        if(state == "Dn") 
            {
            return s(2);
            }
        else
            {
            throw ITError("State " + state + " not recognized");
            }
        return IndexVal{};
        }

    ITensor
    op(std::string const& opname,
       Args const& args = Args::global()) const
        {
        auto alpha = args.getReal("alpha",0.75);
        
        // Use higher precision pi constant
        const double pi = M_PI;  // From <cmath>, more precise than hardcoded value
        
        double theta = 1.0;
        std::complex<double> i(0.0, 1.0);
        
        auto sP = prime(s);
        auto Up = s(1);
        auto UpP = sP(1);
        auto Dn = s(2);
        auto DnP = sP(2);

        auto Op = ITensor(dag(s),sP);
        
        theta = (pi*alpha)/2;
        
        if(opname == "Rz")
            {
            Op.set(Up,UpP,std::exp((-i)*theta));
            Op.set(Dn,DnP,std::exp((i)*theta));
            }
        else
        if(opname == "Z")
            {
            Op.set(Up,UpP,1);
            Op.set(Dn,DnP,-1);
            }
        else
        if(opname == "H")
            {
            Op.set(Up,UpP,1/std::sqrt(2));
            Op.set(Up,DnP,1/std::sqrt(2));
            Op.set(Dn,UpP,1/std::sqrt(2));
            Op.set(Dn,DnP,-1/std::sqrt(2));
            }
        else
        if(opname == "Rx")
            {
            Op.set(Up,UpP,std::cos(theta));
            Op.set(Dn,DnP,std::cos(theta));
            Op.set(Up,DnP,std::sin(theta)*(-i));
            Op.set(Dn,UpP,std::sin(theta)*(-i));
            }
        else
        if(opname == "X")
            {
            Op.set(Up,DnP,1);
            Op.set(Dn,UpP,1);
            }
        else
        if(opname == "X_half")
            {
            Op.set(Up,DnP,0.5);
            Op.set(Dn,UpP,0.5);
            }
        else
        if(opname == "Y_half")
            {
            Op.set(Up,DnP,-0.5*Cplx_i);
            Op.set(Dn,UpP,+0.5*Cplx_i);
            }
        else
        if(opname == "Z_half")
            {
            Op.set(Up,UpP,+0.5);
            Op.set(Dn,DnP,-0.5);
            }
        else
        if(opname == "Sz")
            {
            Op.set(Up,UpP,+0.5);
            Op.set(Dn,DnP,-0.5);
            }
        else
        if(opname == "Sx")
            {
            Op.set(Up,DnP,+0.5);
            Op.set(Dn,UpP,+0.5);
            }
        else
        if(opname == "ISy")
            {
            Op.set(Up,DnP,-0.5);
            Op.set(Dn,UpP,+0.5);
            }
        else
        if(opname == "Sy")
            {
            Op.set(Up,DnP,+0.5*Cplx_i);
            Op.set(Dn,UpP,-0.5*Cplx_i);
            }
        else
        if(opname == "Sp" || opname == "S+")
            {
            Op.set(Dn,UpP,1);
            }
        else
        if(opname == "Sm" || opname == "S-")
            {
            Op.set(Up,DnP,1);
            }
        else
        if(opname == "projUp")
            {
            Op.set(Up,UpP,1);
            }
        else
        if(opname == "projDn")
            {
            Op.set(Dn,DnP,1);
            }
        else
        if(opname == "S2")
            {
            Op.set(Up,UpP,0.75);
            Op.set(Dn,DnP,0.75);
            }
        // New operators for enhanced fraud detection features
        else
        if(opname == "Identity" || opname == "Id")
            {
            Op.set(Up,UpP,1);
            Op.set(Dn,DnP,1);
            }
        else
            {
            throw ITError("Operator \"" + opname + "\" name not recognized");
            }

        return Op;
        }

    //
    // Deprecated, for backwards compatibility
    //
    QubitSite(int n, Args const& args = Args::global())
        {
        *this = QubitSite({args,"SiteNumber=",n});
        }

    };

} //namespace itensor


#include "itensor/all.h"
#include "itensor/util/print_macro.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <iostream>
#include <tuple>
#include <vector>
#include <chrono>
#include <cmath>
#include <stdexcept>

using namespace itensor;
using namespace std::chrono;

struct data
{
    std::string symbol;
    int index1;
    int index2;
    float alpha;
};

// Enhanced apply_gates3 with input validation and optimized positioning
MPS apply_gates3(std::vector<std::tuple<int,int,int,float>> circuits, Qubit site_inds, int N, double cutoff){
    
    // Input validation for production use
    if(N <= 0) {
        throw std::invalid_argument("Number of sites must be positive");
    }
    if(cutoff <= 0 || cutoff >= 1) {
        throw std::invalid_argument("Cutoff must be between 0 and 1");
    }
    
    // Validate gate parameters
    for (const auto& gate : circuits) {
        auto i1 = std::get<1>(gate);
        auto i2 = std::get<2>(gate);
        if(i1 < 0 || i1 >= N) {
            throw std::out_of_range("Gate index1 out of bounds");
        }
        if(i2 >= N || i2 < -1) {  // -1 is valid for single-qubit gates
            throw std::out_of_range("Gate index2 out of bounds");
        }
    }
    
    // Define variables and constants with high precision
    std::complex<double> i(0.0, 1.0);
    const double pi = M_PI;  // Use standard library constant
    
    // Initialize site indices and states
    auto init = InitState(site_inds);
    for(auto n : range1(N))
        {
        init.set(n,"Up");
        }
    
    // Create an MPS from the site indices
    auto psi = MPS(init);
    
    // Apply each gate to the psi sequentially
    for (std::tuple<int,int,int,float> gate : circuits){
        auto sym = std::get<0>(gate);
        auto i1 = std::get<1>(gate);
        auto i2 = std::get<2>(gate);
        auto a = std::get<3>(gate);
        
        double theta = (pi*a)/2;
        
        if (sym == 0) {  // Hadamard
            auto G = op(site_inds,"H",i1+1, {"alpha=",a});
            psi.position(i1+1);
            auto new_MPS = G*psi(i1+1);
            new_MPS.noPrime();
            psi.set(i1+1,new_MPS);
            
        } else if (sym == 1){  // Rx
            auto G = op(site_inds,"Rx",i1+1, {"alpha=",a});
            psi.position(i1+1);
            auto new_MPS = G*psi(i1+1);
            new_MPS.noPrime();
            psi.set(i1+1,new_MPS);
            
        } else if (sym == 2){  // Rz
            auto G = op(site_inds,"Rz",i1+1, {"alpha=",a});
            psi.position(i1+1);
            auto new_MPS = G* psi(i1+1);
            new_MPS.noPrime();
            psi.set(i1+1,new_MPS);
            
        } else if (sym == 3){  // XXPhase - Optimized positioning
            // Position once at the minimum index
            psi.position(std::min(i1+1, i2+1));
            
            auto opx1 = op(site_inds,"X",i1+1, {"alpha=",a});
            auto opx2 = op(site_inds,"X",i2+1, {"alpha=",a});
            auto G = expHermitian(opx1 * opx2, -i*theta);
            
            auto wf1 = psi(i1+1)*psi(i2+1);
            auto wf = G*wf1;
            wf.noPrime();
            
            // Use configurable cutoff
            auto [U,S,V] = svd(wf,commonInds(wf, psi(i1+1)),{"Cutoff=",cutoff,"SVDMethod=","automatic"});
            
            psi.set(i1+1,U);
            psi.set(i2+1,S*V);
            
        } else if (sym == 4){  // ZZPhase - Optimized
            psi.position(std::min(i1+1, i2+1));
            
            auto op1 = op(site_inds,"Z",i1+1, {"alpha=",a});
            auto op2 = op(site_inds,"Z",i2+1, {"alpha=",a});
            auto G = expHermitian(op1 * op2,-i*theta);
            
            auto wf = psi(i1+1)*psi(i2+1);
            wf *= G;
            wf.noPrime();
            
            // Use configurable cutoff
            auto [U,S,V] = svd(wf,inds(psi(i1+1)),{"Cutoff=",cutoff});
            
            psi.set(i1+1,U);
            psi.set(i2+1,S*V);
            
        } else if (sym == 5){  // SWAP
            psi.position(i1+1);
            
            auto G = op(site_inds,"Z",i1+1, {"alpha=",a})*op(site_inds,"Z",i2+1, {"alpha=",a});
            G.set(1,1,2,2, 0);
            G.set(2,2,1,1, 0);
            G.set(1,2,2,1, 1);
            G.set(2,1,1,2, 1);
            
            auto wf1 = psi(i1+1)*psi(i2+1);
            auto wf = G*wf1;
            wf.noPrime();
            
            // Use configurable cutoff
            auto [U,S,V] = svd(wf,inds(psi(i1+1)),{"Cutoff=",cutoff});
            
            psi.set(i1+1,U);
            psi.set(i2+1,S*V);
            
        } else {
            throw std::invalid_argument("Incorrect gate type: " + std::to_string(sym));
        }
    }
    
    return psi;
}

// Testing functions (backward compatible)
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

int hello(){
    std::cout << "Hello" << std::endl;
    return 0;
}

int main(){
    hello();
    return 0;
}

template<typename T>
T add(T i, T j) {
    return i + j;
}

template<typename T>
std::vector<T> list_return(std::vector<T> vector1) {
    std::cout<<"Vector_Function"<<std::endl;
    std::vector<T> vector2;
    for (T i=0; i<vector1.size(); i++){
        vector2.push_back(vector1[i]);
    }
    return vector2;
}

template<typename T>
int tuple_return(std::vector<std::tuple<int,int,int,T>> vect_tup) {
    std::cout<<"Vec_Tup_Function"<<std::endl;
    for (int i=0; i<vect_tup.size(); i++){
        std::cout<<"["<< std::get<0>(vect_tup[i]) <<","<< std::get<1>(vect_tup[i]) <<","<< std::get<2>(vect_tup[i]) <<","<< std::get<3>(vect_tup[i]) <<"]"<< std::endl;
    }
    return 0;
}

// Original circuit_xyz_exp (backward compatible)
template<typename T1, typename T2>
std::vector<std::vector<T1>> circuit_xyz_exp(std::vector<std::tuple<int,int,int,T2>> tensor_vec, int no_sites) {
    Qubit tensor_sites = Qubit(no_sites);
    MPS tensor_mps = apply_gates3(tensor_vec, tensor_sites, no_sites, 1E-8);
    
    std::vector<std::vector<T1>> return_vec;
    auto start_itensor = std::chrono::high_resolution_clock::now();
    
    for (int i=0; i<no_sites; i++){
        std::vector<T1> xyz;
        tensor_mps.position(i+1);
        
        auto scalar_x = eltC((dag(prime(tensor_mps.A(i+1),"Site"))*tensor_sites.op("X_half",i+1)*tensor_mps.A(i+1))).real();
        auto scalar_y = eltC((dag(prime(tensor_mps.A(i+1),"Site"))*tensor_sites.op("Y_half",i+1)*tensor_mps.A(i+1))).real();
        auto scalar_z = eltC((dag(prime(tensor_mps.A(i+1),"Site"))*tensor_sites.op("Z_half",i+1)*tensor_mps.A(i+1))).real();
        
        xyz.push_back(scalar_x);
        xyz.push_back(scalar_y);
        xyz.push_back(scalar_z);
        return_vec.push_back(xyz);
    }
    
    auto end_itensor = std::chrono::high_resolution_clock::now();
    return return_vec;
}

// NEW: Enhanced feature extraction with entanglement metrics
template<typename T1, typename T2>
std::vector<std::vector<T1>> circuit_xyz_exp_enhanced(std::vector<std::tuple<int,int,int,T2>> tensor_vec, int no_sites, double cutoff = 1E-8) {
    Qubit tensor_sites = Qubit(no_sites);
    MPS tensor_mps = apply_gates3(tensor_vec, tensor_sites, no_sites, cutoff);
    
    std::vector<std::vector<T1>> return_vec;
    
    for (int i=0; i<no_sites; i++){
        std::vector<T1> features;
        tensor_mps.position(i+1);
        
        // Original XYZ expectation values
        auto scalar_x = eltC((dag(prime(tensor_mps.A(i+1),"Site"))*tensor_sites.op("X_half",i+1)*tensor_mps.A(i+1))).real();
        auto scalar_y = eltC((dag(prime(tensor_mps.A(i+1),"Site"))*tensor_sites.op("Y_half",i+1)*tensor_mps.A(i+1))).real();
        auto scalar_z = eltC((dag(prime(tensor_mps.A(i+1),"Site"))*tensor_sites.op("Z_half",i+1)*tensor_mps.A(i+1))).real();
        
        features.push_back(scalar_x);
        features.push_back(scalar_y);
        features.push_back(scalar_z);
        
        // Entanglement proxy: von Neumann entropy approximation via bond dimension
        if(i < no_sites-1) {
            auto bond_dim = commonIndex(tensor_mps(i+1), tensor_mps(i+2)).dim();
            features.push_back(static_cast<T1>(std::log(static_cast<double>(bond_dim))));
        } else {
            features.push_back(static_cast<T1>(0.0));  // No bond for last qubit
        }
        
        return_vec.push_back(features);
    }
    
    return return_vec;
}

// NEW: Extract correlation features between adjacent qubits
template<typename T1, typename T2>
std::vector<T1> circuit_correlations(std::vector<std::tuple<int,int,int,T2>> tensor_vec, int no_sites, double cutoff = 1E-8) {
    Qubit tensor_sites = Qubit(no_sites);
    MPS tensor_mps = apply_gates3(tensor_vec, tensor_sites, no_sites, cutoff);
    
    std::vector<T1> correlations;
    
    // Extract ZZ correlations between adjacent qubits
    for (int i=0; i<no_sites-1; i++){
        tensor_mps.position(i+1);
        
        auto op_z1 = tensor_sites.op("Z_half",i+1);
        auto op_z2 = tensor_sites.op("Z_half",i+2);
        
        auto wf = tensor_mps(i+1) * tensor_mps(i+2);
        auto wf_prime = prime(wf, "Site");
        auto op_combined = op_z1 * op_z2;
        
        auto corr = eltC((dag(wf_prime) * op_combined * wf)).real();
        correlations.push_back(corr);
    }
    
    return correlations;
}

// NEW: Batch processing for improved throughput
template<typename T1, typename T2>
std::vector<std::vector<std::vector<T1>>> circuit_xyz_exp_batch(
    std::vector<std::vector<std::tuple<int,int,int,T2>>> batch_circuits, 
    int no_sites,
    double cutoff = 1E-8) {
    
    // Reuse site indices across batch for efficiency
    Qubit tensor_sites = Qubit(no_sites);
    std::vector<std::vector<std::vector<T1>>> batch_results;
    batch_results.reserve(batch_circuits.size());
    
    for(const auto& tensor_vec : batch_circuits) {
        MPS tensor_mps = apply_gates3(tensor_vec, tensor_sites, no_sites, cutoff);
        std::vector<std::vector<T1>> return_vec;
        
        for (int i=0; i<no_sites; i++){
            std::vector<T1> xyz;
            tensor_mps.position(i+1);
            
            auto scalar_x = eltC((dag(prime(tensor_mps.A(i+1),"Site"))*tensor_sites.op("X_half",i+1)*tensor_mps.A(i+1))).real();
            auto scalar_y = eltC((dag(prime(tensor_mps.A(i+1),"Site"))*tensor_sites.op("Y_half",i+1)*tensor_mps.A(i+1))).real();
            auto scalar_z = eltC((dag(prime(tensor_mps.A(i+1),"Site"))*tensor_sites.op("Z_half",i+1)*tensor_mps.A(i+1))).real();
            
            xyz.push_back(scalar_x);
            xyz.push_back(scalar_y);
            xyz.push_back(scalar_z);
            return_vec.push_back(xyz);
        }
        batch_results.push_back(return_vec);
    }
    
    return batch_results;
}

// PyBind11 module definition with all functions
PYBIND11_MODULE(helloitensor, m) {
    m.doc() = "ITensor quantum feature encoding for fraud detection with LightGBM";
    
    // Original functions (backward compatible)
    m.def("add", &add<int>, "A function that adds two numbers");
    m.def("add", &add<float>, "A function that adds two floats");
    m.def("hello", &hello, "Hello function");
    m.def("vec_return", &list_return<int>, "Return input list as a vector");
    m.def("tuple_return", &tuple_return<int>, "Print vector of tuples (int)");
    m.def("tuple_return", &tuple_return<float>, "Print vector of tuples (float)");
    m.def("tuple_return", &tuple_return<double>, "Return vector of tuples (double)");
    
    // Original feature extraction (backward compatible)
    m.def("circuit_xyz_exp", &circuit_xyz_exp<double, float>, 
          "Extract single qubit XYZ expectation values. Returns list of num_qubit x 3 values.");
    
    // NEW: Enhanced feature extraction
    m.def("circuit_xyz_exp_enhanced", &circuit_xyz_exp_enhanced<double, float>, 
          "Enhanced feature extraction with XYZ + entanglement metrics. Returns num_qubit x 4 values.",
          pybind11::arg("tensor_vec"), 
          pybind11::arg("no_sites"), 
          pybind11::arg("cutoff") = 1E-8);
    
    // NEW: Correlation features
    m.def("circuit_correlations", &circuit_correlations<double, float>, 
          "Extract ZZ correlation features between adjacent qubits. Returns (num_qubit-1) values.",
          pybind11::arg("tensor_vec"), 
          pybind11::arg("no_sites"), 
          pybind11::arg("cutoff") = 1E-8);
    
    // NEW: Batch processing
    m.def("circuit_xyz_exp_batch", &circuit_xyz_exp_batch<double, float>, 
          "Batch processing for multiple circuits. Returns batch_size x num_qubit x 3 values.",
          pybind11::arg("batch_circuits"), 
          pybind11::arg("no_sites"), 
          pybind11::arg("cutoff") = 1E-8);
}


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

# Import all functions from enhanced helloitensor module
from helloitensor import (
    circuit_xyz_exp,  # Original function (backward compatible)
    circuit_xyz_exp_enhanced,  # NEW: Enhanced with entanglement
    circuit_correlations,  # NEW: Correlation features
    circuit_xyz_exp_batch  # NEW: Batch processing
)

# Pauli matrices for reference
X = np.array([[0,1],[1,0]])
Y = np.array([[0,-1j],[1j,0]])
I = np.array([[1,0],[0,1]])
Z = np.array([[1,0],[0,-1]])

class ProjectedQuantumFeatures:
    """Class that creates and stores a symbolic ansatz circuit and can be used to
    produce instances of the circuit U(x)|0> for given parameters.
    
    Attributes:
        ansatz_circ: The symbolic circuit to be used as ansatz.
        feature_symbol_list: The list of symbols in the circuit, each corresponding to a feature.
        svd_cutoff: SVD truncation cutoff for tensor network operations.
    """
    
    def __init__(
        self,
        num_features: int,
        reps: int,
        gamma: float,
        entanglement_map: list[tuple[int, int]],
        ansatz: str,
        hadamard_init: bool=True,
        svd_cutoff: float=1e-8  # NEW: Configurable cutoff parameter
    ):
        """Generate the ansatz circuit and store it. The circuit has as many symbols as qubits, which
        is also the same number of features in the data set. Multiple gates will use the same symbols;
        particularly, 1-qubit gates acting on qubit `i` all use the same symbol, and two qubit gates
        acting qubits `(i,j)` will use the symbols for feature `i` and feature `j`.
        
        Args:
            num_qubits: number of qubits is the number of features to be encoded.
            reps: number of times to repeat the layer of unitaries.
            gamma: hyper parameter in unitary to be optimized for overfitting.
            entanglement_map: pairs of qubits to be entangled together in the ansatz,
                for now limit entanglement only to two body terms
            hadamard_init: whether a layer of H gates should be applied to all qubits
                at the beginning of the circuit.
            svd_cutoff: SVD truncation cutoff for MPS operations (default: 1e-8).
        """
        self.one_q_symbol_list = []
        self.two_q_symbol_list = []
        self.ansatz_circ = Circuit(num_features)
        self.feature_symbol_list = [Symbol('f_'+str(i)) for i in range(num_features)]
        self.reps = reps
        self.gamma = gamma
        self.num_features = num_features
        self.hadamard_init = hadamard_init
        self.entanglement_map = entanglement_map
        self.svd_cutoff = svd_cutoff  # NEW: Store cutoff
        
        if ansatz == "hamiltonian":
            self.hamiltonian_ansatz()
        elif ansatz == "magic":
            self.magic_ansatz()
        else:
            raise RuntimeError("You have not entered a valid ansatz.")
    
    def circuit_for_data(self, feature_values: list[float]) -> Circuit:
        """Produce the circuit with its symbols being replaced by the given values."""
        if len(feature_values) != len(self.feature_symbol_list):
            raise RuntimeError(
                f"The number of values ({len(feature_values)}) must match "
                f"the number of symbols ({len(self.feature_symbol_list)})."
            )
        
        # NEW: Input validation for production use
        if not all(np.isfinite(val) for val in feature_values):
            raise ValueError("All feature values must be finite numbers.")
        
        symbol_map = {symb: val for symb, val in zip(self.feature_symbol_list, feature_values)}
        the_circuit = self.ansatz_circ.copy()
        the_circuit.symbol_substitution(symbol_map)
        return the_circuit
    
    def circuit_to_list(self, circuit):
        """Convert pytket circuit to gate list for ITensor processing."""
        gates = []
        
        for gate in circuit.get_commands():
            if gate.op.type == OpType.H:
                gates.append([0,gate.qubits[0].index[0],-1, 0])
            elif gate.op.type == OpType.Rx:
                gates.append([1,gate.qubits[0].index[0],-1, gate.op.params[0]])
            elif gate.op.type == OpType.Rz:
                gates.append([2,gate.qubits[0].index[0],-1, gate.op.params[0]])
            elif gate.op.type == OpType.ZZPhase:
                gates.append([4,gate.qubits[0].index[0],gate.qubits[1].index[0], gate.op.params[0]])
            elif gate.op.type == OpType.XXPhase:
                gates.append([3,gate.qubits[0].index[0],gate.qubits[1].index[0], gate.op.params[0]])
            elif gate.op.type == OpType.SWAP:
                gates.append([5,gate.qubits[0].index[0],gate.qubits[1].index[0], 0])
            elif gate.op.type == OpType.T:
                gates.append([6,gate.qubits[0].index[0],-1, 0])
            elif gate.op.type == OpType.CZ:
                gates.append([7,gate.qubits[0].index[0],gate.qubits[1].index[0], 0])
            else:
                raise Exception(f"Unknown gate {gate.op.type}")
        
        return gates
    
    # NEW: Enhanced feature extraction method
    def extract_enhanced_features(self, circuit, n_qubits: int) -> np.ndarray:
        """Extract enhanced quantum features including entanglement proxies.
        
        Args:
            circuit: The quantum circuit to process
            n_qubits: Number of qubits in the circuit
            
        Returns:
            Enhanced feature vector with XYZ expectation values and entanglement metrics
        """
        circ_gates = self.circuit_to_list(circuit)
        enhanced_features = circuit_xyz_exp_enhanced(circ_gates, n_qubits, self.svd_cutoff)
        return np.asarray(enhanced_features).flatten()
    
    # NEW: Correlation feature extraction
    def extract_correlation_features(self, circuit, n_qubits: int) -> np.ndarray:
        """Extract quantum correlation features between adjacent qubits.
        
        Args:
            circuit: The quantum circuit to process
            n_qubits: Number of qubits in the circuit
            
        Returns:
            Correlation feature vector of length (n_qubits - 1)
        """
        circ_gates = self.circuit_to_list(circuit)
        correlations = circuit_correlations(circ_gates, n_qubits, self.svd_cutoff)
        return np.asarray(correlations)
    
    def hamiltonian_ansatz(self):
        """Build the hamiltonian circuit ansatz."""
        print('Built the hamiltonian circuit')
        
        if self.hadamard_init:
            for i in range(self.num_features):
                self.ansatz_circ.H(i)
        
        # Apply TKET routing to compile circuit to line architecture
        for _ in range(self.reps):
            for i in range(self.num_features):
                exponent = (1/np.pi)*self.gamma*self.feature_symbol_list[i]
                self.ansatz_circ.Rz(exponent, i)
            
            for (q0, q1) in self.entanglement_map:
                symb0 = self.feature_symbol_list[q0]
                symb1 = self.feature_symbol_list[q1]
                exponent = self.gamma*self.gamma*(1 - symb0)*(1 - symb1)
                self.ansatz_circ.XXPhase(exponent, q0, q1)
        
        cu = CompilationUnit(self.ansatz_circ)
        architecture = Architecture(
            [(i, i + 1) for i in range(self.ansatz_circ.n_qubits - 1)]
        )
        DefaultMappingPass(architecture).apply(cu)
        self.ansatz_circ = cu.circuit
        Transform.DecomposeBRIDGE().apply(self.ansatz_circ)
        
        return 0
    
    def magic_ansatz(self):
        """Build the magic circuit ansatz."""
        print('Built the magic circuit')
        
        for _ in range(self.reps):
            for q in range(self.num_features):
                self.ansatz_circ.H(q)
                self.ansatz_circ.T(q)
            
            for (q0, q1) in self.entanglement_map:
                self.ansatz_circ.CZ(q0,q1)
                
                if q1 == self.num_features - 1 or q1 == self.num_features-2:
                    for q in range(self.num_features):
                        self.ansatz_circ.H(q)
                        self.ansatz_circ.T(q)
            
            for q in range(self.num_features):
                self.ansatz_circ.Rz(self.feature_symbol_list[q],q)
        
        return 0


def build_qf_matrix(
    mpi_comm,
    ansatz: ProjectedQuantumFeatures,
    X,
    info_file=None,
    cpu_max_mem=6,
    use_enhanced_features: bool = False,  # NEW: Enhanced features flag
    include_correlations: bool = False,   # NEW: Correlation features flag
    use_batch_processing: bool = False,   # NEW: Batch processing flag
) -> np.ndarray:
    """Calculation of entries of the kernel matrix with enhanced features.
    
    Args:
        mpi_comm: The MPI communicator created by the caller of this function.
        ansatz: a symbolic circuit describing the ansatz.
        X: A 2D array where `X[i, :]` corresponds to the i-th data point.
        info_file: The name of the file where to save performance information.
        cpu_max_mem: The number of GB available in each CPU. Defaults to 6 GB.
        use_enhanced_features: If True, extract additional entanglement features (4 per qubit).
        include_correlations: If True, add two-qubit correlation features.
        use_batch_processing: If True, use batch ITensor processing for efficiency.
    
    Returns:
        A projected feature matrix of dimensions `len(X)` x `feature_dim`.
    """
    n_qubits = ansatz.ansatz_circ.n_qubits
    
    # MPI information
    root = 0
    rank = mpi_comm.Get_rank()
    n_procs = mpi_comm.Get_size()
    entries_per_chunk = int(np.ceil(len(X) / n_procs))
    
    # Dictionary to keep track of profiling information
    if rank == root:
        profiling_dict = dict()
        profiling_dict["lenX"] = (len(X), "entries")
        start_time = MPI.Wtime()
        print(f"Using enhanced features: {use_enhanced_features}")
        print(f"Including correlations: {include_correlations}")
        print(f"Batch processing: {use_batch_processing}")
        print(f"SVD cutoff: {ansatz.svd_cutoff}")
        sys.stdout.flush()
    
    # Determine feature dimensions
    if use_enhanced_features:
        feature_dim = n_qubits * 4  # X, Y, Z, + entanglement per qubit
    else:
        feature_dim = n_qubits * 3  # X, Y, Z per qubit
    
    if include_correlations:
        feature_dim += (n_qubits - 1)  # Add correlation features
    
    # Report configuration
    if rank == root:
        print(f"\nContracting MPS from dataset with {len(X)} samples...")
        print(f"Feature dimension: {feature_dim}")
        sys.stdout.flush()
    
    # Batch processing optimization
    if use_batch_processing:
        exp_x_chunk = []
        exp_x_time = []
        
        # Prepare batch of circuits
        batch_circuits = []
        batch_indices = []
        
        for k in range(entries_per_chunk):
            offset = rank * entries_per_chunk
            if k + offset < len(X):
                circ = ansatz.circuit_for_data(X[k+offset, :])
                batch_circuits.append(ansatz.circuit_to_list(circ))
                batch_indices.append(k)
        
        if len(batch_circuits) > 0:
            time0 = MPI.Wtime()
            
            # Batch process all circuits at once
            if use_enhanced_features:
                batch_results = []
                for circ_gates in batch_circuits:
                    enhanced = circuit_xyz_exp_enhanced(circ_gates, n_qubits, ansatz.svd_cutoff)
                    features = np.asarray(enhanced).flatten()
                    
                    if include_correlations:
                        corr = circuit_correlations(circ_gates, n_qubits, ansatz.svd_cutoff)
                        features = np.concatenate([features, np.asarray(corr)])
                    
                    batch_results.append(features)
            else:
                # Use standard batch processing
                batch_results = circuit_xyz_exp_batch(batch_circuits, n_qubits, ansatz.svd_cutoff)
                batch_results = [np.asarray(res).flatten() for res in batch_results]
                
                if include_correlations:
                    for i, circ_gates in enumerate(batch_circuits):
                        corr = circuit_correlations(circ_gates, n_qubits, ansatz.svd_cutoff)
                        batch_results[i] = np.concatenate([batch_results[i], np.asarray(corr)])
            
            exp_x_time.append(MPI.Wtime() - time0)
            exp_x_chunk.extend(batch_results)
            
            if rank == root:
                print(f"Batch processed {len(batch_results)} circuits in {exp_x_time[-1]:.2f} seconds")
                sys.stdout.flush()
    
    else:
        # Original sequential processing with enhancements
        exp_x_chunk = []
        exp_x_time = []
        progress_bar = 0
        progress_tick = int(np.ceil(entries_per_chunk / 10))
        
        for k in range(entries_per_chunk):
            offset = rank * entries_per_chunk
            if k + offset < len(X):
                circ = ansatz.circuit_for_data(X[k+offset, :])
            else:
                circ = None
            
            if circ is not None:
                time0 = MPI.Wtime()
                circ_gates = ansatz.circuit_to_list(circ)
                
                # Choose feature extraction method
                if use_enhanced_features:
                    features = circuit_xyz_exp_enhanced(circ_gates, n_qubits, ansatz.svd_cutoff)
                    features = np.asarray(features).flatten()
                else:
                    features = circuit_xyz_exp(circ_gates, n_qubits)
                    features = np.asarray(features).flatten()
                
                # Add correlation features if requested
                if include_correlations:
                    corr = circuit_correlations(circ_gates, n_qubits, ansatz.svd_cutoff)
                    features = np.concatenate([features, np.asarray(corr)])
                
                exp_x_time.append(MPI.Wtime() - time0)
                exp_x_chunk.append(features)
                
                if rank == root and progress_bar * progress_tick < k:
                    print(f"{progress_bar*10}%")
                    sys.stdout.flush()
                    progress_bar += 1
    
    # Reshape and aggregate results
    if rank == n_procs-1:
        exp_x_chunk = np.asarray(exp_x_chunk).reshape((len(X) - (rank*entries_per_chunk), feature_dim))
    else:
        exp_x_chunk = np.asarray(exp_x_chunk).reshape((entries_per_chunk, feature_dim))
    
    ind1 = entries_per_chunk * rank
    if rank != n_procs - 1:
        ind2 = entries_per_chunk * (rank + 1)
    else:
        ind2 = len(X)
    
    projected_features = np.zeros(shape=(len(X), feature_dim))
    projected_features[ind1:ind2, :] = exp_x_chunk
    
    # Reporting
    if rank == root:
        print("100%")
        duration = sum(exp_x_time)
        print(f"[Rank 0] MPS of chunk X contracted. Time taken: {round(duration,2)} seconds.")
        profiling_dict["r0_circ_sim"] = [duration, "seconds"]
        
        if len(exp_x_time) > 0:
            average = mean(exp_x_time)
            print(f"\tAverage time per MPS contraction: {round(average,4)} seconds.")
            profiling_dict["avg_circ_sim"] = [average, "seconds"]
            profiling_dict["median_circ_sim"] = [median(exp_x_time), "seconds"]
        
        print(f"Feature dimension: {feature_dim}")
        print("\nFinished contracting all MPS.")
        sys.stdout.flush()
    
    projected_features = mpi_comm.reduce(projected_features, op=MPI.SUM, root=root)
    
    return projected_features


# NEW: Helper function for fraud detection integration with LightGBM
def extract_fraud_detection_features(
    ansatz: ProjectedQuantumFeatures,
    X: np.ndarray,
    feature_type: str = "enhanced"  # "basic", "enhanced", or "full"
) -> np.ndarray:
    """Extract quantum features optimized for fraud detection with LightGBM.
    
    Args:
        ansatz: Configured ProjectedQuantumFeatures instance
        X: Input feature array (n_samples, n_features)
        feature_type: 
            - "basic": XYZ expectation values only (3*n_qubits features)
            - "enhanced": XYZ + entanglement (4*n_qubits features)
            - "full": enhanced + correlations (4*n_qubits + n_qubits-1 features)
    
    Returns:
        Quantum-encoded feature matrix for LightGBM
    """
    n_qubits = ansatz.num_features
    n_samples = len(X)
    
    if feature_type == "basic":
        feature_dim = n_qubits * 3
    elif feature_type == "enhanced":
        feature_dim = n_qubits * 4
    elif feature_type == "full":
        feature_dim = n_qubits * 4 + (n_qubits - 1)
    else:
        raise ValueError(f"Unknown feature_type: {feature_type}")
    
    quantum_features = np.zeros((n_samples, feature_dim))
    
    print(f"Extracting {feature_type} quantum features for {n_samples} samples...")
    
    for i in range(n_samples):
        circ = ansatz.circuit_for_data(X[i, :])
        circ_gates = ansatz.circuit_to_list(circ)
        
        if feature_type == "basic":
            features = circuit_xyz_exp(circ_gates, n_qubits)
            quantum_features[i, :] = np.asarray(features).flatten()
        
        elif feature_type == "enhanced":
            features = circuit_xyz_exp_enhanced(circ_gates, n_qubits, ansatz.svd_cutoff)
            quantum_features[i, :] = np.asarray(features).flatten()
        
        elif feature_type == "full":
            features = circuit_xyz_exp_enhanced(circ_gates, n_qubits, ansatz.svd_cutoff)
            corr = circuit_correlations(circ_gates, n_qubits, ansatz.svd_cutoff)
            quantum_features[i, :] = np.concatenate([
                np.asarray(features).flatten(),
                np.asarray(corr)
            ])
        
        # Progress reporting
        if (i + 1) % max(1, n_samples // 10) == 0:
            print(f"  Processed {i+1}/{n_samples} samples ({100*(i+1)//n_samples}%)")
    
    print(f"Completed feature extraction: {quantum_features.shape}")
    return quantum_features
