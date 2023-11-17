from typing import List, Tuple

import numpy as np
import os
from spins import fdfd_tools
from spins import gridlock
from spins.invdes import problem
from spins.invdes.problem_graph import creator_em
from spins.invdes.problem_graph import grid_utils
from spins.invdes.problem_graph import optplan
from spins.invdes.problem_graph import workspace
from spins.invdes.problem_graph.simspace import SimulationSpace
import scipy.sparse as sparse
from schematics import types

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import FormatStrFormatter

# Have a single shared direct solver object because we need to use
# multiprocessing to actually parallelize the solve.
from spins.fdfd_solvers import local_matrix_solvers
DIRECT_SOLVER = local_matrix_solvers.MultiprocessingSolver(
    local_matrix_solvers.DirectSolver())

class PowerTransmission(optplan.Function):
    """Defines a function that measures amount of power passing through plane.

    The amount of power is computed by summing the Poynting vector across
    the desired plane.

    Attributes:
        field: The simulation field to use.
        center: Center of plane to compute power.
        extents: Extents of the plane over which to compute power.
        normal: Normal direction of the plane. This determines the sign of the
            power flowing through the plane.
    """
    type = optplan.define_schema_type("function.poynting.plane_power")
    field = optplan.ReferenceType(optplan.FdfdSimulation)
    center = optplan.vec3d()
    extents = optplan.vec3d()
    normal = optplan.vec3d()


class PowerTransmissionFunction(problem.OptimizationFunction):
    """Evalutes power passing through a plane.

    The power is computed by summing over the Poynting vector on the plane.
    Specifically, the power is `sum(0.5 * real(E x H*))[axis]` where `axis`
    indicates the component of the Poynting vector to use.

    Currently the plane must be an axis-aligned plane and the permeability
    is assumed to be unity.

    Note that the calculation of Poynting vector does not take into account
    PMLs.
    """

    def __init__(
            self,
            field: creator_em.FdfdSimulation,
            simspace: SimulationSpace,
            wlen: float,
            plane_slice: Tuple[slice, slice, slice],
            axis: int,
            polarity: int,
    ) -> None:
        """Creates a new power plane function.

        Args:
            field: The `FdfdSimulation` to use to calculate power.
            simspace: Simulation space corresponding to the field.
            wlen: Wavelength of the field.
            plane_slice: Represents the locations of the field that are part
                of the plane.
            axis: Which Poynting vector field component to use.
            polarity: Indicates directionality of the plane.
        """
        super().__init__(field)
        self._omega = 2 * np.pi / wlen
        self._dxes = simspace.dxes
        self._plane_slice = plane_slice
        self._axis = axis
        self._polarity = polarity
        
        dx_e = simspace.dxes[0]
        self.dx = dx_e[0][0]
        self.dy = dx_e[1][0]
        self.dz = dx_e[2][0]
        self.dim = simspace.dims          

        # Precompute any operations that can be computed without knowledge of
        # the field.
        self._op_e2h = fdfd_tools.operators.e2h(self._omega, self._dxes)
        self._op_curl_e = fdfd_tools.operators.curl_e(self._dxes)

        # Create a filter that sets a 1 in every position that is included in
        # the computation of the Poynting vector.
        filter_grid = [np.zeros(simspace.dims) for i in range(3)]
        filter_grid[self._axis][tuple(self._plane_slice)] = 1
        self._filter_vec = fdfd_tools.vec(filter_grid)

    # TODO(logansu): Make it work for arbitrary mu.
    def eval(self, input_val: List[np.ndarray]) -> np.ndarray:
        efield = input_val[0]

        hfield = self._op_e2h @ efield
        op_e_cross = fdfd_tools.operators.poynting_chew_e_cross(
            efield, self._dxes)
        poynting = 0.5 * np.real(op_e_cross @ np.conj(hfield))
        return self._polarity * np.sum(poynting * self._filter_vec)

    def grad(self, input_vals: List[np.ndarray],
             grad_val: np.ndarray) -> List[np.ndarray]:

        efield = input_vals[0]

        hfield = self._op_e2h @ efield
        
        op_e_cross = fdfd_tools.operators.poynting_chew_e_cross(
            efield, self._dxes)
        op_h_cross = fdfd_tools.operators.poynting_chew_h_cross(
            hfield, self._dxes)

        # Compute the gradient across all of space.
        mu = 1
        # After add the 2* the gradient error was minimized from E-1 to E-6.
#        grad_mat = -0.25 * (1 / (1j * self._omega * mu) * op_e_cross.conj() *
#                            self._op_curl_e + op_h_cross.conj())        
        grad_mat = -2*0.25 * (1 / (1j * self._omega * mu) * op_e_cross.conj() *
                            self._op_curl_e + op_h_cross.conj())
        dF = [grad_val * self._polarity * self._filter_vec.T @ grad_mat]
        return dF


@optplan.register_node(PowerTransmission)
def create_power_transmission_function(
        params: PowerTransmission,
        context: workspace.Workspace) -> PowerTransmissionFunction:
    simspace = context.get_object(params.field.simulation_space)
    return PowerTransmissionFunction(
        field=context.get_object(params.field),
        simspace=simspace,
        wlen=params.field.wavelength,
        plane_slice=grid_utils.create_region_slices(
            simspace.edge_coords, params.center, params.extents),
        axis=gridlock.axisvec2axis(params.normal),
        polarity=gridlock.axisvec2polarity(params.normal))


###############################################################################      
# Purcell Factor FOM
# Introduced by E.G.Melo in 11/2021        
###############################################################################
        
class Purcell(optplan.Function):
    """Defines a function that measures amount of power passing through plane.

    The amount of power is computed by summing the Poynting vector across
    the desired plane.

    Attributes:
        sim: The simulation object to use.
        center: Center of plane to compute power.
        extents: Extents of the plane over which to compute power.
        normal: Normal direction of the plane. This determines the sign of the
            power flowing through the plane.
    """
    type = optplan.define_schema_type("function.poynting.purcell_region")
    sim = optplan.ReferenceType(optplan.FdfdSimulation)
    sim_bulk = optplan.ReferenceType(optplan.FdfdSimulation)


class PurcellFunction(problem.OptimizationFunction):
    """Evalutes the local density of states for a dipole source and calculates the
       inverse of the LDOS as it is necessary for minimization problems.
    """

    def __init__(
            self,
            sim: creator_em.FdfdSimulation,
            source: np.ndarray,
            dV: float,
            bulk_ldos: float,
    ) -> None:
        """Creates a new power plane function.

        Args:
            sim: The `FdfdSimulation` to use to calculate fields.
            source: Sources matrix of the simulation.
            dV: grid volume.
            bulk_ldos: LDOS calculated for a bulk medium.
        """
        super().__init__(sim)
        self._conjJ = np.conj(fdfd_tools.vec(source))
        self._df_num = 2*(bulk_ldos*dV*3/np.pi)*self._conjJ
        self._dV = dV
        self.bulk_ldos = bulk_ldos
        
    def eval(self, input_val: List[np.ndarray]) -> np.ndarray:
        """ LDOS calculated as:
            X. Liang and S. G. Johnson, “Formulation for scalable optimization of microcavities via the frequency-averaged
            local density of states,” Opt. Express 21, 30812–30841 (2013).

        Parameters
        ----------
        input_val : List[np.ndarray]
            Electric fields calculated by FDFD.

        Returns the inverse value of LDOS as it is necessary for minimization problems: 1/LDOS
        -------
        obj : np.ndarray
            1/LDOS.
        """
        E = input_val[0]

        ldos = -(self._dV*6/np.pi)*np.sum(np.real(E * self._conjJ))  
        obj = self.bulk_ldos/ldos        
        return obj

    def grad(self, input_vals: List[np.ndarray],
             grad_val: np.ndarray) -> List[np.ndarray]:
        """ Calculates the Gradient of 1/LDOS 

        Parameters
        ----------
        input_val : List[np.ndarray]
            Forward fields calculated by FDFD.

        """
        
        E = np.asarray(input_vals[0])
        dFdx = np.zeros_like(E)
        df_den = (np.sum((self._dV*6/np.pi)*np.real(E * self._conjJ)))**2
        dFdx = self._df_num/df_den
        
        return [grad_val * dFdx]


@optplan.register_node(Purcell)
def create_purcell_function(
        params: Purcell,
        work: workspace.Workspace) -> PurcellFunction:
    
    """Creates a `FdfdSimulation` object."""
    simspace = work.get_object(params.sim.simulation_space)
    
    if type(params.sim.source.phase) is list:
        source = creator_em.DipoleSourceOffAxis(params.sim.source)
    else:
        source = creator_em.DipoleSource(params.sim.source)
    solver = creator_em._create_solver(params.sim.solver, simspace)
    J = source(simspace, params.sim.epsilon.wavelength, solver)
    
    dx_e = simspace.dxes[0]
    dx = dx_e[0][0]
    dy = dx_e[1][0]
    dz = dx_e[2][0]
    dV = dx*dy*dz
    
    simspace_bulk = work.get_object(params.sim_bulk.simulation_space)
    bulk_ldos = calc_bulk_ldos(params.sim_bulk, simspace_bulk, dV)
    
    return PurcellFunction(
        sim=work.get_object(params.sim),
        source=J,
        dV=dV,
        bulk_ldos=bulk_ldos)        


def calc_bulk_ldos(sim_bulk: optplan.FdfdSimulation, simspace_bulk: SimulationSpace, dV: float):
    """Calculates the LDOS of a bulk medium."""

    wl = sim_bulk.epsilon.wavelength
    if type(sim_bulk.source.phase) is list:
        source = creator_em.DipoleSourceOffAxis(sim_bulk.source)
    else:
        source = creator_em.DipoleSource(sim_bulk.source)    

    solver = creator_em._create_solver(sim_bulk.solver, simspace_bulk)
    J_bulk = source(simspace_bulk, wl, solver)                  

    eps_grid = simspace_bulk(wl).eps_bg.grids
    eps = problem.Constant(fdfd_tools.vec(eps_grid))

    sim = creator_em.FdfdSimulation(
        eps=eps,
        solver=solver,
        wlen=wl,
        source=fdfd_tools.vec(J_bulk),
        simspace=simspace_bulk,
    )

    E = fdfd_tools.vec(fdfd_tools.unvec(problem.graph_executor.eval_fun(sim, None), eps_grid[0].shape))
    conjJ = np.conj(fdfd_tools.vec(J_bulk))
    
    # Calculate LDOS.
    ldos = -(dV*6/np.pi)*np.sum(np.real(E * conjJ))     
    return ldos

###############################################################################      
# Source Power
# Introduced by E.G.Melo in 11/2021        
###############################################################################

class SourcePower(optplan.Function):
    """Defines a function that measures the source power.

    The amount of power is computed by summing the Poynting vector across
    the planes of a box.

    Attributes:
        field: The simulation field to use.
        center: Center position of the source power box.
        box_size: Dimensions of the box over which to compute power.
    """
    type = optplan.define_schema_type("function.poynting.source_power")
    field = optplan.ReferenceType(optplan.FdfdSimulation)
    center = types.ListType(types.FloatType())
    box_size = types.ListType(types.FloatType())
    
class SourcePowerFunction(problem.OptimizationFunction):
    """Represents an optimization function for source power."""

    def __init__(self, simulation: problem.OptimizationFunction,
                simspace: SimulationSpace,
                wlen: float,
                center: list,
                box_size: list):
        """Constructs the objective.

        Args:

        """
        super().__init__(simulation)
        
        self._sim = simulation
        self._simspace = simspace
        self._wlen = wlen
        self._src_center = center
        self._src_box = box_size

        # 2D or 3D source box.        
        self.box_dim = len(self._src_box)
        # Simulation parameters. 
        self._omega = 2 * np.pi / wlen
        self._dxes = simspace.dxes   
        
        dx_e = simspace.dxes[0]
        self.dx = dx_e[0][0]
        self.dy = dx_e[1][0]
        self.dz = dx_e[2][0]        
        
        # Precompute any operations that can be computed without knowledge of
        # the field.
        self._op_e2h = fdfd_tools.operators.e2h(self._omega, self._dxes)
        self._op_curl_e = fdfd_tools.operators.curl_e(self._dxes)
        self._filter_vec = [] 
        self._polarity = [] 

        # Minimum x-plane        
        plane_slice = grid_utils.create_region_slices(simspace.edge_coords, 
                                                      [self._src_center[0] - self._src_box[0]/2, self._src_center[1], self._src_center[2]], 
                                                      [self.dx, self._src_box[1], self._src_box[2] if self.box_dim == 3 else self.dz])
        axis = gridlock.axisvec2axis([1, 0, 0])
        filter_grid = [np.zeros(simspace.dims) for i in range(3)]
        filter_grid[axis][tuple(plane_slice)] = 1
        self._filter_vec.append(fdfd_tools.vec(filter_grid))        
        self._polarity.append(gridlock.axisvec2polarity([-1, 0, 0]))
        
        # Maximum x-plane        
        plane_slice = grid_utils.create_region_slices(simspace.edge_coords, 
                                                      [self._src_center[0] + self._src_box[0]/2, self._src_center[1], self._src_center[2]], 
                                                      [self.dx, self._src_box[1], self._src_box[2] if self.box_dim == 3 else self.dz])
        axis = gridlock.axisvec2axis([1, 0, 0])
        filter_grid = [np.zeros(simspace.dims) for i in range(3)]
        filter_grid[axis][tuple(plane_slice)] = 1
        self._filter_vec.append(fdfd_tools.vec(filter_grid))
        self._polarity.append(gridlock.axisvec2polarity([1, 0, 0]))
        
        # Minimum y-plane        
        plane_slice = grid_utils.create_region_slices(simspace.edge_coords, 
                                                      [self._src_center[0], self._src_center[1] - self._src_box[1]/2, self._src_center[2]], 
                                                      [self._src_box[0], self.dy, self._src_box[2] if self.box_dim == 3 else self.dz])
        axis = gridlock.axisvec2axis([0, 1, 0])
        filter_grid = [np.zeros(simspace.dims) for i in range(3)]
        filter_grid[axis][tuple(plane_slice)] = 1
        self._filter_vec.append(fdfd_tools.vec(filter_grid))   
        self._polarity.append(gridlock.axisvec2polarity([0, -1, 0]))
        
        # Maximum y-plane        
        plane_slice = grid_utils.create_region_slices(simspace.edge_coords, 
                                                      [self._src_center[0], self._src_center[1] + self._src_box[1]/2, self._src_center[2]], 
                                                      [self._src_box[0], self.dy, self._src_box[2] if self.box_dim == 3 else self.dz])
        axis = gridlock.axisvec2axis([0, 1, 0])
        filter_grid = [np.zeros(simspace.dims) for i in range(3)]
        filter_grid[axis][tuple(plane_slice)] = 1
        self._filter_vec.append(fdfd_tools.vec(filter_grid))  
        self._polarity.append(gridlock.axisvec2polarity([0, 1, 0]))
        
        if self.box_dim == 3:
            # Minimum z-plane        
            plane_slice = grid_utils.create_region_slices(simspace.edge_coords, 
                                                          [self._src_center[0], self._src_center[1], self._src_center[2] - self._src_box[2]/2], 
                                                          [self._src_box[0], self._src_box[1], self.dz])
            axis = gridlock.axisvec2axis([0, 0, 1])
            filter_grid = [np.zeros(simspace.dims) for i in range(3)]
            filter_grid[axis][tuple(plane_slice)] = 1
            self._filter_vec.append(fdfd_tools.vec(filter_grid))
            self._polarity.append(gridlock.axisvec2polarity([0, 0, -1]))
            
            # Maximum z-plane        
            plane_slice = grid_utils.create_region_slices(simspace.edge_coords, 
                                                          [self._src_center[0], self._src_center[1], self._src_center[2] + self._src_box[2]/2], 
                                                          [self._src_box[0], self._src_box[1], self.dz])
            axis = gridlock.axisvec2axis([0, 0, 1])
            filter_grid = [np.zeros(simspace.dims) for i in range(3)]
            filter_grid[axis][tuple(plane_slice)] = 1
            self._filter_vec.append(fdfd_tools.vec(filter_grid))
            self._polarity.append(gridlock.axisvec2polarity([0, 0, 1]))            

    def eval(self, input_vals: List[np.ndarray]) -> np.ndarray:
        """Returns the output of the function.

        Args: List[np.ndarray]
            Electric fields calculated by FDFD.

        Returns: np.ndarray
            source power calculating by the integration of the Poynting vector.
        """
        E = input_vals[0]
        H = self._op_e2h @ E
        op_e_cross = fdfd_tools.operators.poynting_chew_e_cross(E, self._dxes)
        poynting = 0.5 * np.real(op_e_cross @ np.conj(H))
      
        src_power = 0
        for s in range(self.box_dim*2):
            src_power += self._polarity[s] * np.sum(poynting * self._filter_vec[s])     

        return src_power
    

    def grad(self, input_vals: List[np.ndarray],
            grad_val: np.ndarray) -> List[np.ndarray]:
        """Returns the gradient of the function.
        Args:
            input_vals: List of the input values.
            grad_val: Gradient of the output.
        Returns:
            gradient.
        """
        E = input_vals[0]
        H = self._op_e2h @ E
        op_e_cross = fdfd_tools.operators.poynting_chew_e_cross(E, self._dxes)
        op_h_cross = fdfd_tools.operators.poynting_chew_h_cross(H, self._dxes)
        # Compute the gradient across all of space.
        mu = 1
        dP = -0.50 * (1/(1j * self._omega * mu) * op_e_cross.conj() * self._op_curl_e + op_h_cross.conj())        
      
        dF = np.asarray(self._polarity[0] * self._filter_vec[0].T @ dP)
        dF += np.asarray(self._polarity[1] * self._filter_vec[1].T @ dP)
        dF += np.asarray(self._polarity[2] * self._filter_vec[2].T @ dP)
        dF += np.asarray(self._polarity[3] * self._filter_vec[3].T @ dP)        
        if self.box_dim == 3:        
            dF += np.asarray(self._polarity[4] * self._filter_vec[4].T @ dP)
            dF += np.asarray(self._polarity[5] * self._filter_vec[5].T @ dP)

        return [grad_val * dF]        

    def __str__(self):
        return "Source Power({})".format(self._sim)
    
@optplan.register_node(SourcePower)
def create_source_power_function(params: SourcePower,
                                 work: workspace.Workspace) -> SourcePowerFunction:
    simspace = work.get_object(params.field.simulation_space)
    wlen = params.field.wavelength
    return SourcePowerFunction(simulation=work.get_object(params.field),
                                      simspace=simspace,
                                      wlen=wlen,
                                      center=params.center,
                                      box_size=params.box_size)


###############################################################################      
# Energy Fabrication Constraint FOM
# Introduced by E.G.Melo in 06/2022        
###############################################################################
class EnergyConstraint(optplan.Function):
    """Defines a function that measures amount of energy on the cladding with 
    respect to the total energy.

    Attributes:
        field: The simulation field to use.
        eps: The dielectric distribution.   
        n_core: refractive index of the core material.
        n_clad: refractive index of the cladding material.
    """
    type = optplan.define_schema_type("function.poynting.energy_constraint")
    field = optplan.ReferenceType(optplan.FdfdSimulation)
    epsilon = optplan.ReferenceType(optplan.Function)
    n_core = types.FloatType()
    n_clad = types.FloatType()
    center = optplan.vec3d()
    extents = optplan.vec3d()


class EnergyConstraintFunction(problem.OptimizationFunction):
    """Evalutes the energy on the cladding with respect to the total energy.
    """

    def __init__(
            self,
            field: creator_em.FdfdSimulation,
            eps: creator_em.Epsilon,
            source: np.ndarray,            
            n_core: float,
            n_clad: float,
            simspace: SimulationSpace,
            plane_slice: Tuple[slice, slice, slice],            
    ) -> None:
        """Creates a new energy constraint function.

        Args:
            field: The simulation field to use.
            eps: The dielectric distribution. 
            source: source of the simulation.
            n_core: refractive index of the core material.
            n_clad: refractive index of the cladding material. 
            simspace: simulation space.
            plane_slice: region to compute the energy constraint.
        """
        super().__init__([field, eps])
        self._ncore = n_core
        self._nclad = n_clad
        # Create a filter that sets a 1 in every position that is included in
        # the computation of the energy fabrication contraint.
        filter_grid = [np.zeros(simspace.dims) for i in range(3)]
        filter_grid[0][tuple(plane_slice)] = 1
        filter_grid[1][tuple(plane_slice)] = 1
        filter_grid[2][tuple(plane_slice)] = 1
        #filter_grid = [np.ones(simspace.dims) for i in range(3)]
        filter_grid[0][source[0] > 0.001] = 0
        filter_grid[1][source[1] > 0.001] = 0
        filter_grid[2][source[2] > 0.001] = 0
        self._filter_vec = fdfd_tools.vec(filter_grid)        

    def eval(self, input_val: List[np.ndarray]) -> np.ndarray:
        """ Calculated as:
            GUOWU ZHANG et al, “Topological inverse design of nanophotonic
            devices with energy constraint,” Opt. Express 29(8), 12681, (2021).
        """
        efield = input_val[0] * self._filter_vec
        efield = efield.reshape(3, int(efield.size/3))
        eps = input_val[1] * self._filter_vec
        eps = np.real(eps.reshape(3, int(eps.size/3)))  
        eps_core = self._ncore**2
        eps_clad = self._nclad**2
        
        e_int = np.real(efield[0]*np.conj(efield[0]) + efield[1]*np.conj(efield[1]) + efield[2]*np.conj(efield[2]))
        
        rho = (1.0 - ((eps[0] - eps_clad)/(eps_core - eps_clad)))
        num = np.sum(rho*0.5*eps[0]*e_int)                                     
        den = np.sum(0.5*eps[0]*e_int)
        
        obj = num/den
        
        return obj

    def grad(self, input_vals: List[np.ndarray],
              grad_val: np.ndarray) -> List[np.ndarray]:

        efield = input_vals[0] * self._filter_vec
        efield = efield.reshape(3, int(efield.size/3))
        eps = input_vals[1] * self._filter_vec   
        eps = np.real(eps.reshape(3, int(eps.size/3)))
        eps_core = self._ncore**2
        eps_clad = self._nclad**2
        
        e_int = np.real(efield[0]*np.conj(efield[0]) + efield[1]*np.conj(efield[1]) + efield[2]*np.conj(efield[2]))   
        
        rho = (1.0 - ((eps - eps_clad)/(eps_core - eps_clad)))
        
        fE = np.sum(rho[0]*0.5*eps[0]*e_int)
        gE = np.sum(0.5*eps[0]*e_int)        
        
        dfE = rho*0.5*eps*(np.conj(efield))
        dgE = 0.5*eps*(np.conj(efield))
        
        dF = 2.0*(dfE*gE - fE*dgE)/(gE**2)    
        return [grad_val * dF.flatten()]


@optplan.register_node(EnergyConstraint)
def create_energy_constraint(
        params: EnergyConstraint,
        context: workspace.Workspace) -> EnergyConstraintFunction:
    simspace = context.get_object(params.field.simulation_space)
    source = creator_em.DipoleSource(params.field.source)
    solver = creator_em._create_solver(params.field.solver, simspace)
    J = source(simspace, params.epsilon.wavelength, solver)    
    return EnergyConstraintFunction(
        field=context.get_object(params.field),
        eps=context.get_object(params.epsilon),
        source=J, 
        n_core=params.n_core,
        n_clad=params.n_clad,
        simspace=simspace,
        plane_slice=grid_utils.create_region_slices(simspace.edge_coords,params.center,params.extents))       




        # eps_plot = eps[0].reshape(81, 251)
        # field_plot = efield[1].reshape(81, 251)
    
        # fig = plt.figure(figsize=[10,5])
        # ax = plt.gca()
        
        # data = np.real(eps_plot)
        # data_min = np.amin(data)
        # data_max = np.amax(data)
        # w = data.shape[1]*20/1000
        # h = data.shape[0]*20/1000
        # divnorm = mcolors.TwoSlopeNorm(vmin=data_min,vcenter=(data_min+data_max)/2,vmax=data_max)
        # ax.imshow(data, extent=[-w/2,w/2,-h/2,h/2], norm=divnorm, cmap='Greys', alpha=0.7, interpolation='spline36')

        # data = np.real(field_plot)
        # data_min = np.amin(data)
        # data_max = np.amax(data)
        # divnorm = mcolors.TwoSlopeNorm(vmin=data_min,vcenter=(data_min+data_max)/2,vmax=data_max)
        # image = ax.imshow(data, extent=[-w/2,w/2,-h/2,h/2], norm=divnorm, cmap='bwr', alpha=0.5, interpolation='spline36')

        # ax.set_xlim(-w/2,w/2)
        # ax.set_ylim(-h/2,h/2)        
        # ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))                
        # ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))            
        # ax.set_xlabel('x ($\mu$m)')
        # ax.set_ylabel('y ($\mu$m)') 
        # cbar = fig.colorbar(image, ax=ax, fraction=0.1, pad=0.01, shrink=0.3)
        # cbar.ax.minorticks_off()
        
        # fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        # plt.savefig('eps_field.pdf', bbox_inches='tight')          
        # plt.close()   