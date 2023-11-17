""" Wraps the scipy.optimize optimizers. """
import logging
import numpy as np
import scipy.optimize
import nlopt

from spins.invdes.problem.objective import (OptimizationFunction,
                                            OptimizationProblem)
from spins.invdes.parametrization import Parametrization

logger = logging.getLogger(__name__)

class ScipyOptimizer:
    """ Wraps scipy.optimize.minimize. """

    def __init__(self,
                 objective: OptimizationFunction = None,
                 parametrization: Parametrization = None,
                 method: str = None,
                 tol: float = None,
                 options=None,
                 constraints=(),
                 bounds=None):
        """ Initializes parameters to use for scipy.optimize.minimize.

        Args:
            objective: Objective function to minimize.
            parametrization: Parametrization.
            method: Method to use. See scipy.optimize.minimize documentation.
            options: Options to pass to scipy.optimize.minimize.
            constraints: See scipy.optimize.minimize documentation.
            bounds: See scipy.optimize.minimize documentation. Overrides any
                    bounds in the parametrization.
        """
        self.objective = objective
        self.param = parametrization
        self.method = method
        self.options = options
        self.constraints = constraints
        self.bounds = bounds
        self.tol = tol

        # Spins places tols in options, tol saved in self.tol and removed from self.options
        if options:
            if "tol" in options.keys():
                self.tol = options["tol"]
                del self.options["tol"]

    def __call__(self, opt, param, callback=None):
        if isinstance(opt, OptimizationFunction):
            opt = OptimizationProblem(opt)

        objective = opt.get_objective()

        def func(x):
            param.from_vector(x)
            return objective.calculate_objective_function(param)

        def jacobian(x):
            param.from_vector(x)
            return np.asfortranarray(objective.calculate_gradient(param))

        # To avoid scipy warning, only pass Jacobian to methods that need it.
        methods_that_need_jacobian = {
            'CG', 'BFGS', 'L-BFGS-B', 'TNC', 'SLSQP', 'dogleg', 'trust-ncg', 'MMA'
        }
        jac = None
        if self.method in methods_that_need_jacobian:
            jac = jacobian

        # Set bounds only if optimization method uses them.
        methods_that_accept_bounds = {'L-BFGS-B', 'TNC', 'SLSQP', 'MMA'}
        bounds = None
        if self.method in methods_that_accept_bounds:
            # Let optimizer-defined bounds override default bounds.
            bounds = self.bounds
            if bounds is None:
                bounds = param.get_bounds()
                # Unpack bounds into a list of tuples for each element.
                if bounds:
                    bounds = list(zip(*param.get_bounds()))

        # Handle constraints.
        methods_that_accept_constraints = {'SLSQP', 'COBYLA'}
        constraints = None
        if self.method in methods_that_accept_constraints:
            constraints = []
            for eq in opt.get_equality_constraints():

                def cons_fun(x):
                    param.from_vector(x)
                    return eq.calculate_objective_function(param)

                def cons_jac(x):
                    param.from_vector(x)
                    return eq.calculate_gradient(param)

                constraints.append({
                    'type': 'eq',
                    'fun': cons_fun,
                    'jac': cons_jac
                })
            for ineq in opt.get_inequality_constraints():
                # Note the negative sign because of opposite convention
                # for inequalities (f >= 0 vs f <= 0).
                def cons_fun(x):
                    param.from_vector(x)
                    return -ineq.calculate_objective_function(param)

                def cons_jac(x):
                    param.from_vector(x)
                    return -ineq.calculate_gradient(param)

                constraints.append({
                    'type': 'ineq',
                    'fun': cons_fun,
                    'jac': cons_jac
                })
            # Append any additional constraints.
            if self.constraints:
                constraints += self.constraints
        elif (len(opt.get_equality_constraints()) > 0 or
              len(opt.get_inequality_constraints()) > 0):
            logger.warning('Ignoring constraints...')
        
        if self.method == 'MMA':
            def nlopt_obj(x, gradient):
                fval = func(x)
                if gradient.size > 0:
                    gradient[:] = jac(x)
                callback(x)
                return np.float64(fval)
            
            alg = nlopt.LD_MMA                  # Method of Moving Asymptotes.
            x = np.asarray(param.to_vector())   # Initial values.
            n = len(x)                          # Number of parameters.    
            lb = [b[0] for b in bounds]         # Lower bounds.
            ub = [b[1] for b in bounds]         # Upper bounds.   
            # Solver setup.
            solver = nlopt.opt(alg, n)
            solver.set_lower_bounds(lb)
            solver.set_upper_bounds(ub)
            solver.set_min_objective(nlopt_obj)
            solver.set_maxeval(self.options["maxiter"])
            solver.set_xtol_rel(1e-4)
            x[:] = solver.optimize(x)
            param.from_vector(x)
        else:
            #self.options["disp"] = True
            #self.options["iprint"] = 99
            self.results = scipy.optimize.minimize(
                                                    func,
                                                    param.to_vector(),
                                                    jac=jac,
                                                    tol=self.tol,
                                                    method=self.method,
                                                    options=self.options,
                                                    callback=callback,
                                                    constraints=constraints,
                                                    bounds=bounds)
            param.from_vector(self.results['x'])
            #print('Status : %s' % self.results['message'])
            #print('Total Evaluations: %d' % self.results['nfev'])
        return param

    def optimize(self, callback=None):
        """ Run scipy.optimize.minimize.

        Args:
            callback: Callback to pass to scipy.optimize.minimize.
        """
        return self(self.objective, self.param, callback)

    def get_results(self):
        return self.results

#class ScipyOptimizer:
#    """ Wraps scipy.optimize.minimize. """
#
#    def __init__(self,
#                 objective: OptimizationFunction = None,
#                 parametrization: Parametrization = None,
#                 method: str = None,
#                 tol: float = None,
#                 options=None,
#                 constraints=(),
#                 bounds=None):
#        """ Initializes parameters to use for scipy.optimize.minimize.
#
#        Args:
#            objective: Objective function to minimize.
#            parametrization: Parametrization.
#            method: Method to use. See scipy.optimize.minimize documentation.
#            options: Options to pass to scipy.optimize.minimize.
#            constraints: See scipy.optimize.minimize documentation.
#            bounds: See scipy.optimize.minimize documentation. Overrides any
#                    bounds in the parametrization.
#        """
#        self.objective = objective
#        self.param = parametrization
#        self.method = method
#        self.options = options
#        self.constraints = constraints
#        self.bounds = bounds
#        self.tol = tol
#
#        # Spins places tols in options, tol saved in self.tol and removed from self.options
#        if options:
#            if "tol" in options.keys():
#                self.tol = options["tol"]
#                del self.options["tol"]
#
#    def __call__(self, opt, param, callback=None):
#        if isinstance(opt, OptimizationFunction):
#            opt = OptimizationProblem(opt)
#
#        objective = opt.get_objective()
#
#        def func(x):
#            param.from_vector(x)
#            return objective.calculate_objective_function(param)
#
#        def jacobian(x):
#            param.from_vector(x)
#            return np.asfortranarray(objective.calculate_gradient(param))
#
#        # To avoid scipy warning, only pass Jacobian to methods that need it.
#        methods_that_need_jacobian = {
#            'CG', 'BFGS', 'L-BFGS-B', 'TNC', 'SLSQP', 'dogleg', 'trust-ncg'
#        }
#        jac = None
#        if self.method in methods_that_need_jacobian:
#            jac = jacobian
#
#        # Set bounds only if optimization method uses them.
#        methods_that_accept_bounds = {'L-BFGS-B', 'TNC', 'SLSQP'}
#        bounds = None
#        if self.method in methods_that_accept_bounds:
#            # Let optimizer-defined bounds override default bounds.
#            bounds = self.bounds
#            if bounds is None:
#                bounds = param.get_bounds()
#                # Unpack bounds into a list of tuples for each element.
#                if bounds:
#                    bounds = list(zip(*param.get_bounds()))
#
#        # Handle constraints.
#        methods_that_accept_constraints = {'SLSQP', 'COBYLA'}
#        constraints = None
#        if self.method in methods_that_accept_constraints:
#            constraints = []
#            for eq in opt.get_equality_constraints():
#
#                def cons_fun(x):
#                    param.from_vector(x)
#                    return eq.calculate_objective_function(param)
#
#                def cons_jac(x):
#                    param.from_vector(x)
#                    return eq.calculate_gradient(param)
#
#                constraints.append({
#                    'type': 'eq',
#                    'fun': cons_fun,
#                    'jac': cons_jac
#                })
#            for ineq in opt.get_inequality_constraints():
#                # Note the negative sign because of opposite convention
#                # for inequalities (f >= 0 vs f <= 0).
#                def cons_fun(x):
#                    param.from_vector(x)
#                    return -ineq.calculate_objective_function(param)
#
#                def cons_jac(x):
#                    param.from_vector(x)
#                    return -ineq.calculate_gradient(param)
#
#                constraints.append({
#                    'type': 'ineq',
#                    'fun': cons_fun,
#                    'jac': cons_jac
#                })
#            # Append any additional constraints.
#            if self.constraints:
#                constraints += self.constraints
#        elif (len(opt.get_equality_constraints()) > 0 or
#              len(opt.get_inequality_constraints()) > 0):
#            logger.warning('Ignoring constraints...')
#
#        self.results = scipy.optimize.minimize(
#            func,
#            param.to_vector(),
#            jac=jac,
#            tol=self.tol,
#            method=self.method,
#            options=self.options,
#            callback=callback,
#            constraints=constraints,
#            bounds=bounds)
#        param.from_vector(self.results['x'])
#        return param
#
#    def optimize(self, callback=None):
#        """ Run scipy.optimize.minimize.
#
#        Args:
#            callback: Callback to pass to scipy.optimize.minimize.
#        """
#        return self(self.objective, self.param, callback)
#
#    def get_results(self):
#        return self.results
