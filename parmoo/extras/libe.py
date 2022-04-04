import numpy as np
from parmoo import MOOP


def parmoo_persis_gen(H, persis_info, gen_specs, libE_info):
    """ A persistent ParMOO generator function for libEnsemble.

    This generator function is meant to be called from within libEnsemble.

    Args:
        H (numpy structured array): The current history.

        persis_info (dict): Any information that should persist after this
            generator has exited. Must contain the following field:
             * 'moop' (parmoo.MOOP)

        gen_specs (dict): A list of specifications for the generator function.

        libE_info (dict): Other information that will be used by libEnsemble.

    Returns:
        dict: The final simulation history.

        dict: The persistent information after completion of the generator.

        int: The stop tag.

    """

    from libensemble.message_numbers import STOP_TAG, PERSIS_STOP, EVAL_GEN_TAG
    from libensemble.message_numbers import FINISHED_PERSISTENT_GEN_TAG
    from libensemble.tools.persistent_support import PersistentSupport

    # Get moop from pers_info
    if 'moop' in persis_info.keys():
        moop = persis_info['moop']
        if not isinstance(moop, MOOP):
            raise TypeError("persis_info['moop'] must be an instance of " +
                            "parmoo.MOOP class")
    else:
        raise KeyError("'moop' key is required in persis_info dict")
    # Setup persistent support
    ps = PersistentSupport(libE_info, EVAL_GEN_TAG)
    # Send batches until manager sends stop tag
    tag = None
    k = 0
    sim_count = 0
    # Iterate until the termination condition is reached
    while tag not in [STOP_TAG, PERSIS_STOP]:
        # Generate a batch by running one iteration
        x_out = moop.iterate(k)
        # Check for duplicates in simulation databases
        xbatch = []
        ibatch = []
        for (xi, i) in x_out:
            if moop.check_sim_db(xi, i) is None:
                xbatch.append(xi)
                ibatch.append(i)
        # Get the batch size and allocate the H_o structured array
        b = len(xbatch)
        H_o = np.zeros(b, dtype=gen_specs['out'])
        # Populate the H_o structured array 'x' values as appropriate
        if moop.use_names:
            for name in moop.des_names:
                for i in range(b):
                    H_o[name[0]][i] = xbatch[i][name[0]]
        else:
            H_o['x'] = np.asarray(xbatch)
        for i, namei in enumerate(ibatch):
            H_o['sim_name'][i] = namei
        # Evaluate H_o and add to the simulation database
        batch = []
        if isinstance(x_out[0][-1], str) or x_out[0][-1] >= 0:
            tag, Work, calc_in = ps.send_recv(H_o)
            if calc_in is not None:
                for s_out in calc_in:
                    sim_name = s_out['sim_name']
                    # Check whether design variables are all named
                    if moop.use_names:
                        xx = np.zeros(1, dtype=moop.des_names)[0]
                        for name in moop.des_names:
                            xx[name[0]] = s_out[name[0]]
                        sim_num = -1
                        for j, sj in enumerate(moop.sim_names):
                            if sj[0] == sim_name:
                                sim_num = j
                                break
                        sx = np.zeros(moop.m[sim_num])
                        sx[:] = s_out[moop.sim_names[sim_num][0]]
                        sname = sim_name.decode('utf-8')
                    else:
                        xx = np.zeros(moop.n)
                        xx[:] = s_out['x'][:]
                        sx = np.zeros(moop.m[sim_name])
                        sx[:] = s_out['f'][:]
                        sname = int(sim_name)
                    # Copy sim results into ParMOO databases
                    moop.update_sim_db(xx, sx, sname)
                    batch.append((xx, sname))
                    sim_count += 1
            else:
                new_count = 0
                for s_out in Work[sim_count:]:
                    sim_name = s_out['sim_name']
                    # Check whether design variables are all named
                    if moop.use_names:
                        xx = np.zeros(1, dtype=moop.des_names)[0]
                        for name in moop.des_names:
                            xx[name[0]] = s_out[name[0]]
                        sim_num = -1
                        for j, sj in enumerate(moop.sim_names):
                            if sj[0] == sim_name:
                                sim_num = j
                                break
                        sx = np.zeros(moop.m[sim_num])
                        sx[:] = s_out[moop.sim_names[sim_num][0]]
                        sname = sim_name.decode('utf-8')
                    else:
                        xx = np.zeros(moop.n)
                        xx[:] = s_out['x'][:]
                        sx = np.zeros(moop.m[sim_name])
                        sx[:] = s_out['f'][:]
                        sname = int(sim_name)
                    # Copy sim results into ParMOO databases
                    moop.update_sim_db(xx, sx, sname)
                    batch.append((xx, sname))
                    new_count += 1
                sim_count += new_count
        # Update the ParMOO databases
        moop.updateAll(k, batch)
        k += 1
    # Return the results
    persis_info['moop'] = moop
    return H_o, persis_info, FINISHED_PERSISTENT_GEN_TAG


class libE_MOOP(MOOP):
    """ Class for solving a MOOP using libEnsemble to manage parallelism.

    Upon initialization, one must supply a scalar optimization procedure
    and dictionary of hyperparameters. Class methods are summarized below.

    Objectives and algebraic constraints on the design variables and
    objective values can be added using the following functions:
     * ``addDesign(*args)``
     * ``addSimulation(*args)``
     * ``addObjective(*args)``
     * ``addConstraint(*args)``

    Acquisition functions (used for scalarizing problems/setting targets) are
    added using:
     * ``addAcquisition(*args)``

    After creating a MOOP, the following methods are used to get the
    numpy.dtype used to create each of the following input/output arrays:
     * ``getDesignType()``
     * ``getSimulationType()``
     * ``getObjectiveType()``
     * ``getConstraintType()``

    The following methods are used to save/load ParMOO objects from memory.
     * ``setCheckpoint(checkpoint, savedata=False, filename="parmoo")``
     * ``save(filename="parmoo")``
     * ``load(filename="parmoo")``

    The following methods are used for solving the MOOP and managing the
    internal simulation/objective databases:
     * ``check_sim_db(x, s_name)``
     * ``update_sim_db(x, sx, s_name)``
     * ``evaluateSimulation(x, s_name)``
     * ``addData(x, sx)``
     * ``iterate(k)``
     * ``updateAll(k, batch)``
     * ``solve(budget)``

    Finally, the following methods are used to retrieve data after the
    problem has been solved:
     * ``getPF()``
     * ``getSimulationData()``
     * ``getObjectiveData()``

    Other private methods from the MOOP class do not work for a libE_MOOP.

    """

    __slots__ = ['moop']

    def __init__(self, opt_func, hyperparams=None):
        """ Initializer for the libE interface to the MOOP class.

        Args:
            opt_func (SurrogateOptimizer): A solver for the surrogate problems.

            hyperparams (dict, optional): A dictionary of hyperparameters for
                the opt_func, and any other procedures that will be used.

        Returns:
            libE_MOOP: A new libE_MOOP object with no design variables,
                objectives, or constraints.

        """

        if hyperparams is None:
            hp = {}
        else:
            hp = hyperparams
        # Create a MOOP
        self.moop = MOOP(opt_func, hyperparams=hp)
        return

    def addDesign(self, *args):
        """ Add a new design variables to the MOOP.

        Append new design variables to the problem. Note that every design
        variable must be added before any simulations or acquisition functions
        can be added since the number of design variables is used to infer
        the size of simulation databases and acquisition function policies.

        Args:
            args (dict): Each argument is a dictionary representing one design
                variable. The dictionary contains information about that
                design variable, including:
                 * 'name' (String, optional): The name of this design
                   if any are left blank, then ALL names are considered
                   unspecified.
                 * 'des_type' (String): The type for this design variable.
                   Currently supported options are:
                    * 'continuous'
                    * 'categorical'
                 * 'lb' (float): When des_type is 'continuous', this specifies
                   the lower bound for the design variable. This value must
                   be specified, and must be strictly less than 'ub'
                   (below) up to the tolerance (below).
                 * 'ub' (float): When des_type is 'continuous', this specifies
                   the upper bound for the design variable. This value
                   must be specified, and must be strictly greater than
                   'lb' (above) up to the tolerance (below).
                 * 'tol' (float): When des_type is 'continuous', this specifies
                   the tolerance, i.e., the minimum spacing along this
                   dimension, before two design values are considered
                   to have equal values in this dimension. If not specified,
                   the default value is 1.0e-8.
                 * 'levels' (String []): When des_type is 'categorical', this
                   specifies the name for each level of the design variable.
                   The number of levels is inferred from the length of
                   the list.

        """

        self.moop.addDesign(*args)
        return

    def addSimulation(self, *args):
        """ Add new simulations to the MOOP.

        Append new simulation functions to the problem.

        Args:
            args (dict): Each argument is a dictionary representing one
                simulation function. The dictionary must contain information
                about that simulation function, including:
                 * name (String, optional): The name of this simulation
                   (defaults to "sim" + str(i), where i = 1, 2, 3, ... for
                   the first, second, third, ... simulation added to the
                   MOOP).
                 * m (int): The number of outputs for this simulation.
                 * sim_func (function): An implementation of the simulation
                   function, mapping from R^n -> R^m. The interface should
                   match:
                   `sim_out = sim_func(x, der=False)`,
                   where `der` is an optional argument specifying whether
                   to take the derivative of the simulation. Unless
                   otherwise specified by your solver, `der` is always
                   omitted by ParMOO's internal structures, and need not
                   be implemented.
                 * search (GlobalSearch): A GlobalSearch object for performing
                   the initial search over this simulation's design space.
                 * surrogate (SurrogateFunction): A SurrogateFunction object
                   specifying how this simulation's outputs will be modeled.
                 * des_tol (float): The tolerance for this simulation's
                   design space; a new design point that is closer than
                   des_tol to a point that is already in this simulation's
                   database will not be reevaluated.
                 * hyperparams (dict): A dictionary of hyperparameters, which
                   will be passed to the surrogate and search routines.
                   Most notably, search_budget (int) can be specified
                   here.
                 * sim_db (dict, optional): A dictionary of previous
                   simulation evaluations. When present, contains:
                    * x_vals (np.ndarray): A 2d array of pre-evaluated
                      design points.
                    * s_vals (np.ndarray): A 2d array of corresponding
                      simulation outputs.
                    * g_vals (np.ndarray): A 3d array of corresponding
                      Jacobian values. This value is only needed
                      if the provided SurrogateFunction uses gradients.

        """

        self.moop.addSimulation(*args)
        return

    def addObjective(self, *args):
        """ Add a new objective to the MOOP.

        Append a new objective to the problem. The objective must be an
        algebraic function of the simulations. Note that all objectives
        must be specified before any acquisition functions can be added.

        Args:
            *args (dict): Python dictionary containing objective function info.
            Contains the fields:
             * 'name' (String, optional): The name of this objective
               (defaults to "obj" + str(i), where i = 1, 2, 3, ... for
               the first, second, third, ... simulation added to the MOOP).
             * 'obj_func' (function): An algebraic objective function that maps
               from R^n X R^m --> R. Interface should match:
               `cost = obj_func(x, sim_func(x), der=0)`,
               where `der` is an optional argument specifying whether to
               take the derivative of the objective function
                * 0 (or any other value) -- not at all,
                * 1 -- wrt x, or
                * 2 -- wrt sim(x).

        """

        self.moop.addObjective(*args)
        return

    def addConstraint(self, *args):
        """ Add a new constraint to the MOOP.

        Append a new design constraint to the problem. The constraint can
        be nonlinear and depend on the design values and simulation outputs.

        Args:
            *args (dict): Python dictionary containing constraint function
            information. Contains the keys:
             * 'name' (String, optional): The name of this constraint
               (defaults to "const" + str(i), where i = 1, 2, 3, ... for
               the first, second, third, ... constraint added to the MOOP).
             * 'constraint' (function): An algebraic constraint function that
               maps from R^n X R^m --> R and evaluates to zero or a
               negative number when feasible and positive otherwise.
               Interface should match:
               `violation = constraint(x, sim_func(x), der=0)`,
               where `der` is an optional argument specifying whether to
               take the derivative of the constraint function
                * 0 (or any other value) -- not at all,
                * 1 -- wrt x, or
                * 2 -- wrt sim(x).

        """

        self.moop.addConstraint(*args)
        return

    def addAcquisition(self, *args):
        """ Add an acquisition function to the MOOP.

        Append a new acquisition function to the problem. In each iteration,
        each acquisition is used to generate 1 or more points to evaluate.

        Args:
            args (dict): Python dictionary of acquisition function info.
                Contains the fields:
                 * 'acquisition' (AcquisitionFunction): An acquisition function
                   that maps from R^o --> R for scalarizing outputs.
                 * 'hyperparams' (dict): A dictionary of hyperparams for the
                   acquisition functions. Can be omitted if no hyperparams
                   are needed.

        """

        self.moop.addAcquisition(*args)
        return

    def getDesignType(self):
        """ Get the numpy dtype of a design point for this MOOP.

        Use this type if allocating a numpy array to store the design
        points for this MOOP object.

        Returns:
            The numpy dtype of this MOOP's design points.
            If no design variables have yet been added, returns None.

        """

        return self.moop.getDesignType()

    def getSimulationType(self):
        """ Get the numpy dtypes of the simulation outputs for this MOOP.

        Use this type if allocating a numpy array to store the simulation
        outputs of this MOOP object.

        Returns:
            The numpy dtype of this MOOP's simulation outputs.
            If no simulations have been given, returns None.

        """

        return self.moop.getSimulationType()

    def getObjectiveType(self):
        """ Get the numpy dtype of an objective point for this MOOP.

        Use this type if allocating a numpy array to store the objective
        values of this MOOP object.

        Returns:
            The numpy dtype of this MOOP's objective points.
            If no objectives have yet been added, returns None.

        """

        return self.moop.getObjectiveType()

    def getConstraintType(self):
        """ Get the numpy dtype of the constraint violations for this MOOP.

        Use this type if allocating a numpy array to store the constraint
        scores output of this MOOP object.

        Returns:
            The numpy dtype of this MOOP's constraint violation output.
            If no constraints have been given, returns None.

        """

        return self.moop.getConstraintType()

    def check_sim_db(self, x, s_name):
        """ Check the sim_db[s_name] in this MOOP for a design point.

        x (np.ndarray): A 1d array specifying the point to check for.

        s_name (String, int): The name or index of the simulation where
            (x, sx) will be added. Note, indices are assigned in the order
            the simulations were listed during initialization.

        """

        return self.moop.check_sim_db(x, s_name)

    def update_sim_db(self, x, sx, s_name):
        """ Update sim_db[s_name] by adding a new design/objective pair.

        x (np.ndarray): A 1d array specifying the design point to add.

        sx (np.ndarray): A 1d array with the corresponding objective value.

        s_name (String, int): The name or index of the simulation where
            (x, sx) will be added. Note, indices are assigned in the order
            the simulations were listed during initialization.

        """

        self.moop.update_sim_db(x, sx, s_name)
        return

    def evaluateSimulation(self, x, s_name):
        """ Evaluate the simulation[s_name] and store the result.

        Args:
            x (numpy.ndarray): Either a numpy structured array (when 'name'
                key was given for all design variables) or a 1D array
                containing design variable values (in the order that they
                were added to the MOOP).

            s_name (String, int): The name or index of the simulation to
                evaluate. Note, indices are assigned in the order
                the simulations were listed during initialization.

        Returns:
            numpy.ndarray: A 1d array containing the output from the
            simulation[s_name] at x.

        """

        return self.moop.evaluateSimulation(x, s_name)

    def addData(self, x, sx):
        """ Update the internal objective database by truly evaluating x.

        Args:
            x (numpy.ndarray): Either a numpy structured array (when 'name'
                key was given for all design variables) or a 1D array
                containing design variable values (in the order that they
                were added to the MOOP).

            sx (numpy.ndarray): The corresponding simulation outputs.

        """

        self.moop.addData(x, sx)
        return

    def setCheckpoint(self, checkpoint, savedata=True, filename="parmoo"):
        """ Set ParMOO's checkpointing feature.

        Args:
            checkpoint (bool): Turn checkpointing on (True) or off (False).

            savedata (bool, optional): Also save raw simulation output in
                a separate .json file (True) or rely on ParMOO's internal
                simulation database (False). When omitted, this parameter
                defaults to False.

            filename (str, optional): Set the base checkpoint filename/path.
                The checkpoint file will have the JSON format and the
                extension ".moop" appended to the end of filename.
                Additional checkpoint files may be created with the same
                filename but different extensions, depending on the choice
                of AcquisitionFunction, SurrogateFunction, and GlobalSearch.
                When omitted, this parameter defaults to "parmoo" and
                is saved inside current working directory.

        """

        self.moop.setCheckpoint(checkpoint, savedata=savedata,
                                filename=filename)

    def iterate(self, k):
        """ Perform one iteration of ParMOO and generate a batch of candidates.

        Args:
            k (int): The iteration counter.

        Returns:
            (list): A list of ordered pairs.
            The first entry is either a 1D numpy structured array (when
            'name' key was given for all design variables) or a 2D ndarray
            where each row contains design variable values in the order
            that they were added to the MOOP. This output specifies the
            list of design points that ParMOO suggests for evaluation
            in this iteration.
            The second entry is either the name of the simulation to
            evaluate (when 'name' key was given for all design variables)
            or the integer index of the simulation to evaluate.

        """

        return self.moop.iterate(k)

    def updateAll(self, k, batch):
        """ Update all surrogates given a batch of freshly evaluated data.

        Args:
            k (int): The iteration counter.

            batch (list): A list of design point (x) simulation index (i)
                pairs: [(x1, i1), (x2, i2), ...]. Each 'x' is either
                a numpy structured array (when 'name' key was given for
                all design variables) or a 1D array containing design
                variable values (in the order that they were added to
                the MOOP).

        """

        return self.moop.updateAll(k, batch)

    def solve(self, sim_max=200, wt_max=3600, profile=False):
        """ Solve a MOOP using ParMOO.

        Args:
            sim_max (int): The max number of simulation to be performed by
                libEnsemble (default is 200).

            wt_max (int): The max number of seconds that the simulation may
                run for (the default is 3600 secs, i.e., 1 hr).

            profile (bool): Specifies whether to run libE with the profiler.

        """

        # Import libEnsemble libraries
        import sys
        from libensemble.libE import libE
        from libensemble.alloc_funcs.start_only_persistent \
            import only_persistent_gens as alloc_f
        from libensemble.tools import parse_args

        def moop_sim(H, persis_info, sim_specs, _):
            """ Evaluates the sim function for a collection of points given in
            ``H['x']``.

            """

            batch = len(H)
            sim_names = H['sim_name']
            H_o = np.zeros(batch, dtype=sim_specs['out'])
            for i in range(batch):
                namei = sim_names[i]
                if self.moop.use_names:
                    j = -1
                    for jj, jname in enumerate(self.moop.sim_names):
                        if jname[0] == sim_names[i]:
                            j = jj
                            break
                else:
                    j = namei
                if self.moop.use_names:
                    xx = np.zeros(1, dtype=self.moop.des_names)[0]
                    for name in self.moop.des_names:
                        xx[name[0]] = H[name[0]][i]
                    H_o[self.moop.sim_names[j][0]][i] = \
                        self.moop.sim_funcs[j](xx)
                else:
                    H_o['f'][i, :self.moop.m[j]] = \
                        self.moop.sim_funcs[j](H['x'][i])
            return H_o, persis_info

        nworkers, is_manager, libE_specs, _ = parse_args()
        if self.moop.use_names:
            libE_specs['final_fields'] = []
            for name in self.moop.des_names:
                libE_specs['final_fields'].append(name[0])
            for name in self.moop.sim_names:
                libE_specs['final_fields'].append(name[0])
            libE_specs['final_fields'].append('sim_name')
        else:
            libE_specs['final_fields'] = ['x', 'f', 'sim_name']
        # Set optional libE specs
        libE_specs['profile'] = profile

        if nworkers < 2:
            raise ValueError("Cannot run ParMOO + libE with less than 2 " +
                             "workers -- aborting...")

        # Get the max m for all SimGroups
        max_m = max(self.moop.m)

        # Set the input dictionaries
        if self.moop.use_names:
            x_type = self.moop.des_names.copy()
            x_type.append(('sim_name', 'a10'))
            f_type = self.moop.sim_names.copy()
            all_types = x_type.copy()
            for name in f_type:
                all_types.append(name)
            sim_specs = {'sim_f': moop_sim,
                         'in': [name[0] for name in x_type],
                         'out': f_type}

            gen_specs = {'gen_f': parmoo_persis_gen,
                         'persis_in': [name[0] for name in all_types],
                         'out': x_type,
                         'user': {}}
        else:
            sim_specs = {'sim_f': moop_sim,
                         'in': ['x', 'sim_name'],
                         'out': [('f', float, max_m)]}

            gen_specs = {'gen_f': parmoo_persis_gen,
                         'persis_in': ['x', 'sim_name', 'f'],
                         'out': [('x', float, self.moop.n),
                                 ('sim_name', int)],
                         'user': {}}

        alloc_specs = {'alloc_f': alloc_f, 'out': [('given_back', bool)]}

        persis_info = {}
        for i in range(nworkers + 1):
            persis_info[i] = {}
        persis_info[1]['moop'] = self.moop

        exit_criteria = {'sim_max': sim_max, 'elapsed_wallclock_time': wt_max}

        # Perform the run
        H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria,
                                    persis_info, alloc_specs, libE_specs)

        self.moop = persis_info[1]['moop']
        return

    def getPF(self):
        """ Extract the nondominated and efficient sets from internal database.

        Returns:
            A discrete approximation of the Pareto front and efficient set.

            If all design names were given, then this is a 1d numpy
            structured array whose fields match the names for design
            variables, objectives, and constraints (if any).

            Otherwise, this is a dict containing the following keys:
             * x_vals (numpy.ndarray): A 2d array containing a list
               of nondominated points discretely approximating the
               Pareto front.
             * f_vals (numpy.ndarray): A 2d array containing the list
               of corresponding efficient design points.
             * c_vals (numpy.ndarray): A 2d array containing the list
               of corresponding constraint satisfaction scores,
               all less than or equal to 0.

        """

        return self.moop.getPF()

    def getSimulationData(self):
        """ Extract all computed simulation outputs from database.

        Returns:
            (dict) A dictionary containing all evaluated simulations.

            If all design names were given, then each key is the
            'name' for a different simulation, and each value
            is a 1d numpy structured array whose keys match the
            names for each design variables plus an
            additional 'out' key for simulation outputs.

            Otherwise, this is a dict containing the following keys:
             * x_vals (numpy.ndarray): A 2d array containing a list
               of design points that have been evaluated for this
               simulation.
             * s_vals (numpy.ndarray): A 1d or 2d array containing
               the list of corresponding simulation outputs.

        """

        return self.moop.getSimulationData()

    def getObjectiveData(self):
        """ Extract all computed objective scores from database.

        Returns:
            A database of all designs that have been fully evaluated,
            and their corresponding objective scores.

            If all design names were given, then this is a 1d numpy
            structured array whose fields match the names for design
            variables, objectives, and constraints (if any).

            Otherwise, this is a dict containing the following keys:
             * x_vals (numpy.ndarray): A 2d array containing a list
               of all fully evaluated design points.
             * f_vals (numpy.ndarray): A 2d array containing the list
               of corresponding objective values.
             * c_vals (numpy.ndarray): A 2d array containing the list
               of corresponding constraint satisfaction scores,
               all less than or equal to 0.

        """

        return self.moop.getObjectiveData()

    def save(self, filename="parmoo"):
        """ Serialize and save the MOOP object and all of its dependencies.

        Args:
            filename (string, optional): The filepath to serialized
                checkpointing file(s). Do not include file extensions,
                they will be appended automaically. May create
                several save files with extensions of this name, in order
                to recursively save dependencies objects. Defaults to
                the value "parmoo" (filename will be "parmoo.moop").

        """

        self.moop.save(filename=filename)
        return

    def load(self, filename="parmoo"):
        """ Load a serialized MOOP object and all of its dependencies.

        Args:
            filename (string, optional): The filepath to serialized
                checkpointing file(s). Do not include file extensions,
                they will be appended automaically. May also load from
                other save files with different extensions of this name,
                in order to recursively load dependencies objects.
                Defaults to the value "parmoo" (filename will be
                "parmoo.moop").

        """

        self.moop.load(filename=filename)
        return
