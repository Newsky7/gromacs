#include "gmxpre.h"

#include "config.h"

#include <cmath>
#include <cstring>
#include <ctime>

#include <algorithm>
#include <limits>
#include <vector>

#include "gromacs/commandline/filenm.h"
#include "gromacs/domdec/collect.h"
#include "gromacs/domdec/dlbtiming.h"
#include "gromacs/domdec/domdec.h"
#include "gromacs/domdec/domdec_struct.h"
#include "gromacs/domdec/mdsetup.h"
#include "gromacs/domdec/partition.h"
#include "gromacs/ewald/pme_pp.h"
#include "gromacs/fileio/confio.h"
#include "gromacs/fileio/mtxio.h"
#include "gromacs/gmxlib/network.h"
#include "gromacs/gmxlib/nrnb.h"
#include "gromacs/imd/imd.h"
#include "gromacs/linearalgebra/sparsematrix.h"
#include "gromacs/listed_forces/listed_forces.h"
#include "gromacs/math/functions.h"
#include "gromacs/math/vec.h"
#include "gromacs/mdlib/constr.h"
#include "gromacs/mdlib/coupling.h"
#include "gromacs/mdlib/dispersioncorrection.h"
#include "gromacs/mdlib/ebin.h"
#include "gromacs/mdlib/enerdata_utils.h"
#include "gromacs/mdlib/energyoutput.h"
#include "gromacs/mdlib/force.h"
#include "gromacs/mdlib/force_flags.h"
#include "gromacs/mdlib/forcerec.h"
#include "gromacs/mdlib/gmx_omp_nthreads.h"
#include "gromacs/mdlib/md_support.h"
#include "gromacs/mdlib/mdatoms.h"
#include "gromacs/mdlib/stat.h"
#include "gromacs/mdlib/tgroup.h"
#include "gromacs/mdlib/trajectory_writing.h"
#include "gromacs/mdlib/update.h"
#include "gromacs/mdlib/vsite.h"
#include "gromacs/mdrunutility/handlerestart.h"
#include "gromacs/mdrunutility/printtime.h"
#include "gromacs/mdtypes/checkpointdata.h"
#include "gromacs/mdtypes/commrec.h"
#include "gromacs/mdtypes/forcebuffers.h"
#include "gromacs/mdtypes/forcerec.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/interaction_const.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/mdtypes/mdatom.h"
#include "gromacs/mdtypes/mdrunoptions.h"
#include "gromacs/mdtypes/observablesreducer.h"
#include "gromacs/mdtypes/state.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/timing/wallcycle.h"
#include "gromacs/timing/walltime_accounting.h"
#include "gromacs/topology/mtop_util.h"
#include "gromacs/topology/topology.h"
#include "gromacs/utility/cstringutil.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/logger.h"
#include "gromacs/utility/smalloc.h"

#include "legacysimulator.h"
#include "shellfc.h"

#include <iostream>
#include <fstream>  

using gmx::ArrayRef;
using gmx::MdrunScheduleWorkload;
using gmx::RVec;
using gmx::VirtualSitesHandler;

//! Utility structure for manipulating states during EM
typedef struct em_state
{
    //! Copy of the global state
    t_state s;
    //! Force array
    gmx::ForceBuffers f;
    //! Potential energy
    real epot;
    //! Norm of the force
    real fnorm;
    //! Maximum force
    real fmax;
    //! Direction
    int a_fmax;
} em_state_t;

//! Print the EM starting conditions
static void print_em_start(FILE*                     fplog,
                           const t_commrec*          cr,
                           gmx_walltime_accounting_t walltime_accounting,
                           gmx_wallcycle*            wcycle,
                           const char*               name)
{
    std::cout << "## /src/gromacs/mdrun/minimize.cpp: start of print_em_start()" << std::endl;

    walltime_accounting_start_time(walltime_accounting);
    wallcycle_start(wcycle, WallCycleCounter::Run);
    print_start(fplog, cr, walltime_accounting, name);

    std::cout << "## /src/gromacs/mdrun/minimize.cpp: end of print_em_start()" << std::endl;

}

//! Stop counting time for EM
static void em_time_end(gmx_walltime_accounting_t walltime_accounting, gmx_wallcycle* wcycle)
{
    wallcycle_stop(wcycle, WallCycleCounter::Run);

    walltime_accounting_end_time(walltime_accounting);
}

//! Printing a log file and console header
static void sp_header(FILE* out, const char* minimizer, real ftol, int nsteps)
{

    std::cout << "## /src/gromacs/mdrun/minimize.cpp: start of sp_header()" << std::endl;

    fprintf(out, "\n");
    fprintf(out, "%s:\n", minimizer);
    fprintf(out, "   Tolerance (Fmax)   = %12.5e\n", ftol);
    fprintf(out, "   Number of steps    = %12d\n", nsteps);

    std::cout << "## /src/gromacs/mdrun/minimize.cpp: end of sp_header()" << std::endl;
}

//! Print warning message
static void warn_step(FILE* fp, real ftol, real fmax, gmx_bool bLastStep, gmx_bool bConstrain)
{
    constexpr bool realIsDouble = GMX_DOUBLE;
    char           buffer[2048];

    if (!std::isfinite(fmax))
    {
        sprintf(buffer,
                "\nEnergy minimization has stopped because the force "
                "on at least one atom is not finite. This usually means "
                "atoms are overlapping. Modify the input coordinates to "
                "remove atom overlap or use soft-core potentials with "
                "the free energy code to avoid infinite forces.\n%s",
                !realIsDouble ? "You could also be lucky that switching to double precision "
                                "is sufficient to obtain finite forces.\n"
                              : "");
    }
    else if (bLastStep)
    {
        sprintf(buffer,
                "\nEnergy minimization reached the maximum number "
                "of steps before the forces reached the requested "
                "precision Fmax < %g.\n",
                ftol);
    }
    else
    {
        sprintf(buffer,
                "\nEnergy minimization has stopped, but the forces have "
                "not converged to the requested precision Fmax < %g (which "
                "may not be possible for your system). It stopped "
                "because the algorithm tried to make a new step whose size "
                "was too small, or there was no change in the energy since "
                "last step. Either way, we regard the minimization as "
                "converged to within the available machine precision, "
                "given your starting configuration and EM parameters.\n%s%s",
                ftol,
                !realIsDouble ? "\nDouble precision normally gives you higher accuracy, but "
                                "this is often not needed for preparing to run molecular "
                                "dynamics.\n"
                              : "",
                bConstrain ? "You might need to increase your constraint accuracy, or turn\n"
                             "off constraints altogether (set constraints = none in mdp file)\n"
                           : "");
    }

    fputs(wrap_lines(buffer, 78, 0, FALSE), stderr);
    fputs(wrap_lines(buffer, 78, 0, FALSE), fp);
}

//! Print message about convergence of the EM
static void print_converged(FILE*             fp,
                            const char*       alg,
                            real              ftol,
                            int64_t           count,
                            gmx_bool          bDone,
                            int64_t           nsteps,
                            const em_state_t* ems,
                            double            sqrtNumAtoms)
{


    std::cout << "## /src/gromacs/mdrun/minimize.cpp: start of print_converged()" << std::endl;

    char buf[STEPSTRSIZE];

    if (bDone)
    {
        fprintf(fp, "\n%s converged to Fmax < %g in %s steps\n", alg, ftol, gmx_step_str(count, buf));
    }
    else if (count < nsteps)
    {
        fprintf(fp,
                "\n%s converged to machine precision in %s steps,\n"
                "but did not reach the requested Fmax < %g.\n",
                alg,
                gmx_step_str(count, buf),
                ftol);
    }
    else
    {
        fprintf(fp, "\n%s did not converge to Fmax < %g in %s steps.\n", alg, ftol, gmx_step_str(count, buf));
    }

#if GMX_DOUBLE
    fprintf(fp, "Potential Energy  = %21.14e\n", ems->epot);
    fprintf(fp, "Maximum force     = %21.14e on atom %d\n", ems->fmax, ems->a_fmax + 1);
    fprintf(fp, "Norm of force     = %21.14e\n", ems->fnorm / sqrtNumAtoms);
#else
    fprintf(fp, "Potential Energy  = %14.7e\n", ems->epot);
    fprintf(fp, "Maximum force     = %14.7e on atom %d\n", ems->fmax, ems->a_fmax + 1);
    fprintf(fp, "Norm of force     = %14.7e\n", ems->fnorm / sqrtNumAtoms);
#endif

    std::cout << "## /src/gromacs/mdrun/minimize.cpp: end of print_converged()" << std::endl;

}

//! Compute the norm and max of the force array in parallel
static void get_f_norm_max(const t_commrec*               cr,
                           const t_grpopts*               opts,
                           t_mdatoms*                     mdatoms,
                           gmx::ArrayRef<const gmx::RVec> f,
                           real*                          fnorm,
                           real*                          fmax,
                           int*                           a_fmax)
{

    std::cout << "# void get_f_norm_max()" << std::endl;

    double fnorm2, *sum;
    real   fmax2, fam;
    int    la_max, a_max, start, end, i, m, gf;

    /* This routine finds the largest force and returns it.
     * On parallel machines the global max is taken.
     */
    fnorm2 = 0;
    fmax2  = 0;
    la_max = -1;
    start  = 0;
    end    = mdatoms->homenr;
    if (!mdatoms->cFREEZE.empty())
    {
        for (i = start; i < end; i++)
        {
            gf  = mdatoms->cFREEZE[i];
            fam = 0;
            for (m = 0; m < DIM; m++)
            {
                if (!opts->nFreeze[gf][m])
                {
                    fam += gmx::square(f[i][m]);
                }
            }
            fnorm2 += fam;
            if (fam > fmax2)
            {
                fmax2  = fam;
                la_max = i;
            }
        }
    }
    else
    {
        for (i = start; i < end; i++)
        {
            fam = norm2(f[i]);
            fnorm2 += fam;
            if (fam > fmax2)
            {
                fmax2  = fam;
                la_max = i;
            }
        }
    }

    if (la_max >= 0 && haveDDAtomOrdering(*cr))
    {
        a_max = cr->dd->globalAtomIndices[la_max];
    }
    else
    {
        a_max = la_max;
    }
    if (PAR(cr))
    {
        snew(sum, 2 * cr->nnodes + 1);
        sum[2 * cr->nodeid]     = fmax2;
        sum[2 * cr->nodeid + 1] = a_max;
        sum[2 * cr->nnodes]     = fnorm2;
        gmx_sumd(2 * cr->nnodes + 1, sum, cr);
        fnorm2 = sum[2 * cr->nnodes];
        /* Determine the global maximum */
        for (i = 0; i < cr->nnodes; i++)
        {
            if (sum[2 * i] > fmax2)
            {
                fmax2 = sum[2 * i];
                a_max = gmx::roundToInt(sum[2 * i + 1]);
            }
        }
        sfree(sum);
    }

    if (fnorm)
    {
        *fnorm = sqrt(fnorm2);
    }
    if (fmax)
    {
        *fmax = sqrt(fmax2);
    }
    if (a_fmax)
    {
        *a_fmax = a_max;
    }
}

//! Compute the norm of the force
static void get_state_f_norm_max(const t_commrec* cr, const t_grpopts* opts, t_mdatoms* mdatoms, em_state_t* ems)
{
    get_f_norm_max(cr, opts, mdatoms, ems->f.view().force(), &ems->fnorm, &ems->fmax, &ems->a_fmax);
}

//! Initialize the energy minimization
static void init_em(FILE*                fplog,
                    const gmx::MDLogger& mdlog,
                    const char*          title,
                    const t_commrec*     cr,
                    const t_inputrec*    ir,
                    gmx::ImdSession*     imdSession,
                    pull_t*              pull_work,
                    t_state*             state_global,
                    const gmx_mtop_t&    top_global,
                    em_state_t*          ems,
                    gmx_localtop_t*      top,
                    t_nrnb*              nrnb,
                    t_forcerec*          fr,
                    gmx::MDAtoms*        mdAtoms,
                    gmx_global_stat_t*   gstat,
                    VirtualSitesHandler* vsite,
                    gmx::Constraints*    constr,
                    gmx_shellfc_t**      shellfc)
{

    std::cout << "## /src/gromacs/mdrun/minimize.cpp: start of init_em()" << std::endl;

    real dvdl_constr;

    if (fplog)
    {
        fprintf(fplog, "Initiating %s\n", title);
    }
    
    std::cout << "Rank in default communicator is: " << cr->rankInDefaultCommunicator << std::endl;
    std::cout << "Size of default communicator is: " << cr->sizeOfDefaultCommunicator << std::endl;

    if (MAIN(cr))
    {
        state_global->ngtc = 0;
    }
    int*                fep_state = MAIN(cr) ? &state_global->fep_state : nullptr;

    gmx::ArrayRef<real> lambda    = MAIN(cr) ? state_global->lambda : gmx::ArrayRef<real>();

    initialize_lambdas(
            fplog, ir->efep, ir->bSimTemp, *ir->fepvals, ir->simtempvals->temperatures, nullptr, MAIN(cr), fep_state, lambda);

    if (ir->eI == IntegrationAlgorithm::NM)
    {
        GMX_ASSERT(shellfc != nullptr, "With NM we always support shells");

        *shellfc = init_shell_flexcon(stdout,
                                      top_global,
                                      constr ? constr->numFlexibleConstraints() : 0,
                                      ir->nstcalcenergy,
                                      haveDDAtomOrdering(*cr),
                                      thisRankHasDuty(cr, DUTY_PME));
    }
    else
    {
        GMX_ASSERT(EI_ENERGY_MINIMIZATION(ir->eI),
                   "This else currently only handles energy minimizers, consider if your algorithm "
                   "needs shell/flexible-constraint support");

        /* With energy minimization, shells and flexible constraints are
         * automatically minimized when treated like normal DOFS.
         */
        if (shellfc != nullptr)
        {
            *shellfc = nullptr;
        }
    }

    if (haveDDAtomOrdering(*cr))
    {
        // Local state only becomes valid now.
        dd_init_local_state(*cr->dd, state_global, &ems->s);

        /* Distribute the charge groups over the nodes from the main node */
        dd_partition_system(fplog,
                            mdlog,
                            ir->init_step,
                            cr,
                            TRUE,
                            state_global,
                            top_global,
                            *ir,
                            imdSession,
                            pull_work,
                            &ems->s,
                            &ems->f,
                            mdAtoms,
                            top,
                            fr,
                            vsite,
                            constr,
                            nrnb,
                            nullptr,
                            FALSE);
        dd_store_state(*cr->dd, &ems->s);
    }
    else
    {
        state_change_natoms(state_global, state_global->natoms);
        /* Just copy the state */
        ems->s = *state_global;
        state_change_natoms(&ems->s, ems->s.natoms);

        mdAlgorithmsSetupAtomData(
                cr, *ir, top_global, top, fr, &ems->f, mdAtoms, constr, vsite, shellfc ? *shellfc : nullptr);
    }

    update_mdatoms(mdAtoms->mdatoms(), ems->s.lambda[FreeEnergyPerturbationCouplingType::Mass]);

    if (constr)
    {
        // TODO how should this cross-module support dependency be managed?
        if (ir->eConstrAlg == ConstraintAlgorithm::Shake && gmx_mtop_ftype_count(top_global, F_CONSTR) > 0)
        {
            gmx_fatal(FARGS,
                      "Can not do energy minimization with %s, use %s\n",
                      enumValueToString(ConstraintAlgorithm::Shake),
                      enumValueToString(ConstraintAlgorithm::Lincs));
        }

        if (!ir->bContinuation)
        {
            /* Constrain the starting coordinates */
            bool needsLogging  = true;
            bool computeEnergy = true;
            bool computeVirial = false;
            dvdl_constr        = 0;
            constr->apply(needsLogging,
                          computeEnergy,
                          -1,
                          0,
                          1.0,
                          ems->s.x.arrayRefWithPadding(),
                          ems->s.x.arrayRefWithPadding(),
                          ArrayRef<RVec>(),
                          ems->s.box,
                          ems->s.lambda[FreeEnergyPerturbationCouplingType::Fep],
                          &dvdl_constr,
                          gmx::ArrayRefWithPadding<RVec>(),
                          computeVirial,
                          nullptr,
                          gmx::ConstraintVariable::Positions);
        }
    }

    if (PAR(cr))
    {
        *gstat = global_stat_init(ir);
    }
    else
    {
        *gstat = nullptr;
    }

    calc_shifts(ems->s.box, fr->shift_vec);



    std::cout << "## /src/gromacs/mdrun/minimize.cpp: end of init_em()" << std::endl;

}

//! Finalize the minimization
static void finish_em(const t_commrec*          cr,
                      gmx_mdoutf_t              outf,
                      gmx_walltime_accounting_t walltime_accounting,
                      gmx_wallcycle*            wcycle)
{
    if (!thisRankHasDuty(cr, DUTY_PME))
    {
        /* Tell the PME only node to finish */
        gmx_pme_send_finish(cr);
    }

    done_mdoutf(outf);

    em_time_end(walltime_accounting, wcycle);
}

//! Swap two different EM states during minimization
static void swap_em_state(em_state_t** ems1, em_state_t** ems2)
{
    em_state_t* tmp;

    tmp   = *ems1;
    *ems1 = *ems2;
    *ems2 = tmp;
}

//! Save the EM trajectory
static void write_em_traj(FILE*               fplog,
                          const t_commrec*    cr,
                          gmx_mdoutf_t        outf,
                          gmx_bool            bX,
                          gmx_bool            bF,
                          const char*         confout,
                          const gmx_mtop_t&   top_global,
                          const t_inputrec*   ir,
                          int64_t             step,
                          em_state_t*         state,
                          t_state*            state_global,
                          ObservablesHistory* observablesHistory)
{
    int mdof_flags = 0;

    if (bX)
    {
        mdof_flags |= MDOF_X;
    }
    if (bF)
    {
        mdof_flags |= MDOF_F;
    }

    /* If we want IMD output, set appropriate MDOF flag */
    if (ir->bIMD)
    {
        mdof_flags |= MDOF_IMD;
    }

    gmx::WriteCheckpointDataHolder checkpointDataHolder;
    mdoutf_write_to_trajectory_files(fplog,
                                     cr,
                                     outf,
                                     mdof_flags,
                                     top_global.natoms,
                                     step,
                                     static_cast<double>(step),
                                     &state->s,
                                     state_global,
                                     observablesHistory,
                                     state->f.view().force(),
                                     &checkpointDataHolder);

    if (confout != nullptr)
    {
        if (haveDDAtomOrdering(*cr))
        {
            /* If bX=true, x was collected to state_global in the call above */
            if (!bX)
            {
                auto globalXRef = MAIN(cr) ? state_global->x : gmx::ArrayRef<gmx::RVec>();
                dd_collect_vec(
                        cr->dd, state->s.ddp_count, state->s.ddp_count_cg_gl, state->s.cg_gl, state->s.x, globalXRef);
            }
        }
        else
        {
            /* Copy the local state pointer */
            state_global = &state->s;
        }

        if (MAIN(cr))
        {
            if (ir->pbcType != PbcType::No && !ir->bPeriodicMols && haveDDAtomOrdering(*cr))
            {
                /* Make molecules whole only for confout writing */
                do_pbc_mtop(ir->pbcType, state->s.box, &top_global, state_global->x.rvec_array());
            }

            write_sto_conf_mtop(confout,
                                *top_global.name,
                                top_global,
                                state_global->x.rvec_array(),
                                nullptr,
                                ir->pbcType,
                                state->s.box);
        }
    }
}




//! Prepare EM for using domain decomposition parallellization
static void em_dd_partition_system(FILE*                fplog,
                                   const gmx::MDLogger& mdlog,
                                   int                  step,
                                   const t_commrec*     cr,
                                   const gmx_mtop_t&    top_global,
                                   const t_inputrec*    ir,
                                   gmx::ImdSession*     imdSession,
                                   pull_t*              pull_work,
                                   em_state_t*          ems,
                                   gmx_localtop_t*      top,
                                   gmx::MDAtoms*        mdAtoms,
                                   t_forcerec*          fr,
                                   VirtualSitesHandler* vsite,
                                   gmx::Constraints*    constr,
                                   t_nrnb*              nrnb,
                                   gmx_wallcycle*       wcycle)
{


    std::cout << " ## em_dd_partition_system()" << std::endl;

    /* Repartition the domain decomposition */
    dd_partition_system(fplog,
                        mdlog,
                        step,
                        cr,
                        FALSE,
                        nullptr,
                        top_global,
                        *ir,
                        imdSession,
                        pull_work,
                        &ems->s,
                        &ems->f,
                        mdAtoms,
                        top,
                        fr,
                        vsite,
                        constr,
                        nrnb,
                        wcycle,
                        FALSE);
    dd_store_state(*cr->dd, &ems->s);
}

namespace
{

//! Copy coordinates, OpenMP parallelized, from \p refCoords to coords
void setCoordinates(std::vector<RVec>* coords, ArrayRef<const RVec> refCoords)
{
    coords->resize(refCoords.size());

    const int gmx_unused nthreads = gmx_omp_nthreads_get(ModuleMultiThread::Update);
#pragma omp parallel for num_threads(nthreads) schedule(static)
    for (int i = 0; i < ssize(refCoords); i++)
    {
        (*coords)[i] = refCoords[i];
    }
}

//! Returns the maximum difference an atom moved between two coordinate sets, over all ranks
real maxCoordinateDifference(ArrayRef<const RVec> coords1, ArrayRef<const RVec> coords2, MPI_Comm mpiCommMyGroup)
{
    GMX_RELEASE_ASSERT(coords1.size() == coords2.size(), "Coordinate counts should match");

    real maxDiffSquared = 0;

#ifndef _MSC_VER // Visual Studio has no support for reduction(max)
    const int gmx_unused nthreads = gmx_omp_nthreads_get(ModuleMultiThread::Update);
#    pragma omp parallel for reduction(max : maxDiffSquared) num_threads(nthreads) schedule(static)
#endif
    for (int i = 0; i < ssize(coords1); i++)
    {
        maxDiffSquared = std::max(maxDiffSquared, gmx::norm2(coords1[i] - coords2[i]));
    }

#if GMX_MPI
    int numRanks = 1;
    if (mpiCommMyGroup != MPI_COMM_NULL)
    {
        MPI_Comm_size(mpiCommMyGroup, &numRanks);
    }
    if (numRanks > 1)
    {
        real maxDiffSquaredReduced;
        MPI_Allreduce(
                &maxDiffSquared, &maxDiffSquaredReduced, 1, GMX_DOUBLE ? MPI_DOUBLE : MPI_FLOAT, MPI_MAX, mpiCommMyGroup);
        maxDiffSquared = maxDiffSquaredReduced;
    }
#else
    GMX_UNUSED_VALUE(mpiCommMyGroup);
#endif

    return std::sqrt(maxDiffSquared);
}

/*! \brief Class to handle the work of setting and doing an energy evaluation.
 *
 * This class is a mere aggregate of parameters to pass to evaluate an
 * energy, so that future changes to names and types of them consume
 * less time when refactoring other code.
 *
 * Aggregate initialization is used, for which the chief risk is that
 * if a member is added at the end and not all initializer lists are
 * updated, then the member will be value initialized, which will
 * typically mean initialization to zero.
 *
 * Use a braced initializer list to construct one of these. */
class EnergyEvaluator
{
public:
    /*! \brief Evaluates an energy on the state in \c ems.
     *
     * \todo In practice, the same objects mu_tot, vir, and pres
     * are always passed to this function, so we would rather have
     * them as data members. However, their C-array types are
     * unsuited for aggregate initialization. When the types
     * improve, the call signature of this method can be reduced.
     */
    void run(em_state_t* ems, rvec mu_tot, tensor vir, tensor pres, int64_t count, gmx_bool bFirst, int64_t step);
    //! Handles logging (deprecated).
    FILE* fplog;
    //! Handles logging.
    const gmx::MDLogger& mdlog;
    //! Handles communication.
    const t_commrec* cr;
    //! Coordinates multi-simulations.
    const gmx_multisim_t* ms;
    //! Holds the simulation topology.
    const gmx_mtop_t& top_global;
    //! Holds the domain topology.
    gmx_localtop_t* top;
    //! User input options.
    const t_inputrec* inputrec;
    //! The Interactive Molecular Dynamics session.
    gmx::ImdSession* imdSession;
    //! The pull work object.
    pull_t* pull_work;
    //! Data for rotational pulling.
    gmx_enfrot* enforcedRotation;
    //! Manages flop accounting.
    t_nrnb* nrnb;
    //! Manages wall cycle accounting.
    gmx_wallcycle* wcycle;
    //! Legacy coordinator of global reduction.
    gmx_global_stat_t gstat;
    //! Coordinates reduction for observables
    gmx::ObservablesReducer* observablesReducer;
    //! Handles virtual sites.
    VirtualSitesHandler* vsite;
    //! Handles constraints.
    gmx::Constraints* constr;
    //! Per-atom data for this domain.
    gmx::MDAtoms* mdAtoms;
    //! Handles how to calculate the forces.
    t_forcerec* fr;
    //! Schedule of force-calculation work each step for this task.
    MdrunScheduleWorkload* runScheduleWork;
    //! Stores the computed energies.
    gmx_enerdata_t* enerd;
    //! The DD partitioning count at which the pair list was generated
    int ddpCountPairSearch;
    //! The local coordinates that were used for pair searching, stored for computing displacements
    std::vector<RVec> pairSearchCoordinates;
};







void EnergyEvaluator::run(em_state_t* ems, rvec mu_tot, tensor vir, tensor pres, int64_t count, gmx_bool bFirst, int64_t step)
{

    std::cout << "## EnergyEvaluator::run()" << std::endl;

    real     t;
    gmx_bool bNS;
    tensor   force_vir, shake_vir, ekin;
    real     dvdl_constr;
    real     terminate = 0;

    /* Set the time to the initial time, the time does not change during EM */
    t = inputrec->init_t;

    if (vsite)
    {   
        std::cout << "vsite is TRUE" << std::endl;
        vsite->construct(ems->s.x, {}, ems->s.box, gmx::VSiteOperation::Positions);
    }

    // Compute the buffer size of the pair list
    const real bufferSize = inputrec->rlist - std::max(inputrec->rcoulomb, inputrec->rvdw);

    std::cout << "# BufferSize is: " << bufferSize << std::endl;

    if (bFirst || bufferSize <= 0 || ems->s.ddp_count != ddpCountPairSearch)
    {
        /* This is the first state or an old state used before the last ns */
        bNS = TRUE;
    }
    else
    {
        // We need to generate a new pairlist when one atom moved more than half the buffer size
        ArrayRef<const RVec> localCoordinates = ArrayRef<const RVec>(ems->s.x).subArray(0, mdAtoms->mdatoms()->homenr);

        std::cout << localCoordinates.size() << std::endl;


        // So if atom is moved more than half the buffer size?
        bNS = 2 * maxCoordinateDifference(pairSearchCoordinates, localCoordinates, cr->mpi_comm_mygroup) > bufferSize;
    }

    if (bNS)
    {

        std::cout << "### bNS is TRUE" << std::endl;

        if (haveDDAtomOrdering(*cr))
        {

            std::cout << "#### haveDDAtomOrdering is TRUE: Repartitioning the domain decomposition" << std::endl;

            /* Repartition the domain decomposition */
            em_dd_partition_system(fplog,
                                   mdlog,
                                   count,
                                   cr,
                                   top_global,
                                   inputrec,
                                   imdSession,
                                   pull_work,
                                   ems,
                                   top,
                                   mdAtoms,
                                   fr,
                                   vsite,
                                   constr,
                                   nrnb,
                                   wcycle);
            ddpCountPairSearch = cr->dd->ddp_count;
        }
        else
        {
            // Without DD we increase the search counter here
            ddpCountPairSearch++;
            // Store the count in the state, so we check whether we later need
            // to do pair search after resetting to this, by then, old state
            ems->s.ddp_count = ddpCountPairSearch;
        }
    }

    /* Store the local coordinates that will be used in the pair search, after we re-partitioned */
    if (bufferSize > 0 && bNS)
    {
        ArrayRef<const RVec> localCoordinates =
                constArrayRefFromArray(ems->s.x.data(), mdAtoms->mdatoms()->homenr);
        setCoordinates(&pairSearchCoordinates, localCoordinates);
    }

    fr->longRangeNonbondeds->updateAfterPartition(*mdAtoms->mdatoms());

    /* Calc force & energy on new trial position  */
    /* do_force always puts the charge groups in the box and shifts again
     * We do not unshift, so molecules are always whole in congrad.c
     */
    do_force(fplog,
             cr,
             ms,
             *inputrec,
             nullptr,
             enforcedRotation,
             imdSession,
             pull_work,
             count,
             nrnb,
             wcycle,
             top,
             ems->s.box,
             ems->s.x.arrayRefWithPadding(),
             &ems->s.hist,
             &ems->f.view(),
             force_vir,
             mdAtoms->mdatoms(),
             enerd,
             ems->s.lambda,
             fr,
             runScheduleWork,
             vsite,
             mu_tot,
             t,
             nullptr,
             fr->longRangeNonbondeds.get(),
             GMX_FORCE_STATECHANGED | GMX_FORCE_ALLFORCES | GMX_FORCE_VIRIAL | GMX_FORCE_ENERGY
                     | (bNS ? GMX_FORCE_NS : 0),
             DDBalanceRegionHandler(cr));

    /* Clear the unused shake virial and pressure */
    clear_mat(shake_vir);
    clear_mat(pres);

    /* Communicate stuff when parallel */
    if (PAR(cr) && inputrec->eI != IntegrationAlgorithm::NM)
    {
        wallcycle_start(wcycle, WallCycleCounter::MoveE);

        global_stat(*gstat,
                    cr,
                    enerd,
                    force_vir,
                    shake_vir,
                    *inputrec,
                    nullptr,
                    nullptr,
                    std::vector<real>(1, terminate),
                    FALSE,
                    CGLO_ENERGY | CGLO_PRESSURE | CGLO_CONSTRAINT,
                    step,
                    observablesReducer);

        wallcycle_stop(wcycle, WallCycleCounter::MoveE);
    }

    if (fr->dispersionCorrection)
    {
        /* Calculate long range corrections to pressure and energy */
        const DispersionCorrection::Correction correction = fr->dispersionCorrection->calculate(
                ems->s.box, ems->s.lambda[FreeEnergyPerturbationCouplingType::Vdw]);

        enerd->term[F_DISPCORR] = correction.energy;
        enerd->term[F_EPOT] += correction.energy;
        enerd->term[F_PRES] += correction.pressure;
        enerd->term[F_DVDL] += correction.dvdl;
    }
    else
    {
        enerd->term[F_DISPCORR] = 0;
    }

    ems->epot = enerd->term[F_EPOT];

    if (constr)
    {
        /* Project out the constraint components of the force */
        bool needsLogging  = false;
        bool computeEnergy = false;
        bool computeVirial = true;
        dvdl_constr        = 0;
        auto f             = ems->f.view().forceWithPadding();
        constr->apply(needsLogging,
                      computeEnergy,
                      count,
                      0,
                      1.0,
                      ems->s.x.arrayRefWithPadding(),
                      f,
                      f.unpaddedArrayRef(),
                      ems->s.box,
                      ems->s.lambda[FreeEnergyPerturbationCouplingType::Bonded],
                      &dvdl_constr,
                      gmx::ArrayRefWithPadding<RVec>(),
                      computeVirial,
                      shake_vir,
                      gmx::ConstraintVariable::ForceDispl);
        enerd->term[F_DVDL_CONSTR] += dvdl_constr;
        m_add(force_vir, shake_vir, vir);
    }
    else
    {
        copy_mat(force_vir, vir);
    }

    clear_mat(ekin);
    enerd->term[F_PRES] = calc_pres(fr->pbcType, inputrec->nwall, ems->s.box, ekin, vir, pres);

    if (inputrec->efep != FreeEnergyPerturbationType::No)
    {
        accumulateKineticLambdaComponents(enerd, ems->s.lambda, *inputrec->fepvals);
    }

    if (EI_ENERGY_MINIMIZATION(inputrec->eI))
    {
        get_state_f_norm_max(cr, &(inputrec->opts), mdAtoms->mdatoms(), ems);
    }
}

} // namespace




















//! Parallel utility summing energies and forces
static double reorder_partsum(const t_commrec*  cr,
                              const t_grpopts*  opts,
                              const gmx_mtop_t& top_global,
                              const em_state_t* s_min,
                              const em_state_t* s_b)
{
    if (debug)
    {
        fprintf(debug, "Doing reorder_partsum\n");
    }

    auto fm = s_min->f.view().force();
    auto fb = s_b->f.view().force();

    /* Collect fm in a global vector fmg.
     * This conflicts with the spirit of domain decomposition,
     * but to fully optimize this a much more complicated algorithm is required.
     */
    const int natoms = top_global.natoms;
    rvec*     fmg;
    snew(fmg, natoms);

    gmx::ArrayRef<const int> indicesMin = s_min->s.cg_gl;
    int                      i          = 0;
    for (int a : indicesMin)
    {
        copy_rvec(fm[i], fmg[a]);
        i++;
    }
    gmx_sum(top_global.natoms * 3, fmg[0], cr);

    /* Now we will determine the part of the sum for the cgs in state s_b */
    gmx::ArrayRef<const int> indicesB = s_b->s.cg_gl;

    double partsum                        = 0;
    i                                     = 0;
    int                                gf = 0;
    gmx::ArrayRef<const unsigned char> grpnrFREEZE =
            top_global.groups.groupNumbers[SimulationAtomGroupType::Freeze];
    for (int a : indicesB)
    {
        if (!grpnrFREEZE.empty())
        {
            gf = grpnrFREEZE[i];
        }
        for (int m = 0; m < DIM; m++)
        {
            if (!opts->nFreeze[gf][m])
            {
                partsum += (fb[i][m] - fmg[a][m]) * fb[i][m];
            }
        }
        i++;
    }

    sfree(fmg);

    return partsum;
}

//! Print some stuff, like beta, whatever that means.
static real pr_beta(const t_commrec*  cr,
                    const t_grpopts*  opts,
                    t_mdatoms*        mdatoms,
                    const gmx_mtop_t& top_global,
                    const em_state_t* s_min,
                    const em_state_t* s_b)
{
    double sum;

    /* This is just the classical Polak-Ribiere calculation of beta;
     * it looks a bit complicated since we take freeze groups into account,
     * and might have to sum it in parallel runs.
     */

    if (!haveDDAtomOrdering(*cr)
        || (s_min->s.ddp_count == cr->dd->ddp_count && s_b->s.ddp_count == cr->dd->ddp_count))
    {
        auto fm = s_min->f.view().force();
        auto fb = s_b->f.view().force();
        sum     = 0;
        int gf  = 0;
        /* This part of code can be incorrect with DD,
         * since the atom ordering in s_b and s_min might differ.
         */
        for (int i = 0; i < mdatoms->homenr; i++)
        {
            if (!mdatoms->cFREEZE.empty())
            {
                gf = mdatoms->cFREEZE[i];
            }
            for (int m = 0; m < DIM; m++)
            {
                if (!opts->nFreeze[gf][m])
                {
                    sum += (fb[i][m] - fm[i][m]) * fb[i][m];
                }
            }
        }
    }
    else
    {
        /* We need to reorder cgs while summing */
        sum = reorder_partsum(cr, opts, top_global, s_min, s_b);
    }
    if (PAR(cr))
    {
        gmx_sumd(1, &sum, cr);
    }

    return sum / gmx::square(s_min->fnorm);
}

namespace gmx
{

void LegacySimulator::do_cg()
{

    std::cout << "## /src/gromacs/mdrun/minimize.cpp: start of LegacySimulator::do_cg()" << std::endl;

    // This was deleted as only steepest descent was used

    std::cout << "## /src/gromacs/mdrun/minimize.cpp: end of LegacySimulator::do_cg()" << std::endl;

}


void LegacySimulator::do_lbfgs()
{

    
    std::cout << "## /src/gromacs/mdrun/minimize.cpp: start of LegacySimulator::do_lbfgs()" << std::endl;

    // This was deleted as only steepest descens is used at this point

    std::cout << "## /src/gromacs/mdrun/minimize.cpp: end of LegacySimulator::do_lbfgs()" << std::endl;

}

void LegacySimulator::do_nm()
{
    std::cout << "## /src/gromacs/mdrun/minimize.cpp: start of LegacySimulator::do_nm()" << std::endl;

    // This was deleted as only steepest descent is used

    std::cout << "## /src/gromacs/mdrun/minimize.cpp: end of LegacySimulator::do_nm()" << std::endl;
}







static bool do_em_step(const t_commrec*                          cr,
                       const t_inputrec*                         ir,
                       t_mdatoms*                                md,
                       em_state_t*                               ems1,
                       real                                      a,
                       gmx::ArrayRefWithPadding<const gmx::RVec> force,
                       em_state_t*                               ems2,
                       gmx::Constraints*                         constr,
                       int64_t                                   count)

{

    std::cout << " \n\n     ## minimize.cpp: do_em_step()" << std::endl;

    t_state *    s1, *s2;
    int          start, end;
    real         dvdl_constr;
    int nthreads gmx_unused;

    std::cout << "          ## int nthreads is: " << nthreads << std::endl;

    bool validStep = true;

    s1 = &ems1->s;
    s2 = &ems2->s;


    // Here the state is copied?
    // s1 is the previous state and s2 is new state that is to be calculated?

    std::cout << "          ## Number of atoms in s1: " << s1->natoms << std::endl;
    std::cout << "          ## Number of temperature coupling groups in s1: " << s1->ngtc << std::endl;



    if (haveDDAtomOrdering(*cr) && s1->ddp_count != cr->dd->ddp_count)
    {
        gmx_incons("state mismatch in do_em_step");
    }

    s2->flags = s1->flags;

    if (s2->natoms != s1->natoms)
    {
        state_change_natoms(s2, s1->natoms);
        ems2->f.resize(s2->natoms);
    }
    if (haveDDAtomOrdering(*cr) && s2->cg_gl.size() != s1->cg_gl.size())
    {
        s2->cg_gl.resize(s1->cg_gl.size());
    }

    /*
    for(int i = 0; i<DIM; i++){
        for(int j = 0; j< DIM; j++){
            std::cout << " BOX" << std::endl;
            std::cout << s1->box[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    */


    copy_mat(s1->box, s2->box);
    /* Copy free energy state */
    s2->lambda = s1->lambda;
    copy_mat(s1->box, s2->box);

    start = 0;
    end   = md->homenr;

    nthreads = gmx_omp_nthreads_get(ModuleMultiThread::Update);

    std::cout << " # Number of threads is: " << nthreads << std::endl;

#pragma omp parallel num_threads(nthreads)
    {
        const rvec* x1 = s1->x.rvec_array();

        rvec*       x2 = s2->x.rvec_array();


        // Forces on tässä
        const rvec* f  = as_rvec_array(force.unpaddedArrayRef().data());

        int gf = 0; 

        

        // Tästä alkaa oma


        //std::cout << " \n\n # Content of first 5 rvecs in x \n" << std::endl;
        
        
         // /* Tässä kirjoitetaan koordinaatit kullakin iteraatiolla tiedostoon

        const rvec* xcopy = s1->x.rvec_array();

        const rvec* xcopy_s2 = s2->x.rvec_array();
        
        //std::string path = "/home/tapio/Desktop/Gromacs_coordinates/coordinates"+std::to_string(count);
        
        // This is for home setup
        //std::string path_s1 = "/home/tee/Desktop/Gromacs_analytics/coordinates/s1/s1_"+std::to_string(count);
        //std::string path_s2 = "/home/tee/Desktop/Gromacs_analytics/coordinates/s2/s2_"+std::to_string(count);


        // This is for rtx1
        std::string path_s1 = "/home/tapio/PROJECT/GROMACS_DATA/coordinates/s1/s1_"+std::to_string(count);
        std::string path_s2 = "/home/tapio/PROJECT/GROMACS_DATA/coordinates/s2/s2_"+std::to_string(count);


        std::ofstream coordinates_s1(path_s1);

        std::ofstream coordinates_s2(path_s2);


        coordinates_s1 << "Contents of s1->x" << std::endl;
        coordinates_s2 << "Contents of s2->x" << std::endl;

        if (!coordinates_s1 || !coordinates_s2) {
            std::cerr << "Error opening file for writing.\n";
            return 1;
        }
        for(int j = 0; j< s1->natoms; j++){
            
            coordinates_s1 << j << ": ";

            coordinates_s2 << j << ": ";

            for(int i = 0; i<DIM; i++){
            
                coordinates_s1 << xcopy[j][i] << " ";
                coordinates_s2 << xcopy_s2[j][i] << " ";
            }
            coordinates_s1 << std::endl;
            coordinates_s2 << std::endl;

            //xcopy++;
        }

        coordinates_s1.close();
        coordinates_s2.close();
        

        
        //size_t size1 = sizeof(x1);
        //size_t size2 = sizeof(*x1);

        //std::cout<< "size of x1 is : " << size1 << std::endl;
        //std::cout << "size of *x1 is : " << size2 << std::endl;

        //size_t size3 = sizeof(x1[0]);
        //size_t size4 = sizeof(*x1[0]);

        //std::cout<< "size of x1[0] is : " << size3 << std::endl;
        //std::cout << "size of *x1[0] is : " << size4 << std::endl;

        //std::cout << " size of float is : " << sizeof(float) << std::endl;

        //int sizev = sizeof(*x1)/sizeof(float);

        //std::cout << "# size of rvec x1 should be: 3 and it is :" << sizev << std::endl; 

         //*/

        /* Writing forces to file
        std::string path = "/home/tee/Desktop/Gromacs_analytics/forces/f_"+std::to_string(count);
        std::ofstream forces(path);


        forces << "Contents of f. Value of a is: " <<  a <<  std::endl;


        const rvec* fcopy  = as_rvec_array(force.unpaddedArrayRef().data());

        for(int i = 0; i< s1->natoms; i++){
            
            forces << i << ": ";
            

            for(int j = 0; j<DIM; j++){
            
                forces << fcopy[i][j] << " ";
            
            }

            forces << std::endl;
        }

        forces.close();

         */

        for(int i = 0; i<DIM; i++){
            std::cout << *x2[i] << " ";
        }
        std::cout << std::endl;
        

        std::cout << "START: " << start << " , END: " << end << std::endl;

        std::cout << "VALUE OF a is: " << a << std::endl;

        // Oma loppuu

                  
#pragma omp for schedule(static) nowait
        for (int i = start; i < end; i++)
        {
            //std::cout << "start loop: " << i << std::endl;
            try
            {
                if (!md->cFREEZE.empty())
                {
                    std::cout << "  !md->cFREEZE.empty()" << std::endl;
                    gf = md->cFREEZE[i];
                }

                // Tässä käydään läpi vektorin kolme dimensiota
                for (int m = 0; m < DIM; m++)
                {
                    //std::cout << "  m loop: " << m << std::endl;
                    //std::cout << "          x2 value is set here" << std::endl;
                    if (ir->opts.nFreeze[gf][m])
                    {   
                        // Tässä kopioidaan arvo x1:stä x2:een
                        x2[i][m] = x1[i][m];
                    }
                    else
                    {
                        //std::cout << count <<": "<<i <<": " << m <<"-- Previous coordinate is: " << x1[i][m] << std::endl;

                        // Muussa tapauksessa tehdään jotain muuta, mitä on a ja f?
                        x2[i][m] = x1[i][m] + a * f[i][m];
                        //std::cout << a << std::endl;
                        //std::cout << count<< ": " << i <<": " << m <<"-- New coordinate is: " << x2[i][m] << std::endl; 
                    }
                }
            }
            GMX_CATCH_ALL_AND_EXIT_WITH_FATAL_ERROR
        }

        if (s2->flags & enumValueToBitMask(StateEntry::Cgp))
        {
            /* Copy the CG p vector */
            const rvec* p1 = s1->cg_p.rvec_array();
            rvec*       p2 = s2->cg_p.rvec_array();


#pragma omp for schedule(static) nowait
            for (int i = start; i < end; i++)
            {
                // Trivial OpenMP block that does not throw
                copy_rvec(p1[i], p2[i]);
            }
        }

        if (haveDDAtomOrdering(*cr))
        {
            /* OpenMP does not supported unsigned loop variables */
            
#pragma omp for schedule(static) nowait
            for (gmx::index i = 0; i < gmx::ssize(s2->cg_gl); i++)
            {
                s2->cg_gl[i] = s1->cg_gl[i];
            }
        }
    }

    // Copy the DD or pair search counters
    s2->ddp_count       = s1->ddp_count;
    s2->ddp_count_cg_gl = s1->ddp_count_cg_gl;

    if (constr)
    {
        dvdl_constr = 0;
        validStep   = constr->apply(TRUE,
                                  TRUE,
                                  count,
                                  0,
                                  1.0,
                                  s1->x.arrayRefWithPadding(),
                                  s2->x.arrayRefWithPadding(),
                                  ArrayRef<RVec>(),
                                  s2->box,
                                  s2->lambda[FreeEnergyPerturbationCouplingType::Bonded],
                                  &dvdl_constr,
                                  gmx::ArrayRefWithPadding<RVec>(),
                                  false,
                                  nullptr,
                                  gmx::ConstraintVariable::Positions);

        if (cr->nnodes > 1)
        {
            /* This global reduction will affect performance at high
             * parallelization, but we can not really avoid it.
             * But usually EM is not run at high parallelization.
             */
            int reductionBuffer = static_cast<int>(!validStep);
            gmx_sumi(1, &reductionBuffer, cr);
            validStep = (reductionBuffer == 0);
        }

        // We should move this check to the different minimizers
        if (!validStep && ir->eI != IntegrationAlgorithm::Steep)
        {
            gmx_fatal(FARGS,
                      "The coordinates could not be constrained. Minimizer '%s' can not handle "
                      "constraint failures, use minimizer '%s' before using '%s'.",
                      enumValueToString(ir->eI),
                      enumValueToString(IntegrationAlgorithm::Steep),
                      enumValueToString(ir->eI));
        }
    }

    std::cout<< "# do_em_step() returns" << std::endl;

    return validStep;
}








void LegacySimulator::do_steep()
{


    std::cout << "## /src/gromacs/mdrun/minimize.cpp: start of LegacySimulator::do_steep()" << std::endl;

    const char*       SD = "Steepest Descents";
    gmx_global_stat_t gstat;

    // This is 'a'
    real              stepsize;
    
    real              ustep;
    gmx_bool          bDone, bAbort, do_x, do_f;
    tensor            vir, pres;
    rvec              mu_tot = { 0 };
    int               nsteps;
    int               count          = 0;
    int               steps_accepted = 0;
    auto*             mdatoms        = mdAtoms->mdatoms();

    GMX_LOG(mdlog.info)
            .asParagraph()
            .appendText(
                    "Note that activating steepest-descent energy minimization via the "
                    "integrator .mdp option and the command gmx mdrun may "
                    "be available in a different form in a future version of GROMACS, "
                    "e.g. gmx minimize and an .mdp option.");

    /* Create 2 states on the stack and extract pointers that we will swap */
    em_state_t  s0{}, s1{};
    em_state_t* s_min = &s0;
    em_state_t* s_try = &s1;

    // HERE STARTS MY OWN MESS
    std::cout << "\n # em_state_t s_min:" << std::endl;
    std::cout << "epot: "<< s_min->epot << ", fnorm: " << s_min->fnorm << " , fmax: " << s_min->fmax << " , a_fmax: " << s_min->a_fmax <<"\n" <<std::endl;

    std::cout << "\n # em_state_t s_try:" << std::endl;
    std::cout << "epot: "<< s_try->epot << ", fnorm: " << s_try->fnorm << " , fmax: " << s_try->fmax << " , a_fmax: " << s_try->a_fmax << "\n" << std::endl;

    // These are temp
    ForceBuffersView& fbv_min = s_min->f.view();
    ForceBuffersView& fbv_try = s_try->f.view();
    
    ArrayRef<RVec> fbv_min_array = fbv_min.force();

    ArrayRef<RVec> fbv_try_array = fbv_try.force();

    // Alternatively as const:

    const ForceBuffersView& fbv_min_c = s_min->f.view();
    const ForceBuffersView& fbv_try_c = s_try->f.view();

    ArrayRef<const RVec> fbv_min_array_c = fbv_min_c.force();

    ArrayRef<const RVec> fbv_try_array_c = fbv_try_c.force();

    
    std::cout << "# ARRAYREF<RVEC> KOKO: " << fbv_min_array.size() << std::endl;

    for(auto i: fbv_min_array){
        std::cout << i << std::endl;
    }
    
    // How to create BasicVector??
    BasicVector<int> bv = BasicVector<int>(3,5,6);

    std::cout << "# ELEMENTS OF SELFMADE BASIC VECTOR: " << bv[0] << bv[1] << bv[2]<< std::endl;

    BasicVector<int> bv2 = BasicVector<int>(1,2,3);

    BasicVector<int> bv3 = bv + bv2;

    std::cout << "# ELEMENTS OF SELFMADE BASIC VECTOR AFTER ADDITION: " << bv3[0] << bv3[1] << bv3[2]<< std::endl;


    // scaling
    BasicVector<int> bv4 = bv*3;


    std::cout << "# ELEMENTS OF SELFMADE BASIC VECTOR AFTER SCALING: " << bv4[0] << bv4[1] << bv4[2]<< std::endl;


    int dotted = bv.dot(bv2);

    BasicVector<int> bv5 = bv.cross(bv2);


    std::cout << "# DOT PRODUCT RESULT: " << dotted << std::endl;


    std::cout << "# ELEMENTS OF SELFMADE BASIC VECTOR AFTER CROSS PRODUCT: " << bv5[0] << bv5[1] << bv5[2]<< std::endl;


    // HERE ENDS MY OWN MESS



    ObservablesReducer observablesReducer = observablesReducerBuilder->build();

    /* Init em and store the local state in s_try */
    std::cout << "## Calling init_em()" << std::endl; 
    init_em(fplog,
            mdlog,
            SD,
            cr,
            inputrec,
            imdSession,
            pull_work,
            state_global,
            top_global,
            s_try,
            top,
            nrnb,
            fr,
            mdAtoms,
            &gstat,
            vsite,
            constr,
            nullptr);

    std::cout << "## init_em() returned" << std::endl;


    std::cout << "\n # em_state_t s_min:" << std::endl;
    std::cout << "epot: "<< s_min->epot << ", fnorm: " << s_min->fnorm << " , fmax: " << s_min->fmax << " , a_fmax: " << s_min->a_fmax << std::endl;

    std::cout << "\n # em_state_t s_try:" << std::endl;
    std::cout << "epot: "<< s_try->epot << ", fnorm: " << s_try->fnorm << " , fmax: " << s_try->fmax << " , a_fmax: " << s_try->a_fmax << std::endl;


    const bool        simulationsShareState = false;
    gmx_mdoutf*       outf                  = init_mdoutf(fplog,
                                   nfile,
                                   fnm,
                                   mdrunOptions,
                                   cr,
                                   outputProvider,
                                   mdModulesNotifiers,
                                   inputrec,
                                   top_global,
                                   nullptr,
                                   wcycle,
                                   StartingBehavior::NewSimulation,
                                   simulationsShareState,
                                   ms);
    gmx::EnergyOutput energyOutput(mdoutf_get_fp_ene(outf),
                                   top_global,
                                   *inputrec,
                                   pull_work,
                                   nullptr,
                                   false,
                                   StartingBehavior::NewSimulation,
                                   simulationsShareState,
                                   mdModulesNotifiers);

    /* Print to log file  */
    print_em_start(fplog, cr, walltime_accounting, wcycle, SD);

    /* Set variables for stepsize (in nm). This is the largest
     * step that we are going to make in any direction.
     */
    ustep    = inputrec->em_stepsize;
    stepsize = 0;

    /* Max number of steps  */
    nsteps = inputrec->nsteps;

    if (MAIN(cr))
    {
        /* Print to the screen  */
        sp_header(stderr, SD, inputrec->em_tol, nsteps);
    }
    if (fplog)
    {
        sp_header(fplog, SD, inputrec->em_tol, nsteps);
    }
    EnergyEvaluator energyEvaluator{ fplog,
                                     mdlog,
                                     cr,
                                     ms,
                                     top_global,
                                     top,
                                     inputrec,
                                     imdSession,
                                     pull_work,
                                     enforcedRotation,
                                     nrnb,
                                     wcycle,
                                     gstat,
                                     &observablesReducer,
                                     vsite,
                                     constr,
                                     mdAtoms,
                                     fr,
                                     runScheduleWork,
                                     enerd,
                                     -1,
                                     {} };

    /**** HERE STARTS THE LOOP ****
     * count is the counter for the number of steps
     * bDone will be TRUE when the minimization has converged
     * bAbort will be TRUE when nsteps steps have been performed or when
     * the stepsize becomes smaller than is reasonable for machine precision
     */


    std::cout << " ## Entering minimization loop" << std::endl;

    count  = 0;
    bDone  = FALSE;
    bAbort = FALSE;
    while (!bDone && !bAbort)
    {

        std::cout << " ## \n\nMinimization loop iteration: " << count << std::endl;



        bAbort = (nsteps >= 0) && (count == nsteps);

        /* set new coordinates, except for first step */
        bool validStep = true;



        // Mikä on s_min ja s_try tila tässä vaiheessa??


        if (count > 0)
        {   
            std::cout << " ## Setting new coordinates" << std::endl;
            validStep = do_em_step(
                    cr, inputrec, mdatoms, s_min, stepsize, s_min->f.view().forceWithPadding(), s_try, constr, count);
        }



        if(count > 1){
            std::cout << "PEEK TO s_try\n" << std::endl;
            ForceBuffersView& fbv_try2 = s_try->f.view();

            ArrayRef<RVec> fbv_try_array2 = fbv_try2.force();

            std::cout << " # ARRAYREF<RVEC> koko on: " << fbv_try_array2.size()<< std::endl;
            std::cout << " FIRST THREE VECTORS:\n" <<std::endl;
            std::cout << " ## " << fbv_try_array2[0][0] << ", " << fbv_try_array2[0][1] << ", " << fbv_try_array2[0][2] << std::endl;
            std::cout << " ## " << fbv_try_array2[1][0] << ", " << fbv_try_array2[1][1] << ", " << fbv_try_array2[1][2] << std::endl;
            std::cout << " ## " << fbv_try_array2[2][0] << ", " << fbv_try_array2[2][1] << ", " << fbv_try_array2[2][2] << "\n\n" << std::endl;
        }





        if (validStep)
        {
            std::cout << " ## Evaluating energy" << std::endl;
            energyEvaluator.run(s_try, mu_tot, vir, pres, count, count == 0, count);
        }
        else
        {
            // Signal constraint error during stepping with energy=inf
            s_try->epot = std::numeric_limits<real>::infinity();
        }

        if (MAIN(cr))
        {
            EnergyOutput::printHeader(fplog, count, count);
        }

        if (count == 0)
        {
            s_min->epot = s_try->epot;
        }

        /* Print it if necessary  */
        if (MAIN(cr))
        {
            if (mdrunOptions.verbose)
            {
                fprintf(stderr,
                        "Step=%5d, Dmax= %6.1e nm, Epot= %12.5e Fmax= %11.5e, atom= %d%c",
                        count,
                        ustep,
                        s_try->epot,
                        s_try->fmax,
                        s_try->a_fmax + 1,
                        ((count == 0) || (s_try->epot < s_min->epot)) ? '\n' : '\r');
                fflush(stderr);
            }

            if ((count == 0) || (s_try->epot < s_min->epot))
            {
                /* Store the new (lower) energies  */
                matrix nullBox = {};
                energyOutput.addDataAtEnergyStep(false,
                                                 false,
                                                 static_cast<double>(count),
                                                 mdatoms->tmass,
                                                 enerd,
                                                 nullptr,
                                                 nullBox,
                                                 PTCouplingArrays(),
                                                 0,
                                                 vir,
                                                 pres,
                                                 nullptr,
                                                 mu_tot,
                                                 constr);

                imdSession->fillEnergyRecord(count, TRUE);

                const bool do_dr = do_per_step(steps_accepted, inputrec->nstdisreout);
                const bool do_or = do_per_step(steps_accepted, inputrec->nstorireout);
                energyOutput.printStepToEnergyFile(
                        mdoutf_get_fp_ene(outf), TRUE, do_dr, do_or, fplog, count, count, fr->fcdata.get(), nullptr);
                fflush(fplog);
            }
        }

        /* Now if the new energy is smaller than the previous...
         * or if this is the first step!
         * or if we did random steps!
         */

        if ((count == 0) || (s_try->epot < s_min->epot))
        {
            steps_accepted++;

            /* Test whether the convergence criterion is met...  */
            bDone = (s_try->fmax < inputrec->em_tol);

            /* Copy the arrays for force, positions and energy  */
            /* The 'Min' array always holds the coords and forces of the minimal
               sampled energy  */
            swap_em_state(&s_min, &s_try);
            if (count > 0)
            {
                ustep *= 1.2;
            }

            /* Write to trn, if necessary */
            do_x = do_per_step(steps_accepted, inputrec->nstxout);
            do_f = do_per_step(steps_accepted, inputrec->nstfout);
            write_em_traj(
                    fplog, cr, outf, do_x, do_f, nullptr, top_global, inputrec, count, s_min, state_global, observablesHistory);
        }
        else
        {
            /* If energy is not smaller make the step smaller...  */
            ustep *= 0.5;

            if (haveDDAtomOrdering(*cr) && s_min->s.ddp_count != cr->dd->ddp_count)
            {
                /* Reload the old state */
                em_dd_partition_system(fplog,
                                       mdlog,
                                       count,
                                       cr,
                                       top_global,
                                       inputrec,
                                       imdSession,
                                       pull_work,
                                       s_min,
                                       top,
                                       mdAtoms,
                                       fr,
                                       vsite,
                                       constr,
                                       nrnb,
                                       wcycle);
            }
        }

        // If the force is very small after finishing minimization,
        // we risk dividing by zero when calculating the step size.
        // So we check first if the minimization has stopped before
        // trying to obtain a new step size.
        if (!bDone)
        {
            /* Determine new step  */
            std::cout << "\n# ustep is: " << ustep << std::endl;
            std::cout << "# s_min->fmax is: " << s_min->fmax << std::endl;
            stepsize = ustep / s_min->fmax;
        }

        /* Check if stepsize is too small, with 1 nm as a characteristic length */
#if GMX_DOUBLE
        if (count == nsteps || ustep < 1e-12)
#else
        if (count == nsteps || ustep < 1e-6)
#endif
        {
            if (MAIN(cr))
            {
                warn_step(fplog, inputrec->em_tol, s_min->fmax, count == nsteps, constr != nullptr);
            }
            bAbort = TRUE;
        }

        /* Send IMD energies and positions, if bIMD is TRUE. */
        if (imdSession->run(count,
                            TRUE,
                            MAIN(cr) ? state_global->box : nullptr,
                            MAIN(cr) ? state_global->x : gmx::ArrayRef<gmx::RVec>(),
                            0)
            && MAIN(cr))
        {
            imdSession->sendPositionsAndEnergies();
        }

        count++;
        observablesReducer.markAsReadyToReduce();
    } /* End of the loop  */


    // LOOP ENDS HERE


    /* Print some data...  */
    if (MAIN(cr))
    {
        fprintf(stderr, "\nwriting lowest energy coordinates.\n");
    }
    write_em_traj(fplog,
                  cr,
                  outf,
                  TRUE,
                  inputrec->nstfout != 0,
                  ftp2fn(efSTO, nfile, fnm),
                  top_global,
                  inputrec,
                  count,
                  s_min,
                  state_global,
                  observablesHistory);

    if (MAIN(cr))
    {
        double sqrtNumAtoms = sqrt(static_cast<double>(state_global->natoms));

        print_converged(stderr, SD, inputrec->em_tol, count, bDone, nsteps, s_min, sqrtNumAtoms);
        print_converged(fplog, SD, inputrec->em_tol, count, bDone, nsteps, s_min, sqrtNumAtoms);
    }

    finish_em(cr, outf, walltime_accounting, wcycle);

    walltime_accounting_set_nsteps_done(walltime_accounting, count);


    std::cout << "## /src/gromacs/mdrun/minimize.cpp: end of LegacySimulator::do_steep()" << std::endl;

}



} // namespace gmx
