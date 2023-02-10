from baselines.common import explained_variance, zipsame, dataset
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time
from baselines.common import colorize
from collections import deque
from baselines.common import set_global_seeds
from baselines.common.mpi_adam import MpiAdam
from baselines.common.cg import cg
from baselines.common.input import observation_placeholder
from baselines.common.policies import build_policy
from contextlib import contextmanager
import os

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

# The following five functions are specific for LOKI and not original for baselines.

def batch_value_prediction(pi, obz):
    """
    Calculate value predictions for a batch of observations using a trained Tensorflow model.
    
    Parameters
    ----------
    pi : tf.Session
        Tensorflow session with a trained model.
    obz : numpy array
        Array of observations to make value predictions for.
    
    Returns
    -------
    vpreds : numpy array
        Array of value predictions for each observation in obz.
    """
    # Determine number of samples and initialize array to hold value predictions
    n_samples = len(obz)
    vpreds = np.zeros(n_samples, 'float32')
    
    # Set batch size and create dataset from input array
    batch_size = 2048
    dataset = tf.data.Dataset.from_tensor_slices(obz)
    dataset = dataset.batch(batch_size)
    
    # Get iterator and placeholder for observations
    iterator = dataset.make_initializable_iterator()
    ob_ph = iterator.get_next()
    
    # Initialize iterator and make value predictions for each batch
    pi.run(iterator.initializer)
    while True:
        try:
            vpreds_batch = pi.vpred.eval(feed_dict={ob_ph: ob_ph.eval()})
            vpreds = np.concatenate((vpreds, vpreds_batch))
        except tf.errors.OutOfRangeError:
            break
    
    return vpreds

def create_saver(pi, for_expert=False):
    """
    Create a saver for the policy network. 
    If for_expert is set to True, the saver will be able to save variables for expert policy. 
    This is done by creating a dictionary with keys as the names of variables in the expert policy, 
    and values as the variables in the pi policy. 
    
    Args:
        pi (tf.keras.Model): The policy network.
        for_expert (bool, optional): Whether to create a saver for expert policy. Defaults to False.
        
    Returns:
        tf.train.Saver: The saver for the policy network.
        
    Example:
        >>> pi = create_policy_network()
        >>> saver = create_saver(pi)
        >>> saver
        <tensorflow.python.training.saver.Saver object at 0x7f8f9cf938d0>
    """
    # Create saver, need to include some non-trainable variables, such as running mean std stuff.
    pi_mlp_var_list = pi.get_variables()
    if for_expert:
        # Since saved variables have names started with pi/, we work around it by creating a dictionary something like:
        # {"pi/pol/kernel": expert policy var, ...}
        pi_mlp_var_dict = {}
        for var in pi_mlp_var_list:
            splits = var.name.split("/")
            splits[0] = "pi"
            saved_var_name = "/".join(splits)
            # It's weird that the variables saved in the checkpoint do not have ":0", but var.name has,
            # although this ":0" works for pi, but not expert. We need to manually remove it.
            assert saved_var_name.split(":")[1] == "0"  # make sure it's :0
            saved_var_name = saved_var_name.split(":")[0]
            pi_mlp_var_dict[saved_var_name] = var
        pi_mlp_var_list = pi_mlp_var_dict  # a little abuse of list
    saver = tf.train.Saver(pi_mlp_var_list, max_to_keep=50)
    print(f'Create saver for expert: {for_expert}')
    return saver

# This function might be changed for a constant
def compute_max_kl(ilrate):
    """
    Compute the maximum KL divergence for the given ilrate.

    Parameters:
    ----------
    ilrate : float
        Interpolation rate between imitation learning and reinforcement learning. 
        ilrate = 1.0 means using the KL divergence for imitation learning and 
        ilrate = 0.0 means using the KL divergence for reinforcement learning.

    Returns:
    -------
    max_kl : float
        The maximum KL divergence for the given ilrate.

    Example:
    --------
    >>> compute_max_kl(1.0)
    0.1
    >>> compute_max_kl(0.0)
    0.01
    >>> compute_max_kl(0.5)
    ValueError: Unknow max_kl, ilrate: 0.5
    """
    # Hard code the KL constraints for IL and RL.
    RL_MAX_KL = 0.01
    IL_MAX_KL = 0.1
    if np.isclose(ilrate, 1.0):
        max_kl = IL_MAX_KL
    elif np.isclose(ilrate, 0.0):
        max_kl = RL_MAX_KL
    else:
        raise ValueError(f'Unknow max_kl, ilrate: {ilrate}')
    return max_kl

def compute_ilrate(ilrate_multi, iters, ilrate_decay=None, hard_switch_iter=None):
    """
    Compute the ilrate (imitation learning rate) based on the specified parameters.
    
    Parameters
    ----------
    ilrate_multi : float
        The initial value for the ilrate.
    iters : int
        The current number of iterations.
    ilrate_decay : float, optional
        The decay rate for the ilrate. Default is None.
    hard_switch_iter : int, optional
        The iteration number at which the ilrate switches from 1.0 to 0.0. Default is None.
        
    Returns
    -------
    float
        The computed ilrate.
        
    Examples
    --------
    >>> compute_ilrate(1.0, 5)
    1.0
    >>> compute_ilrate(1.0, 5, ilrate_decay=0.9)
    0.531441
    >>> compute_ilrate(1.0, 5, hard_switch_iter=3)
    0.0
    """
    if hard_switch_iter is not None:
        # Hard switch
        if iters < hard_switch_iter:
            ilrate = 1.0
        else:
            ilrate = 0.0
    elif ilrate_decay is not None:
        # Mixing
        ilrate = ilrate_multi * ilrate_decay ** iters  # exponential
    else:
        ilrate = ilrate_multi
    return ilrate

def add_expert_v(expert, seg):
    """
    Add the expert's value predictions to the given segment of data.

    Parameters
    ----------
    expert: object
        The expert policy object with a `act` method to make value predictions.
    seg: dict
        A segment of data containing observations and actions, in the format of 
        {"ob": observation, "ac": action, "nextob": next_observation}.

    Returns
    -------
    None
    """
    # Compute value predictions for the expert's actions taken in each step of the segment
    expert_v = batch_value_prediction(expert, seg["ob"])
    # Append the value prediction for the expert's action taken in the next step
    expert_v = np.append(expert_v, expert.act(True, seg["nextob"])[1])
    # Add the expert's value predictions to the segment
    seg["expert_v"] = expert_v

def traj_segment_generator(pi, env, horizon, stochastic=True, expert=False):
    # Initialize state variables
    t = 0
    ac = env.action_space.sample()
    new = True
    rew = 0.0
    ob = env.reset()

    cur_ep_ret = 0
    cur_ep_len = 0
    ep_rets = []
    ep_lens = []

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    # Indicates whether this sample is the first of a new episode / rollout.
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

    while True:
        prevac = ac
        if expert:
            ac, vpred, _, _ = pi.act(stochastic, ob)
        else:
            ac, vpred, _, _ = pi.act(stochastic, ob)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            # In LOKI the nextob and nextnew are added here
            yield {"ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news,
                    "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens, "nextob": ob, "nextnew": new}
            if expert:
                _ , vpred, _, _ = pi.act(stochastic, ob)
            else:
                _, vpred, _, _ = pi.act(stochastic, ob)
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac

        ob, rew, new, _ = env.step(ac)
        rews[i] = rew

        cur_ep_ret += rew
        cur_ep_len += 1
        if new:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()
        t += 1

def add_vtarg_and_adv(seg, gamma, lam):
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]


def cfc_step(observation, pi, vf_pi, stochastic=True):
    # get the action from the CfC model
    action_probs = pi.predict(observation)[0]
    if stochastic:
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
    else:
        action = np.argmax(action_probs)
    # get the vpred from the value function model
    vpred = vf_pi.predict(np.expand_dims(action_probs, axis=0))[0][0]
    return action, vpred


def learn(policy_fn,
        env,
        timesteps_per_batch, # what to train on
        cg_iters,
        gamma,
        lam, # advantage estimation
        total_timesteps=0,
        seed=None,
        ent_coef=0.0,
        cg_damping=1e-2,
        vf_stepsize=3e-4,
        vf_iters=3,
        max_episodes=0, max_iters=0,  # time constraint
        mode = True,
        policy_save_freq=50, 
        expert_dir=None, 
        ilrate_multi=0.0, 
        ilrate_decay=1.0, 
        hard_switch_iter=None, 
        save_no_policy=False, 
        pretrain_dir=None,  
        il_gae=False,
        callback=None,
        deterministic_expert=False,
        initialize_with_expert_logstd=False
        ):
    '''
    learn a policy function with TRPO algorithm. There are some changes in this function to accomodate the LOKI algorithm.
    The parameters network and network_kwargs have been removed. The following parameters have been added:
    policy_fn, policy_save_freq, expert_dir, ilrate_multi, ilrate_decay, hard_switch_iter, trunctated_horizon, save_no_policy,
    pretrain_dir, il_gae. initlialize_with_expert_logstd. 

    Parameters:
    ----------

    policy_fn               policy function

    env                     environment (one of the gym environments or wrapped via baselines.common.vec_env.VecEnv-type class

    timesteps_per_batch     timesteps per gradient estimation batch

    ent_coef                coefficient of policy entropy term in the optimization objective

    cg_damping              conjugate gradient damping

    cg_iters                number of conjugate gradient iterations per optimization step

    vf_stepsize             learning rate for adam optimizer used to optimie value function loss

    vf_iters                number of iterations of value function optimization iterations per each policy optimization step

    total_timesteps         max number of timesteps

    max_episodes            max number of episodes

    max_iters               maximum number of policy optimization iterations

    policy_save_freq        frequency of saving the policy

    expert_dir              directory where the expert policies are stored

    ilrate_multi            multiplier for the inverse reinforcement learning rate

    ilrate_decay            decay rate for the inverse reinforcement learning rate

    hard_switch_iter        iteration at which the inverse reinforcement learning rate is set to zero

    save_no_policy          boolean indicating whether to save the model without the policy

    pretrain_dir            directory where the pre-trained policy is stored

    il_gae                  boolean indicating whether to use generalized advantage estimation for inverse reinforcement learning
    
    callback                function to be called with (locals(), globals()) each policy optimization step

    deterministic_expert    boolean indicating whether the expert policy is deterministic

    initialize_with_expert_logstd    boolean indicating whether to initialize the policy with the logstd of the expert policy

    Returns:
    -------

    learnt model

    '''

    if MPI is not None:
        nworkers = MPI.COMM_WORLD.Get_size()
        rank = MPI.COMM_WORLD.Get_rank()
    else:
        nworkers = 1
        rank = 0

    cpus_per_worker = 1
    U.get_session(config=tf.ConfigProto(
            allow_soft_placement=True,
            inter_op_parallelism_threads=cpus_per_worker,
            intra_op_parallelism_threads=cpus_per_worker
    ))

    grad_batch_size = 64
    # pretrain_rollouts_file = 'pretrain_rollouts.npz'  # Has to check what this does

    set_global_seeds(seed)

    np.set_printoptions(precision=3)
    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space

    ob = observation_placeholder(ob_space)
    with tf.variable_scope("pi"):
        pi = policy_fn("pi", ob_space, ac_space) # Construct network for new policy, unfinished in LOKI, might be changes
    with tf.variable_scope("oldpi"):
        oldpi = policy_fn("oldpi", ob_space, ac_space) # Network for old policy
    

    # Robin: If we're going to import an expert then uncomment this
    #if expert_dir:
    #    assert os.path.isdir(expert_dir)
    #expert = None if not expert_dir else policy_fn("expert", ob_space, ac_space)
    #if expert_dir:
    #    expert_saver = create_saver(expert, for_expert=True)
    #    expert_path = os.path.join(expert_dir, 'expert.ckpt')
    #if pretrain_dir:
    #    pretrain_path = os.path.join(pretrain_dir, 'pretrained.ckpt')

    atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return

    # created in MlpPolicy (Robin: Which we have to change) init, 2d tensor 
    ob = U.get_placeholder_cached(name="ob")

    ac = pi.pdtype.sample_placeholder([None])
    ac_scale = tf.placeholder(dtype=tf.float32, shape=[ac.shape[1]])
    ilrew_ph = tf.placeholder(dtype=tf.float32, shape=[None])  # imitation learning reward, maybe needed for ilgain
    ilrate = tf.placeholder(dtype=tf.float32, shape=[])  # a float scalar
    phs = [ob, ac, atarg, ilrew_ph, ilrate, ac_scale]  # placeholders
    assign_updates = [tf.assign(oldv, newv) for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())]
    assign_old_eq_new = U.function([], [], updates=assign_updates)
    # Make sure trainable policy parameters are only from pi.
    all_var_list = pi.get_trainable_variables()

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = tf.reduce_mean(kloldnew)
    meanent = tf.reduce_mean(ent)
    entbonus = ent_coef * meanent

    # Build up. 
    # Value function objective.
    vferr = tf.reduce_mean(tf.square(pi.vpred - ret))

    # Policy gradient objective. Lots of changes from baselines to accomodate for LOKI
    ent = pi.pd.entropy()
    meanent = tf.reduce_mean(ent)
    entbonus = ent_coef * meanent
    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac))  # advantage * pnew / pold
    rlgain = tf.reduce_mean(ratio * atarg)
    ilgain = tf.constant(0.0) # Chaned from original LOKI code as it was not made for our purpose
    surrgain = ilgain * ilrate + (1 - ilrate) * rlgain
    optimgain = surrgain + entbonus
    kloldnew = oldpi.pd.kl(pi.pd)
    meankl = tf.reduce_mean(kloldnew)
    losses = [optimgain, meankl, entbonus, ilgain, rlgain, surrgain, meanent]
    loss_names = ["optimgain", "meankl", "entloss", "ilgain", "rlgain", "surrgain", "entropy"]
    compute_losses = U.function(phs, losses)
    compute_lossandgrad = U.function(phs, losses + [U.flatgrad(optimgain, var_list)])

    dist = meankl

    # all_var_list = get_trainable_variables("pi")
    # var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("pol")]
    # vf_var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("vf")]
    var_list = get_pi_trainable_variables("pi")
    vf_var_list = get_vf_trainable_variables("pi")

    # value function is using adam, and policy is not.
    vfadam = MpiAdam(vf_var_list)

    get_flat = U.GetFlat(var_list)
    set_from_flat = U.SetFromFlat(var_list)
    klgrads = tf.gradients(dist, var_list)
    flat_tangent = tf.placeholder(dtype=tf.float32, shape=[None], name="flat_tan")
    shapes = [var.get_shape().as_list() for var in var_list]
    start = 0
    tangents = []
    for shape in shapes:
        sz = U.intprod(shape)
        tangents.append(tf.reshape(flat_tangent[start:start+sz], shape))
        start += sz
    gvp = tf.add_n([tf.reduce_sum(g*tangent) for (g, tangent) in zipsame(klgrads, tangents)]) #pylint: disable=E1111
    fvp = U.flatgrad(gvp, var_list)

    assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
        for (oldv, newv) in zipsame(get_variables("oldpi"), get_variables("pi"))])

    compute_losses = U.function([ob, ac, atarg], losses)
    compute_lossandgrad = U.function([ob, ac, atarg], losses + [U.flatgrad(optimgain, var_list)])
    compute_fvp = U.function([flat_tangent] + [ob, ac], fvp)  # not sure why atarg and ilrew are needed
    # in spite of the name, only grad is computed. This is changed from baselines.
    # ret will be fed by tdlamret, which is adv + vpred, and
    # vferr: tf.reduce_mean(tf.square(pi.vpred - ret))
    # It seems that he is not estimate V^{\pi, \gamma} (Eq.4 in trpo-gae paper), instead some weird stuff:
    # A^{GAE} in Eq.25 + \hat V_t
    compute_vflossandgrad = U.function([ob, ret], U.flatgrad(vferr, vf_var_list))

    @contextmanager
    def timed(msg):
        if rank == 0:
            print(colorize(msg, color='magenta'))
            tstart = time.time()
            yield
            print(colorize("done in %.3f seconds"%(time.time() - tstart), color='magenta'))
        else:
            yield

    def allmean(x):
        assert isinstance(x, np.ndarray)
        if MPI is not None:
            out = np.empty_like(x)
            MPI.COMM_WORLD.Allreduce(x, out, op=MPI.SUM)
            out /= nworkers
        else:
            out = np.copy(x)

        return out

    U.initialize()
    
    # This was removed in LOKI
    #if load_path is not None:
    #    pi.load(load_path)

    # This is two imports are from LOKI
    if expert_dir:
        print('Restoring expert')
        expert_saver = create_saver("expert", for_expert=True)
        expert_saver.restore(tf.get_default_session(), expert_dir)
        with tf.variable_scope("expert", reuse=True):
            with tf.variable_scope("pol", reuse=True):
                logstd = tf.get_variable(name="logstd")
                expert_logstd = logstd.eval()
        with tf.variable_scope("pi", reuse=True):
            with tf.variable_scope("pol", reuse=True):
                logstd = tf.get_variable(name="logstd")
                print('Expert logstd: {expert_logstd}')

    if pretrain_dir:
        print('Initialize learner with pretrained model')
        policy_saver = create_saver("pi", for_expert=False)
        policy_saver.restore(tf.get_default_session(), pretrain_dir)
        assign_old_eq_new()
        with tf.variable_scope("pi", reuse=True):
            with tf.variable_scope("pol", reuse=True):
                logstd = tf.get_variable(name="logstd")
                print(f'Pretrain policy logstd: {logstd.eval()}')

    th_init = get_flat()
    if MPI is not None:
        MPI.COMM_WORLD.Bcast(th_init, root=0)

    set_from_flat(th_init)
    vfadam.sync()
    print("Init param sum", th_init.sum(), flush=True)

    # Prepare for rollouts
    # ----------------------------------------
    # This is from LOKI
    pretrain_il = False # set true id you want to pretrain with IL
    if pretrain_il: # Script which takes keyboard input to play atari
        assert expert # Robin: Expert is not made yet as it's our input
        seg_gen = traj_segment_generator(expert, env=env, horizon=timesteps_per_batch,
                                         stochastic=not deterministic_expert, expert=True)
    else:
        seg_gen = traj_segment_generator(pi, env, timesteps_per_batch, stochastic=True)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=40) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=40) # rolling buffer for episode rewards

    if sum([max_iters>0, total_timesteps>0, max_episodes>0])==0: 
        # noththing to be done
        return pi

    assert sum([max_iters>0, total_timesteps>0, max_episodes>0]) < 2, \
        'out of max_iters, total_timesteps, and max_episodes only one should be specified'

    while True:
        if callback: callback(locals(), globals())
        if total_timesteps and timesteps_so_far >= total_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        logger.log(f"********** Iteration {iters_so_far} ************")

        with timed("sampling"):
            if pretrain_il:
                # assert rank == 0  # only one worker! Not sure why now....
                if iters_so_far == 0:
                    seg = seg_gen.__next__()
                    ac_scale_val = np.std(seg["ac"], axis=0) if rank == 0 else np.ones(seg["ac"][0].shape)
                else:
                    # need to update vpred, nextvpred in seg!
                    seg['vpred'] = batch_value_prediction(pi, seg["ob"])
                    _, seg['nextvpred'] = cfc_step(True, seg['nextob'])
            else:
                seg = seg_gen.__next__()
                if iters_so_far == 0:
                    ac_scale_val = np.ones(seg["ac"][0].shape)
                    if expert_dir:
                        print('Generating expert rollouts for action scaling')
                        if rank == 0:
                            expert_seg_gen = traj_segment_generator(expert, env, 4 * timesteps_per_batch, stochastic=True)
                            expert_seg = next(expert_seg_gen)
                            ac_scale_val = np.std(expert_seg["ac"], axis=0)
        print(f"ac_scale_val: {ac_scale_val}")

        add_vtarg_and_adv(seg, gamma, lam)

        # From LOKI
        seg["ilrew"] = seg["rew"]

        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        ob_val, ac, atarg, tdlamret, ilrew = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"], seg["ilrew"] # Changes from LOKI
        
        # From LOKI. This is very the switch from IL to RL happens by changing the kl weight
        ilrate_val = compute_ilrate(ilrate_multi, iters_so_far, ilrate_decay, hard_switch_iter)
        max_kl = compute_max_kl(ilrate_val) 
        print(f"Max KL: {max_kl}")
        
        vpredbefore = seg["vpred"] # predicted value function before udpate
        atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate

        # From LOKI
        if not np.allclose(ilrew.std(), 0.0):
            ilrew = (ilrew - ilrew.mean()) / ilrew.std()
        else:
            ilrew = (ilrew - ilrew.mean())

        if hasattr(pi, "ret_rms"): pi.ret_rms.update(tdlamret)
        if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob) # update running mean/std for policy

        lossargs = [ob_val, ac] # from LOKI
        fvpargs = [arr[::5] for arr in lossargs]
        lossargs += [atarg, ilrew, ilrate_val, ac_scale_val]

        def fisher_vector_product(p):
            return allmean(compute_fvp(p, *fvpargs)) + cg_damping * p

        assign_old_eq_new() # set old parameter values to new parameter values
        with timed("computegrad"):
            *lossbefore, g = compute_lossandgrad(*lossargs)
        lossbefore = allmean(np.array(lossbefore))
        g = allmean(g)
        if np.allclose(g, 0):
            logger.log("Got zero gradient. not updating")
        else:
            with timed("cg"):
                # g: gradient over processes,
                # fisher_vector_product: after compute kl stuff using samples from local process, uses mpi
                # aggregate them. Although some computation can be saved in cg.
                stepdir = cg(fisher_vector_product, g, cg_iters=cg_iters, verbose=rank==0)
            assert np.isfinite(stepdir).all()
            shs = .5*stepdir.dot(fisher_vector_product(stepdir))
            lm = np.sqrt(shs / max_kl)
            # logger.log("lagrange multiplier:", lm, "gnorm:", np.linalg.norm(g))
            fullstep = stepdir / lm
            expectedimprove = g.dot(fullstep)
            surrbefore = lossbefore[0]
            stepsize = 1.0
            thbefore = get_flat()
            for _ in range(10):
                thnew = thbefore + fullstep * stepsize
                set_from_flat(thnew)
                meanlosses = surr, kl, *_ = allmean(np.array(compute_losses(*lossargs))) # From LOKI change *args to *lossargs
                improve = surr - surrbefore
                logger.log("Expected: %.3f Actual: %.3f"%(expectedimprove, improve))
                if not np.isfinite(meanlosses).all():
                    logger.log("Got non-finite value of losses -- bad!")
                elif kl > max_kl * 1.5:
                    logger.log("violated KL constraint. shrinking step.")
                elif improve < 0:
                    logger.log("surrogate didn't improve. shrinking step.")
                else:
                    logger.log("Stepsize OK!")
                    break
                stepsize *= .5
            else:
                logger.log("couldn't compute a good step")
                set_from_flat(thbefore)
            if nworkers > 1 and iters_so_far % 20 == 0:
                paramsums = MPI.COMM_WORLD.allgather((thnew.sum(), vfadam.getflat().sum())) # list of tuples
                assert all(np.allclose(ps, paramsums[0]) for ps in paramsums[1:])

        for (lossname, lossval) in zip(loss_names, meanlosses):
            logger.record_tabular(lossname, lossval)

        with timed("vf"):

            for _ in range(vf_iters):
                for (mbob, mbret) in dataset.iterbatches((seg["ob"], seg["tdlamret"]),
                include_final_partial_batch=False, batch_size=grad_batch_size):
                    g = allmean(compute_vflossandgrad(mbob, mbret))
                    vfadam.update(g, vf_stepsize)

        logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))

        lrlocal = (seg["ep_lens"], seg["ep_rets"]) # local values
        if MPI is not None:
            listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
        else:
            listoflrpairs = [lrlocal]

        lens, rews = map(flatten_lists, zip(*listoflrpairs))
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)

        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        logger.record_tabular("EpThisIter", len(lens))
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1

        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)
        logger.record_tabular("ilrate", ilrate_val)

        if rank==0:
            logger.dump_tabular()
            # Robin: This must be fixed or removed
            '''
            if iters_so_far % policy_save_freq == 0 and not save_no_policy:
                prefix = '{}_{}'.format(policy_files_prefix, iters_so_far)
                save_path = policy_saver.save(tf.get_default_session(), os.path.join(policy_dir, prefix + '.ckpt'))
                print(f'Intemediate policy has been saved to {save_path}')
            print('Print logstd:')
            with tf.variable_scope("pi", reuse=True):
                with tf.variable_scope("pol", reuse=True):
                    logstd = tf.get_variable(name="logstd")
                    print(logstd.eval())
            '''
    return pi, atarg

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]

def get_variables(scope):
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope)

def get_trainable_variables(scope):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)

def get_vf_trainable_variables(scope):
    return [v for v in get_trainable_variables(scope) if 'vf' in v.name[len(scope):].split('/')]

def get_pi_trainable_variables(scope):
    return [v for v in get_trainable_variables(scope) if 'pi' in v.name[len(scope):].split('/')]

