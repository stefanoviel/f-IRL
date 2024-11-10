'''
f-IRL: Extract policy/reward from specified expert samples
'''
import sys, os, time
import numpy as np
import torch
import gymnasium as gym 
from ruamel.yaml import YAML

from firl.divs.f_div_disc import f_div_disc_loss
from firl.divs.f_div import maxentirl_loss
from firl.divs.ipm import ipm_loss
from firl.models.reward import MLPReward
from firl.models.discrim import SMMIRLDisc as Disc
from firl.models.discrim import SMMIRLCritic as Critic
# from common.sac import ReplayBuffer, SAC
from common.mod_sac import ReplayBuffer, SAC

import envs
from utils import system, collect, logger, eval
from utils.plots.train_plot_high_dim import plot_disc
from utils.plots.train_plot import plot_disc as visual_disc

import datetime
import dateutil.tz
import json, copy
from torch.utils.tensorboard import SummaryWriter

def try_evaluate(itr: int, policy_type: str, sac_info, writer, global_step):
    assert policy_type in ["Running"]
    update_time = itr * v['reward']['gradient_step']
    env_steps = itr * v['sac']['epochs'] * v['env']['T']
    agent_emp_states = samples[0].copy()
    assert agent_emp_states.shape[0] == v['irl']['training_trajs']

    metrics = eval.KL_summary(expert_samples, agent_emp_states.reshape(-1, agent_emp_states.shape[2]), 
                         env_steps, policy_type)
    # eval real reward
    real_return_det = eval.evaluate_real_return(sac_agent.get_action, env_fn(), 
                                            v['irl']['eval_episodes'], v['env']['T'], True)
    real_return_sto = eval.evaluate_real_return(sac_agent.get_action, env_fn(), 
                                            v['irl']['eval_episodes'], v['env']['T'], False)
    
    # Log to tensorboard
    writer.add_scalar('Returns/Deterministic', real_return_det, global_step)
    writer.add_scalar('Returns/Stochastic', real_return_sto, global_step)
    
    # Log KL metrics
    for key, value in metrics.items():
        writer.add_scalar(f'Metrics/{key}', value, global_step)

    if v['obj'] in ["emd"]:
        eval_len = int(0.1 * len(critic_loss["main"]))
        emd = -np.array(critic_loss["main"][-eval_len:]).mean()
        metrics['emd'] = emd
        logger.record_tabular(f"{policy_type} EMD", emd)
    
    # plot_disc(v['obj'], log_folder, env_steps, 
    #     sac_info, critic_loss if v['obj'] in ["emd"] else disc_loss, metrics)
    if "PointMaze" in env_name:
        visual_disc(agent_emp_states, reward_func.get_scalar_reward, disc.log_density_ratio, v['obj'],
                log_folder, env_steps, gym_env.range_lim,
                sac_info, disc_loss, metrics)

    logger.record_tabular(f"{policy_type} Update Time", update_time)
    logger.record_tabular(f"{policy_type} Env Steps", env_steps)

    return real_return_det, real_return_sto

def log_metrics(itr: int, sac_agent, uncertainty_coef: float, loss: float, writer: SummaryWriter, v: dict):
    """
    Log training metrics to tensorboard
    
    Args:
        itr: Current iteration number
        sac_agent: SAC agent instance
        uncertainty_coef: Uncertainty coefficient for exploration
        loss: Current reward loss value
        writer: Tensorboard SummaryWriter instance
        v: Config dictionary
    """
    # Calculate global step
    global_step = itr * v['sac']['epochs'] * v['env']['T']
    
    # Log SAC training metrics
    writer.add_scalar('SAC/Alpha', 
                     sac_agent.alpha.item() if v['sac']['automatic_alpha_tuning'] else v['sac']['alpha'], 
                     global_step)
    writer.add_scalar('SAC/Uncertainty_Coefficient', uncertainty_coef, global_step)
    
    # Log average Q-values and their std
    q_values, q_stds = sac_agent.get_q_stats()
    writer.add_scalar('SAC/Average_Q', q_values, global_step)
    writer.add_scalar('SAC/Q_Std', q_stds, global_step)
    
    # Log reward loss
    writer.add_scalar('Training/Reward_Loss', loss.item(), global_step)
    
    return global_step

if __name__ == "__main__":
    yaml = YAML()
    v = yaml.load(open(sys.argv[1]))
    num_q_pairs = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else v['seed']
    uncertainty_coef = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0

    print("num_q_pairs", num_q_pairs)
    print("seed", seed)

    # common parameters
    env_name = v['env']['env_name']
    state_indices = v['env']['state_indices']
    num_expert_trajs = v['irl']['expert_episodes']

    # system: device, threads, seed, pid
    device = torch.device(f"cuda:{v['cuda']}" if torch.cuda.is_available() and v['cuda'] >= 0 else "cpu")
    torch.set_num_threads(1)
    np.set_printoptions(precision=3, suppress=True)
    system.reproduce(seed)
    pid=os.getpid()
    
    # assumptions
    assert v['obj'] in ['fkl', 'rkl', 'js', 'emd', 'maxentirl']
    assert v['IS'] == False

    # logs
    exp_id = f"logs/{env_name}/exp-{num_expert_trajs}/{v['obj']}" # task/obj/date structure
    # exp_id = 'debug'
    if not os.path.exists(exp_id):
        os.makedirs(exp_id)

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    log_folder = exp_id + '/' + now.strftime('%Y_%m_%d_%H_%M_%S') + f'_q{num_q_pairs}_seed{seed}'
    logger.configure(dir=log_folder)            
    writer = SummaryWriter(log_folder)
    print(f"Logging to directory: {log_folder}")
    os.system(f'cp firl/irl_samples.py {log_folder}')
    os.system(f'cp {sys.argv[1]} {log_folder}/variant_{pid}.yml')
    with open(os.path.join(logger.get_dir(), 'variant.json'), 'w') as f:
        json.dump(v, f, indent=2, sort_keys=True)
    print('pid', pid)
    os.makedirs(os.path.join(log_folder, 'plt'))
    os.makedirs(os.path.join(log_folder, 'model'))

    # environment
    env_fn = lambda: gym.make(env_name)
    gym_env = env_fn()
    state_size = gym_env.observation_space.shape[0]
    action_size = gym_env.action_space.shape[0]
    if state_indices == 'all':
        state_indices = list(range(state_size))

    # load expert samples from trained policy
    expert_trajs = torch.load(f'expert_data/states/{env_name}.pt').numpy()[:, :, state_indices]
    expert_trajs = expert_trajs[:num_expert_trajs, :, :] # select first expert_episodes
    expert_samples = expert_trajs.copy().reshape(-1, len(state_indices))
    print(expert_trajs.shape, expert_samples.shape) # ignored starting state

    # Initilialize reward as a neural network
    reward_func = MLPReward(len(state_indices), **v['reward'], device=device).to(device)
    reward_optimizer = torch.optim.Adam(reward_func.parameters(), lr=v['reward']['lr'], 
        weight_decay=v['reward']['weight_decay'], betas=(v['reward']['momentum'], 0.999))
    
    # Initilialize discriminator
    if v['obj'] in ["emd"]:
        critic = Critic(len(state_indices), **v['critic'], device=device)
    elif v['obj'] != 'maxentirl':
        disc = Disc(len(state_indices), **v['disc'], device=device)

    max_real_return_det, max_real_return_sto = -np.inf, -np.inf
    for itr in range(v['irl']['n_itrs']):

        if v['sac']['reinitialize'] or itr == 0:
            # Reset SAC agent with old policy, new environment, and new replay buffer
            print("Reinitializing sac")
            replay_buffer = ReplayBuffer(
                state_size, 
                action_size,
                device=device,
                size=v['sac']['buffer_size'])
                
            sac_agent = SAC(env_fn, replay_buffer,
                steps_per_epoch=v['env']['T'],
                update_after=v['env']['T'] * v['sac']['random_explore_episodes'], 
                max_ep_len=v['env']['T'],
                seed=seed,
                start_steps=v['env']['T'] * v['sac']['random_explore_episodes'],
                reward_state_indices=state_indices,
                device=device,
                num_q_pairs=int(num_q_pairs),
                uncertainty_coef=uncertainty_coef,
                **v['sac']
            )
        
        sac_agent.reward_function = reward_func.get_scalar_reward # only need to change reward in sac
        sac_info = sac_agent.learn_mujoco(print_out=True)

        start = time.time()
        samples = collect.collect_trajectories_policy_single(gym_env, sac_agent, 
                        n = v['irl']['training_trajs'], state_indices=state_indices)
        # Fit a density model using the samples
        agent_emp_states = samples[0].copy()
        agent_emp_states = agent_emp_states.reshape(-1,agent_emp_states.shape[2]) # n*T states
        print(f'collect trajs {time.time() - start:.0f}s', flush=True)
        # print(agent_emp_states.shape)

        start = time.time()
        if v['obj'] in ["emd"]:
            critic_loss = critic.learn(expert_samples.copy(), agent_emp_states, iter=v['critic']['iter'])
        elif v['obj'] != 'maxentirl': # learn log density ratio
            disc_loss = disc.learn(expert_samples.copy(), agent_emp_states, iter=v['disc']['iter'])
        print(f'train disc {time.time() - start:.0f}s', flush=True)

        # optimization w.r.t. reward
        reward_losses = []
        for _ in range(v['reward']['gradient_step']):
            if v['irl']['resample_episodes'] > v['irl']['expert_episodes']:
                expert_res_indices = np.random.choice(expert_trajs.shape[0], v['irl']['resample_episodes'], replace=True)
                expert_trajs_train = expert_trajs[expert_res_indices].copy() # resampling the expert trajectories
            elif v['irl']['resample_episodes'] > 0:
                expert_res_indices = np.random.choice(expert_trajs.shape[0], v['irl']['resample_episodes'], replace=False)
                expert_trajs_train = expert_trajs[expert_res_indices].copy()
            else:
                expert_trajs_train = None # not use expert trajs

            if v['obj'] in ['fkl', 'rkl', 'js']:
                loss, _ = f_div_disc_loss(v['obj'], v['IS'], samples, disc, reward_func, device, expert_trajs=expert_trajs_train)             
            elif v['obj'] in ['fkl-state', 'rkl-state', 'js-state']:
                loss = f_div_current_state_disc_loss(v['obj'], samples, disc, reward_func, device, expert_trajs=expert_trajs_train)
            elif v['obj'] == 'maxentirl':
                loss = maxentirl_loss(v['obj'], samples, expert_samples, reward_func, device)
            elif v['obj'] == 'emd':
                loss, _ = ipm_loss(v['obj'], v['IS'], samples, critic.value, reward_func, device, expert_trajs=expert_trajs_train)  
            
            reward_losses.append(loss.item())
            print(f"{v['obj']} loss: {loss}")
            reward_optimizer.zero_grad()
            loss.backward()
            reward_optimizer.step()

        # Log metrics and get global step
        global_step = log_metrics(itr, sac_agent, uncertainty_coef, loss, writer, v)
        
        # evaluating the learned reward
        real_return_det, real_return_sto = try_evaluate(itr, "Running", sac_info, writer, global_step)
        if real_return_det > max_real_return_det and real_return_sto > max_real_return_sto:
            max_real_return_det, max_real_return_sto = real_return_det, real_return_sto
            torch.save(reward_func.state_dict(), os.path.join(logger.get_dir(), 
                    f"model/reward_model_itr{itr}_det{max_real_return_det:.0f}_sto{max_real_return_sto:.0f}.pkl"))

        logger.record_tabular("Itration", itr)
        logger.record_tabular("Reward Loss", loss.item())
        if v['sac']['automatic_alpha_tuning']:
            logger.record_tabular("alpha", sac_agent.alpha.item())
        
        # if v['irl']['save_interval'] > 0 and (itr % v['irl']['save_interval'] == 0 or itr == v['irl']['n_itrs']-1):
        #     torch.save(reward_func.state_dict(), os.path.join(logger.get_dir(), f"model/reward_model_{itr}.pkl"))

        logger.dump_tabular()

    writer.close()