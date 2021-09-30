from datetime import datetime
import os
import time

from gym.spaces import Space

import numpy as np
import statistics
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# from rl_pytorch.ppo import RolloutStorage
import sys
sys.path.append('../../')
from ppo import RolloutStorage
from copy import deepcopy
import math

class PPO2:

    def __init__(self,
                 vec_env,
                 actor_critic_class,
                 num_transitions_per_env,
                 num_learning_epochs,
                 num_mini_batches,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 init_noise_std=1.0,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=0.5,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=None,
                 model_cfg=None,
                 device='cpu',
                 sampler='sequential',
                 log_dir='run',
                 is_testing=False,
                 print_log=True,
                 apply_reset=False,
                 asymmetric=False
                 ):

        if not isinstance(vec_env.observation_space, Space):
            raise TypeError("vec_env.observation_space must be a gym Space")
        if not isinstance(vec_env.state_space, Space):
            raise TypeError("vec_env.state_space must be a gym Space")
        if not isinstance(vec_env.action_space, Space):
            raise TypeError("vec_env.action_space must be a gym Space")
        self.observation_space = vec_env.observation_space
        self.action_space = vec_env.action_space
        self.state_space = vec_env.state_space

        self.device = device
        self.asymmetric = asymmetric

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.step_size = learning_rate

        self.sampler = sampler
        # PPO components
        self.vec_env = vec_env
        self.actor_critic = actor_critic_class(self.observation_space.shape, self.state_space.shape, self.action_space.shape,
                                               init_noise_std, model_cfg, asymmetric=asymmetric)
        self.actor_critic.to(self.device)
        # self.storage = RolloutStorage(self.vec_env.num_envs, num_transitions_per_env, self.observation_space.shape,
        #                               self.state_space.shape, self.action_space.shape, self.device, sampler)
        
        # total transitions per train step/ 30
        print("size:", num_transitions_per_env/30)
        # self.storage = RolloutStorage(self.vec_env.num_envs, int(num_transitions_per_env/30), self.observation_space.shape,
        #                               self.state_space.shape, self.action_space.shape, self.device, sampler)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        # self.num_mini_batches = num_mini_batches
        self.num_transitions_per_env = num_transitions_per_env
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        # Log
        self.log_dir = log_dir
        self.print_log = print_log
        self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        self.tot_timesteps = 0
        self.tot_time =  4499 # 93490
        self.is_testing = is_testing
        self.current_learning_iteration = 0

        self.apply_reset = apply_reset

        # Other Bao's stuff
        self.mini_batch_size = model_cfg["mini_batch_size"]
        self.max_num_eps = model_cfg["num_eps_per_train_iteration"]
        print("====number of epoch:", self.num_learning_epochs)

    def test(self, path):
        self.actor_critic.load_state_dict(torch.load(path))
        self.actor_critic.eval()

    def load(self, path):
        self.actor_critic.load_state_dict(torch.load(path))
        self.current_learning_iteration = int(path.split("_")[-1].split(".")[0])
        self.actor_critic.train()

    def save(self, path):
        torch.save(self.actor_critic.state_dict(), path)

    def run(self, num_learning_iterations, log_interval=1):
        # current_obs = self.vec_env.reset()
        # current_states = self.vec_env.get_state()

        if self.is_testing:
            current_obs = self.vec_env.reset()
            current_states = self.vec_env.get_state()
            while True:
                with torch.no_grad():
                    for t in range(10):
                        if t == 0:
                            if self.apply_reset:
                                current_obs = self.vec_env.reset()
                            # Compute the action
                            actions = self.actor_critic.act_inference(current_obs)
                            # Step the vec_environment
                            next_obs, rews, dones, infos = self.vec_env.step(actions,0)
                            current_obs.copy_(next_obs)
                        else:
                            self.vec_env.step(actions, t, False)

        # if self.is_testing:
        #     while True:
        #         with torch.no_grad():
        #             if self.apply_reset:
        #                 current_obs = self.vec_env.reset()
        #             # Compute the action
        #             actions = self.actor_critic.act_inference(current_obs)
        #             # Step the vec_environment
        #             next_obs, rews, dones, infos = self.vec_env.step(actions)
        #             current_obs.copy_(next_obs)
        else:
            rewbuffer = deque(maxlen=100)
            lenbuffer = deque(maxlen=100)
            cur_reward_sum = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)
            cur_episode_length = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)

            reward_sum = []
            episode_length = []

            done_all_steps = False
            total_data_size = 2868
            for it in range(self.current_learning_iteration, num_learning_iterations):
                start = time.time()
                ep_infos = []
                self.batch_size = 0
                print("=====total number of samples:", total_data_size)
                # Rollout
                placeholder_size = self.vec_env.task.max_episode_length * self.max_num_eps * 2  #double for HER
                self.storage = RolloutStorage(self.vec_env.num_envs, placeholder_size, self.observation_space.shape,
                                                self.state_space.shape, self.action_space.shape, self.device, self.sampler)

                for ep in range(self.max_num_eps):   
                    saved_transitions = []                 
                    current_obs = self.vec_env.reset()
                    current_states = self.vec_env.get_state()                    
                    while(not self.vec_env.task.finished_eps):
                        t = self.vec_env.task.step_count
                        # print("--step:", self.vec_env.task.step_count)
                        # print("self.vec_env.task.progress_buf[0]", self.vec_env.task.progress_buf[0])
                        
                        if self.apply_reset:
                            current_obs = self.vec_env.reset()
                            current_states = self.vec_env.get_state()
                        
                                                                       
                        if t % 10 == 0:
                            # Compute the action 
                            actions, actions_log_prob, values, mu, sigma = self.actor_critic.act(current_obs, current_states)
                            self.vec_env.step(actions, t, False)
                        
                        elif t % 10 == 9:
                            # Step the vec_environment
                            next_obs, rews, dones, infos = self.vec_env.step(actions, t)
                            next_states = self.vec_env.get_state()
                            eps_length = deepcopy(self.vec_env.task.progress_buf[0].cpu().numpy())
                            
                            # Record the transition
                            self.storage.add_transitions(current_obs, current_states, actions, rews, dones, values, actions_log_prob, mu, sigma)
                            saved_transitions.append([current_obs, current_states, actions, rews, dones, values, actions_log_prob, mu, sigma, next_obs])
                            current_obs.copy_(next_obs)
                            current_states.copy_(next_states)
                            

                            # Book keeping
                            ep_infos.append(infos)

                            if self.print_log:
                                cur_reward_sum[:] += rews
                                cur_episode_length[:] += 1

                                new_ids = (dones > 0).nonzero(as_tuple=False)
                                reward_sum.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                                episode_length.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                                cur_reward_sum[new_ids] = 0
                                cur_episode_length[new_ids] = 0 

                        else:
                            self.vec_env.step(actions, t, False)

                    # self.batch_size += eps_length     # for NO HER
                    if len(saved_transitions) == 0:
                        print("==== WARNING, saved_transitions = 0 =====")
                    else:
                        # Bao HER
                        assert self.vec_env.task.finished_eps == True
                        reevaluated_transitions = self.vec_env.task.reevaluate_reward(saved_transitions)
                        self.batch_size += eps_length * 2
                        for tr in reevaluated_transitions:
                            self.storage.add_transitions(*tr)                                      
                    
                    self.vec_env.task.finished_eps = False
                    print("-----Ending eps-----", ep + 1)

                # self.batch_size *= 2
                self.storage.modify_storage_size(self.batch_size)
                total_data_size += self.batch_size
                print("BATCH SIZE:", self.batch_size)             
                self.record_batch_size(it, self.batch_size, total_data_size)


                if self.print_log:
                    # reward_sum = [x[0] for x in reward_sum]
                    # episode_length = [x[0] for x in episode_length]
                    rewbuffer.extend(reward_sum)
                    lenbuffer.extend(episode_length)

                _, _, last_values, _, _ = self.actor_critic.act(current_obs, current_states)
                stop = time.time()
                collection_time = stop - start

                mean_trajectory_length, mean_reward = self.storage.get_statistics()

                # Learning step
                start = stop
                self.storage.compute_returns(last_values, self.gamma, self.lam)
                mean_value_loss, mean_surrogate_loss = self.update()
                self.storage.clear()
                stop = time.time()
                learn_time = stop - start
                if self.print_log:
                    self.log(locals())
                if it % log_interval == 0:
                    self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
                ep_infos.clear()
            self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(num_learning_iterations)))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_transitions_per_env * self.vec_env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.actor_critic.log_std.exp().mean()

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        self.writer.add_scalar('Train2/mean_reward/step', locs['mean_reward'], locs['it'])
        self.writer.add_scalar('Train2/mean_episode_length/episode', locs['mean_trajectory_length'], locs['it'])

        fps = int(self.num_transitions_per_env * self.vec_env.num_envs / (locs['collection_time'] + locs['learn_time']))

        str = f" \033[1m Learning iteration {locs['it']}/{locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
                          f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                          f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                          f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        # print("self.storage.observations.shape:", self.storage.observations.shape)
        # count = 0 
        # batch = self.storage.mini_batch_generator(self.num_mini_batches)

        self.num_mini_batches = math.ceil(self.batch_size/self.mini_batch_size)
        batch = self.storage.mini_batch_generator(self.num_mini_batches, self.batch_size, self.mini_batch_size)
        # print("self.num_mini_batches:", self.num_mini_batches)
        # print("*****batch size:", len(batch))
        
        for epoch in range(self.num_learning_epochs):
            # for obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch \
            #        in self.storage.mini_batch_generator(self.num_mini_batches):
            # print("-------------------")
            for indices in batch:
                # print("indices",indices)
                # print(count)
                # count += 1
                obs_batch = self.storage.observations.view(-1, *self.storage.observations.size()[2:])[indices]
                # print("=====obs_batch.shape", obs_batch.shape)
                if self.asymmetric:
                    states_batch = self.storage.states.view(-1, *self.storage.states.size()[2:])[indices]
                else:
                    states_batch = None
                actions_batch = self.storage.actions.view(-1, self.storage.actions.size(-1))[indices]
                # print("actions_batch.shape", actions_batch.shape)
                # print("actions_batch:", actions_batch)
                target_values_batch = self.storage.values.view(-1, 1)[indices]
                returns_batch = self.storage.returns.view(-1, 1)[indices]
                old_actions_log_prob_batch = self.storage.actions_log_prob.view(-1, 1)[indices]
                advantages_batch = self.storage.advantages.view(-1, 1)[indices]
                old_mu_batch = self.storage.mu.view(-1, self.storage.actions.size(-1))[indices]
                old_sigma_batch = self.storage.sigma.view(-1, self.storage.actions.size(-1))[indices]

                actions_log_prob_batch, entropy_batch, value_batch, mu_batch, sigma_batch = self.actor_critic.evaluate(obs_batch,
                                                                                                                       states_batch,
                                                                                                                       actions_batch)

                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':

                    kl = torch.sum(
                        sigma_batch - old_sigma_batch + (torch.square(old_sigma_batch.exp()) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch.exp())) - 0.5, axis=-1)
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.step_size = max(1e-5, self.step_size / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.step_size = min(1e-2, self.step_size * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.step_size

                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                   1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates

        return mean_value_loss, mean_surrogate_loss


    def record_batch_size(self, iter, batch_size, total_num_sample):
        self.writer.add_scalar('Data size over iterations', batch_size, iter)  
        self.writer.add_scalar('Total number of sample over iterations', total_num_sample, iter) 