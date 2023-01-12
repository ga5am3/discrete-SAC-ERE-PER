from collections import deque
import random
import glob
import gym
import torch
import wandb
from DSAC import DSACAgent
import numpy as np
import wandb
import argparse

def train(config, **kwargs):

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    env = gym.env(config.env)
    
    env.seed(config.seed)
    env.action_space.seed(config.seed)

    eta_0 = 0.996
    eta_T = 1.
    c_k_min = 1500
    max_episode_len = 400
    c_k_min = 2500

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    steps = 0
    average10 = deque(maxlen=10)
    total_steps = 0
    
    methods = {'DSAC': DSACAgent}
    agent = methods[config.algorithm]

    with wandb.init(project=config.project, name=config.run_name, config=config):
        
        agent = agent(state_size=env.observation_space.shape[0],
                         action_size=env.action_space.n,
                         device=device)

        wandb.watch(agent, log="gradients", log_freq=10)


        agent.collect_random(env=env, num_samples=10000)
        
        if config.log_video:
            env = gym.wrappers.Monitor(env, './video', video_callable=lambda x: x%10==0, force=True)

        for i in range(1, config.episodes+1):
            state = env.reset()
            episode_steps = 0
            rewards = 0
            while True:
                action, entropy = agent.get_action(state)
                steps += 1
                next_state, reward, done, _ = env.step(action)
                agent.add_exp(state, action, reward, next_state, done, entropy)
                state = next_state
                rewards += reward
                episode_steps += 1
                policy_loss, alpha_loss, bellmann_error1, bellmann_error2, current_alpha= agent.learn()
                eta_t = eta_0 + (eta_T - eta_0)*(steps/T)
                if done:
                    if config.ERE:
                        
                        for k in range(1, min(i, 50)):
                            c_k = max(int(len(agent.memory)*eta_t**(k*(max_episode_len/i))), c_k_min) # replace using numpy
                            policy_loss, alpha_loss, bellmann_error1, bellmann_error2, current_alpha = agent.step(c_k, config.batch_size)
                    break

            

            average10.append(rewards)
            total_steps += episode_steps
            print("Episode: {} | Reward: {} | Polciy Loss: {} | Steps: {}".format(i, rewards, policy_loss, steps,))
            
            wandb.log({"Reward": rewards,
                       "Average10": np.mean(average10),
                       "Steps": total_steps,
                       "Policy Loss": policy_loss,
                       "Alpha Loss": alpha_loss,
                       "Bellmann error 1": bellmann_error1,
                       "Bellmann error 2": bellmann_error2,
                       "Alpha": current_alpha,
                       "Steps": steps,
                       "Episodes": i,
                       "Buffer size": agent.memory.__len__()})

            if (i %10 == 0) and config.log_video:
                mp4list = glob.glob('video/*.mp4')
                if len(mp4list) > 1:
                    mp4 = mp4list[-2]
                    wandb.log({"gameplays": wandb.Video(mp4, caption='episode: '+str(i-10), fps=4, format="gif"), "Episode": i})

            if i % config.save_every == 0:
                agent.save(config, save_name="SAC_discrete", wandb=wandb, ep=i)

            
def get_config():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument("--run_name", type=str, default="DSAC", help="Run name, default: SAC")
    parser.add_argument("--algorithm", type=str, default="DSAC", help="Options: DSAC")
    parser.add_argument("--ERE", type=bool, default=False, help="Whether or not to use ERE")
    parser.add_argument("--memory_type", type=str, default="Replay", help="Options: Replay, PER")
    parser.add_argument("--env", type=str, default="LunarLander-v2", help="Gym environment name, default: CartPole-v0")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes, default: 1000")
    parser.add_argument("--buffer_size", type=int, default=100_000, help="Maximal training dataset size, default: 100_000")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate, default:5e-4")
    parser.add_argument("--gama", type=float, default=0.99, help="Discounting Factor, default:0.99")
    parser.add_argument("--beta", type=float, default=0.01, help="Entropy Penalty, default: 0.01")
    parser.add_argument("--tau", type=float, default=1e-2, help="interpolation parameter, default: 1e-2")
    parser.add_argument("--clip_grad_param", type=float, default=1, help="clip grad param, default:1")
    parser.add_argument("--hidden_size", type=int, default=256, help="Size of Hidden layers, default:256")
    parser.add_argument("--buffer_size", type=int, default=100_000, help="Buffer size, default: 100000")
    parser.add_argument("--seed", type=int, default=250, help="Seed, default: 250")
    parser.add_argument("--log_video", type=int, default=0, help="Log agent behaviour to wanbd when set to 1, default: 0")
    parser.add_argument("--beta_annealign_params", type=list, default=[0.3, 7000], help="[beta_start, beta_frames]")
    parser.add_argument("--save_every", type=int, default=100, help="Saves the network every x epochs, default: 25")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size, default: 256")
    
    args = parser.parse_args()
    return args
        

if __name__ == "__main__":
    config = get_config()
    train(config)