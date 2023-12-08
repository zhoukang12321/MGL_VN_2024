import os
os.environ["OMP_NUM_THREADS"] = '1'
import time
import trainers
import models
import agents
import environment as env
import optimizers
import torch
from tqdm import tqdm
from utils.thordata_utils import get_scene_names, random_divide, get_type, get_test_set
from utils.env_wrapper import make_envs, VecEnv
import numpy as np
from utils.mean_calc import ScalarMeanTracker, LabelScalarTracker
from utils.init_func import search_newest_model, get_args, make_exp_dir, load_or_find_model
from utils.record_utils import data_output
#TODO 输出loss
def main():
    #读取参数
    args = get_args(os.path.basename(__file__))

    #确认gpu可用情况
    if args.gpu_ids == -1:
        args.gpu_ids = [-1]
    else:
        #torch.cuda.manual_seed(args.seed)
        assert torch.cuda.is_available()
    
    gpu_id = args.gpu_ids[0]

    #动态载入构造函数
    creator = {
        'model':getattr(models, args.model),
        'agent':getattr(agents, args.agent),
        'env':getattr(env, args.env),
    }
    #生成全局模型并加载参数
    model = creator['model'](**args.model_args)
    if model is not None:  print(model)
    load_dir = load_or_find_model(args)
    if load_dir is not '':
        model.load_state_dict(torch.load(load_dir))

    #这里用于分配各个线程的环境可以加载的场景以及目标
    scene_names_div, chosen_objects, nums_div, test_set_div = get_test_set(args)
    #print(scene_names_div)

    #生成多线程环境，每个线程可以安排不同的房间或者目标
    agent = creator['agent'](
        list(args.action_dict.keys()),
        model,
        args.threads,
        gpu_id
    )
    if args.verbose:
        print('agent created')

    #生成多线程环境，每个线程可以安排不同的房间或者目标
    env_fns = []
    for i in range(args.threads):
        if get_type(scene_names_div[i][0]) == 'living_room':
            args.max_epi_length == 200
        env_args = dict(
            offline_data_dir = args.offline_data_dir,
            action_dict = args.action_dict,
            target_dict = args.target_dict,
            obs_dict = args.obs_dict,
            reward_dict = args.reward_dict,
            max_steps = args.max_epi_length,
            grid_size = args.grid_size,
            rotate_angle = args.rotate_angle,
            chosen_scenes = scene_names_div[i],
            chosen_targets = chosen_objects
        )
        env_fns.append(make_envs(env_args, creator['env']))
    envs = VecEnv(env_fns, eval_mode = True, test_sche = test_set_div)

    #生成实验文件夹
    make_exp_dir(args, 'TEST')

    n_epis_thread = [0 for _ in range(args.threads)]
    thread_steps = [0 for _ in range(args.threads)]
    thread_reward = [0 for _ in range(args.threads)]
    false_action_ratio = [[] for _ in range(args.threads)]

    test_scalars = LabelScalarTracker()

    pbar = tqdm(total=args.total_eval_epi)
    obs = envs.reset()
    while 1:
        agent.clear_mems()
        action, _ = agent.action(obs)
        obs_new, r, done, info = envs.step(action)
        obs = obs_new
        stop = True
        for i in range(args.threads):
            if n_epis_thread[i] < nums_div[i]:
                stop = False
                t_info = info[i]
                thread_reward[i] += r[i]
                false_action_ratio[i].append(t_info['false_action'] / (thread_steps[i]+1))
                #thread_steps[i] += not t_info['agent_done']
                thread_steps[i] += 1
                if done[i]:
                    n_epis_thread[i] += 1
                    pbar.update(1)
                    spl = 0
                    if t_info['success']:
                        assert t_info['best_len'] <= thread_steps[i]
                        spl = t_info['best_len']/thread_steps[i]
                    data = {
                        'ep_length:':thread_steps[i],
                        'SR:':t_info['success'],
                        'SPL:':spl,
                        'total_reward:':thread_reward[i],
                        'epis':1,
                        'false_action_ratio':false_action_ratio[i]
                    }
                    target_str = get_type(t_info['scene_name'])+'/'+t_info['target']
                    for k in [t_info['scene_name'], target_str]:
                        test_scalars[k].add_scalars(data)
                    thread_steps[i] = 0
                    thread_reward[i] = 0
                    false_action_ratio[i] = []
                    agent.reset_hidden(i)
        
        if stop: break
    envs.close()
    pbar.close()
    
    data_output(args, test_scalars)

if __name__ == "__main__":
    main()

