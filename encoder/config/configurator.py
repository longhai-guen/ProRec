import os
import yaml
import torch
import pickle
import argparse
import numpy as np
import torch.nn as nn

PROFILE_MODES = ['real', 'none', 'shuffled', 'random']
SCORE_MODES = ['model', 'cf', 'llm', 'all']
SIDE_MODES = ['both', 'user', 'item']


def profile_tag(model_configs):
    """Run label used in file names and records: e.g. 'real', 'shuffled', 'real-user', 'real-item'."""
    profile = model_configs.get('profile', 'real')
    side = model_configs.get('profile_side', 'both')
    return profile if side == 'both' else '{}-{}'.format(profile, side)


def perturb_profile(embeds, mode, seed, side, idx=0):
    """Ablation controls for LLM profile embeddings.

    real:     unchanged.
    shuffled: rows permuted, so each user/item gets another entity's profile. The permutation
              depends only on (seed, side), so all profile views of one entity move together.
    random:   Gaussian noise matching the per-dimension mean/std of the real embeddings.
    """
    embeds = np.asarray(embeds)
    side_offset = {'user': 0, 'item': 1}[side]
    if mode == 'shuffled':
        perm = np.random.RandomState(seed * 10 + side_offset).permutation(embeds.shape[0])
        return embeds[perm]
    if mode == 'random':
        rs = np.random.RandomState(seed * 100 + idx * 10 + side_offset)
        mean, std = embeds.mean(axis=0), embeds.std(axis=0)
        return (rs.standard_normal(embeds.shape) * std + mean).astype(np.float32)
    return embeds


def parse_configure(model=None, dataset=None):
    parser = argparse.ArgumentParser(description='RLMRec')
    parser.add_argument('--model', type=str, default='lightgcn_promax', help='Model name')
    parser.add_argument('--dataset', type=str, default='amazon', help='Dataset name')
    parser.add_argument('--device', type=str, default='cuda', help='cpu or cuda')
    parser.add_argument('--seed', type=int, default=None, help='Device number')
    parser.add_argument('--cuda', type=str, default='0', help='Device number')
    parser.add_argument('--profile', type=str, default='real', choices=PROFILE_MODES,
                        help='LLM profile input: real | none (pure LightGCN) | shuffled | random')
    parser.add_argument('--score', type=str, default='model', choices=SCORE_MODES,
                        help='Final-test scorer: model (method default) | cf | llm | all')
    parser.add_argument('--profile_side', type=str, default='both', choices=SIDE_MODES,
                        help='Tier 2: side that keeps --profile; the other side gets random noise (content-free)')
    parser.add_argument('--save_topk', type=int, default=0,
                        help='Tier 2: save per-user top-K lists per scorer and the final embeddings; 0 = off')
    args, _ = parser.parse_known_args()

    # cuda
    if args.device == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

    # model name
    if model is not None:
        model_name = model.lower()
    elif args.model is not None:
        model_name = args.model.lower()
    else:
        model_name = 'default'
        # print("Read the default (blank) configuration.")

    # dataset
    if dataset is not None:
        args.dataset = dataset

    # find yml file
    if not os.path.exists('../encoder/config/modelconf/{}.yml'.format(model_name)):
        raise Exception("Please create the yaml file for your model first.")

    # read yml file
    with open('../encoder/config/modelconf/{}.yml'.format(model_name), encoding='utf-8') as f:
        config_data = f.read()
        configs = yaml.safe_load(config_data)
        configs['model']['name'] = configs['model']['name'].lower()

        if args.model is not None:
            if configs['model']['name'] != args.model:
                raise Exception("Please check whether the model names are consistent.")

        if 'tune' not in configs:
            configs['tune'] = {'enable': False}
        configs['device'] = args.device
        if args.dataset is not None:
            configs['data']['name'] = args.dataset
        if args.seed is not None:
            configs['train']['seed'] = args.seed

        # ablation toggles: profile input and final-test scorers
        is_lightgcn = model_name.startswith('lightgcn')
        if args.profile == 'none' and not is_lightgcn:
            raise Exception("--profile none is only supported for LightGCN-based models.")
        if args.score != 'model' and not is_lightgcn:
            raise Exception("--score {} is only supported for LightGCN-based models.".format(args.score))
        if args.score == 'all':
            scores = ['model', 'cf', 'llm'] if args.profile != 'none' else ['model']
        else:
            scores = [args.score]
        if args.profile == 'none' and 'llm' in scores:
            raise Exception("--score llm needs profiles; it cannot be used with --profile none.")
        if args.profile_side != 'both' and args.profile in ('none', 'random'):
            raise Exception("--profile_side user/item needs --profile real or shuffled.")
        configs['model']['profile'] = args.profile
        configs['model']['profile_side'] = args.profile_side
        # perturbation per side: the side that is not kept gets random noise
        configs['model']['profile_modes'] = {
            side: args.profile if args.profile_side in ('both', side) else 'random' for side in ('user', 'item')}
        configs['test']['scores'] = scores
        configs['test']['save_topk'] = args.save_topk

        if args.profile == 'none':
            return configs

        # semantic embeddings
        usrprf_embeds_path = "../data/{}/usr_emb_np.pkl".format(configs['data']['name'])
        itmprf_embeds_path = "../data/{}/itm_emb_np.pkl".format(configs['data']['name'])
        seed = configs['train']['seed']
        with open(usrprf_embeds_path, 'rb') as f:
            configs['usrprf_embeds'] = perturb_profile(pickle.load(f), configs['model']['profile_modes']['user'], seed, 'user')
        with open(itmprf_embeds_path, 'rb') as f:
            configs['itmprf_embeds'] = perturb_profile(pickle.load(f), configs['model']['profile_modes']['item'], seed, 'item')

        return configs

configs = parse_configure()
