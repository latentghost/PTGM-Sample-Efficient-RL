from sklearn.manifold import TSNE
from sklearn.cluster import KMeans,AgglomerativeClustering
import matplotlib.pyplot as plt
import numpy as np
import random
import os, imageio, sys, pickle
from scipy.spatial.distance import euclidean
sys.path.append(os.path.abspath(''))
from steve1.data.EpisodeStorage import EpisodeStorage
from steve1.data.minecraft_dataset import load_sampling
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import Module
import torch.optim as optim


class Autoencoder(Module):
    def __init__(
        self, 
        input_dim,
        latent_dim
        ):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return z, x_recon



def train_autoencoder(
    embeds, 
    input_dim,
    latent_dim=32,
    epochs=50,
    lr=1e-3
    ):
    autoencoder = Autoencoder(input_dim,latent_dim)
    optimizer = optim.Adam(autoencoder.parameters(),lr=lr)
    criterion = nn.MSELoss()
    
    embeds = torch.tensor(embeds, dtype=torch.float32)
    for epoch in range(epochs):
        optimizer.zero_grad()
        _, recon = autoencoder(embeds)
        loss = criterion(recon, embeds)
        loss.backward()
        optimizer.step()
        if (epoch+1)%10==0:
            print(f"Epoch {epoch+1}/{epochs}\nLoss: {loss.item():.4f}")
    
    latent_embeddings = autoencoder.encoder(embeds).detach().numpy()
    return latent_embeddings



def plot_embedding(data, y, save_path, centers=None):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    #ax = plt.subplot(111)
    plt.scatter(data[:, 0], data[:, 1], c=y)
    
    if centers is not None:
        centers = (centers - x_min) / (x_max - x_min)
        plt.scatter(centers[:, 0], centers[:, 1], c='r')
    plt.savefig(os.path.join(save_path, 'tsne.png'))

'''
def save_frame_labels(frames, labels, path):
    for i, (f, l) in enumerate(zip(frames, labels)):
        p = os.path.join(path, str(l))
        if not os.path.exists(p):
            os.mkdir(p)
        imageio.imsave(os.path.join(p, '{}.png'.format(i)), f)
'''

def save_center_video(paths, steps, codes, center_idxs, output_dir):
    '''
    save the codebook list, and the corresponding 16 frames of each code
    '''
    cts = []
    for _, i in enumerate(center_idxs):
        cts.append(codes[i].reshape((1,-1)))
        episode = EpisodeStorage(paths[i])
        frames = episode.load_frames(only_range=(steps[i], steps[i]+16))
        imageio.mimsave(os.path.join(output_dir, 'center_{}.gif'.format(_)), 
            frames[steps[i]:steps[i]+16], 'GIF', duration=0.05)
    #print(cts)
    with open(os.path.join(output_dir, 'centers.pkl'), 'wb') as f:
        pickle.dump(cts, f)
        #pickle.close()

def dict2array(d):
    ret = []
    for k in d:
        if hasattr(d[k], "__len__"):
            ret += list(d[k])
        else:
            ret.append(d[k])
    #print(d, ret)
    return np.asarray(ret, dtype=float)



def sample_embeddings(files, history_act='add', interval=10):
    '''
    sample goal embeddings from the video dataset, for clustering
    params:
        files: dataset path list
        history_act: how to aggregate last 16 actions to concate with visual embeddings
            add action information can make different behaviors more separable
        interval: how many steps to preserve a goal
    return:
        paths: list of video path
        steps: list of selected index in the video
        codes: list of MineCLIP embedding (for codebook)
        embeds: list of [MineCLIP embedding, actions] (for clustering)
    '''
    act_dim=25 if history_act=='add' else 400
    paths, steps, codes, embeds = [], [], np.empty(shape=(0,512)), np.empty(shape=(0,512+act_dim))

    for f in tqdm(files):
        episode = EpisodeStorage(f)
        es = np.asarray(episode.load_embeds_attn()[15:]).reshape((-1,512))
        #fs = episode.load_frames()[15:]
        pths = [f] * es.shape[0]
        stps = [i for i in range(es.shape[0])]
        acts = episode.load_actions()
        acts = np.asarray([dict2array(a) for a in acts])    
        acts = [np.roll(acts, i, axis=0) for i in range(0,16)] # concat last 16 frames actions
        if history_act=='add':
            acts = np.sum(acts, axis=0)[15:]
        else:
            acts = np.concatenate(acts, axis=1)[15:]
        cs = es
        es = np.concatenate((es, acts), axis=1)
        #frames += fs 
        paths += pths 
        steps += stps
        codes = np.concatenate((codes, cs))
        embeds = np.concatenate((embeds, es))
    embeds = np.asarray(embeds).reshape((-1,512+act_dim))
    codes = np.asarray(codes).reshape((-1,512))
    # preserve a small amount of embeddings with interval
    # because nearby embeddings are very similar. preserving all embeddings makes kmeans very slow
    sample_idxs = np.arange(0, len(paths), interval)
    paths = np.asarray(paths)[sample_idxs]
    steps = np.asarray(steps)[sample_idxs]
    codes = codes[sample_idxs]
    embeds = embeds[sample_idxs]
    print('Loaded {} frames in total'.format(len(paths)))
    #print(len(paths), len(steps), codes.shape, embeds.shape)
    return paths, steps, codes, embeds


def main(args):
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    exp_name = '{}_n{}_s{}'.format(args.embedding, args.n_codebook, args.seed)
    output_dir = os.path.join(args.output_dir, exp_name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)

    data = load_sampling(args.sampling_dir, args.sampling)[0]
    data = random.sample(data, args.video_num)
    print(data)

    paths, steps, codes, embeds = sample_embeddings(data, interval=args.sample_interval)

    if args.embedding == 'tsne':
        tsne = TSNE(n_components=2)
        result = tsne.fit_transform(embeds)
    elif args.embedding == 'autoencoder':
        print('training autoencoder for dimensionality reduction...')
        latent_dim = 32
        input_dim = embeds.shape[1]
        result = train_autoencoder(embeds, input_dim, latent_dim)
    else:
        raise NotImplementedError

    
    if args.clustering == 'k-means':
        print('running kmeans clustering')
        kmeans = KMeans(n_clusters=args.n_codebook, init='k-means++', random_state=args.seed).fit(result)
        y_pred = kmeans.labels_
        centers = kmeans.cluster_centers_
        center_idxs = []
    elif args.clustering == 'hierarchical':
        print('running hierarchical clustering...')
        hierarchical = AgglomerativeClustering(n_clusters=args.n_codebook, affinity='euclidean', linkage='ward').fit(result)
        y_pred = hierarchical.labels_
        clusters = None
        center_idxs = []
    else:
        raise NotImplementedError
    print('done')

    for iclust in range(args.n_codebook):
        cluster_pts = result[y_pred == iclust]
        # get all indices of points assigned to this cluster:
        cluster_pts_indices = np.where(y_pred == iclust)[0]
        cluster_cen = (centers[iclust] if clusters is not None else np.mean(cluster_pts, axis=0))
        min_idx = np.argmin([np.linalg.norm(result[idx] - cluster_cen) for idx in cluster_pts_indices])
        idx = cluster_pts_indices[min_idx]
        #print(idx, cluster_cen, result[idx])
        center_idxs.append(idx)

    plot_embedding(result, y_pred, output_dir, result[center_idxs])
    save_center_video(paths, steps, codes, center_idxs, output_dir)
    #save_frame_labels(frames, y_pred, path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding', type=str, default='tnse')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--output-dir', type=str, default='downloads/codebook/')
    parser.add_argument('--sampling_dir', type=str, default='downloads/samplings/')
    parser.add_argument('--sampling', type=str, default='seed1')
    parser.add_argument('--video-num', type=int, default=50) # number of videos to sample goals
    parser.add_argument('--sample-interval', type=int, default=50) # for each video, sample video_len/interval goals
    parser.add_argument('--n-codebook', type=int, default=100)
    parser.add_argument('--clustering', type=str, default='k-means')

    args = parser.parse_args()
    main(args)
