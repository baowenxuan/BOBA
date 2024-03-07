import torch
from sklearn.decomposition import PCA
from itertools import combinations
from tqdm import tqdm
import time

from .Aggregator import Aggregator


class BOBA(Aggregator):

    def __init__(self, args):
        super(BOBA, self).__init__(args)
        # for stage 1
        self.k = args.num_clients - args.num_byz_resist
        self.c = args.num_labels
        self.pca = PCA(n_components=self.c - 1)
        self.max_iter = args.boba_max_iter

        # for stage 2
        self.pmin = args.boba_pmin

        # for recording
        self.num_pca_call = []
        self.pca_losses = []

        self.server_todo = 'matrix'

    def stage1(self, matrix, server_matrix):

        num_client, length = matrix.shape

        # initialize the subspace with server vectors
        print(server_matrix.shape)
        self.pca.fit(server_matrix)
        neighbors_indices = torch.Tensor([])

        loss = float('inf')
        self.pca_losses.append([])

        for it in range(self.max_iter):
            print(it, end=' ')

            # vectors selection
            # choose k vectors that is close to the subspace
            latent = self.pca.transform(matrix)
            reconstructed = torch.Tensor(self.pca.inverse_transform(latent))
            dist = torch.square(torch.norm(matrix - reconstructed, dim=1))
            new_indices = torch.argsort(dist)[:self.k]
            loss = dist[new_indices].sum()
            self.pca_losses[-1].append(loss)

            if self.verbose:
                print('Iter %d, Loss %f' % (it + 1, loss))

            # break if the algorithm already converge
            # torch.Tensor -> numpy.array -> list of int,
            # otherwise it will be list of tensor which cannot be compared
            if set(neighbors_indices.numpy()) == set(new_indices.numpy()):
                neighbors_indices = new_indices
                self.num_pca_call.append(it + 1)
                break

            neighbors_indices = new_indices

            # subspace update
            self.pca.fit(matrix[neighbors_indices])

        # note that when the algorithm stops, the pca is already the best.

        return neighbors_indices, loss

    def stage2(self, matrix, server_matrix):

        print('explained', self.pca.explained_variance_ratio_.sum())

        latent = torch.Tensor(self.pca.transform(matrix))  # n * (c - 1)
        server_latent = torch.Tensor(self.pca.transform(server_matrix))  # c * (c - 1)

        latent_one = torch.cat([latent, torch.ones(latent.shape[0], 1)], dim=1)
        server_latent_one = torch.cat([server_latent, torch.ones(server_latent.shape[0], 1)], dim=1)

        coef = latent_one @ torch.pinverse(server_latent_one)
        low_coef = torch.min(coef, dim=1)[0]
        sorted_coef, indices = torch.sort(low_coef)

        if sorted_coef[-self.k] > self.pmin:
            correct_indices = torch.where(low_coef >= self.pmin)
        else:
            correct_indices = (indices[-self.k:],)

        if self.verbose:
            print(sorted_coef)
            print('Remained: %d / %d' % (len(correct_indices[0]), coef.shape[0]))

        latent_agg = latent[correct_indices].mean(dim=0)
        agg = self.pca.inverse_transform(latent_agg)
        agg_vector = torch.Tensor(agg)

        return agg_vector

    def aggregate(self, matrix, origin, server_matrix):
        tik = time.time()
        matrix = matrix.cpu()
        server_matrix = server_matrix.cpu()
        neighbors_indices, loss = self.stage1(matrix, server_matrix)
        agg_vector = self.stage2(matrix, server_matrix)
        tok = time.time()
        self.running_times.append(tok - tik)

        return agg_vector.to(self.device)


# Below is for ablation study

class BOBA_ES(BOBA):

    def stage1(self, matrix, server_matrix):
        num_client, length = matrix.shape

        best_loss = float('inf')
        best_indices = None

        for selected_indices in tqdm(combinations(range(num_client), self.k)):

            selected_indices = torch.LongTensor(selected_indices)
            self.pca.fit(matrix[selected_indices])
            latent = self.pca.transform(matrix)
            reconstructed = torch.Tensor(self.pca.inverse_transform(latent))
            dist = torch.square(torch.norm(matrix - reconstructed, dim=1))
            neighbors_indices = torch.argsort(dist)[:self.k]
            loss = dist[neighbors_indices].sum()

            if loss < best_loss:
                best_indices = selected_indices
                best_loss = loss

        return best_indices, best_loss


class BOBA_No_Stage1(BOBA):

    def stage1(self, matrix, server_matrix):
        self.pca.fit(server_matrix)
        latent = self.pca.transform(matrix)
        reconstructed = torch.Tensor(self.pca.inverse_transform(latent))
        dist = torch.square(torch.norm(matrix - reconstructed, dim=1))
        new_indices = torch.argsort(dist)[:self.k]
        loss = dist[new_indices].sum()

        return new_indices, loss


class BOBA_No_Stage2(BOBA):

    def stage2(self, matrix, server_matrix):
        latent = torch.Tensor(self.pca.transform(matrix))  # n * (c - 1)
        latent_agg = latent.mean(dim=0)
        agg = self.pca.inverse_transform(latent_agg)
        agg_vector = torch.Tensor(agg)

        return agg_vector
