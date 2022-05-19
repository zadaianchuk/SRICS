import os
import math
import time

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from .modules import TrackerRNN, BatchApply
from .utils import gumbel_softmax, my_softmax, kl_categorical


class PropGTNet(nn.Module):

    def __init__(self, node_dim_in, node_dim, nf_hidden, node_dim_out, recurent_dyn, self_mask,  no_interaction, device):

        super(PropGTNet, self).__init__()
        self.device = device

        self.node_dim_in = node_dim_in
        self.nf_hidden = nf_hidden
        self.node_dim = node_dim
        self.node_dim_out = node_dim_out
        self.no_interaction = no_interaction
        self.recurent_dyn = recurent_dyn
        self.self_mask = self_mask 
        # descrete z smapling 
        self.n_edge_types = 2 # one of the for "no interuction"
        self.hard = True
        self.tau = 0.5

        print(f"Init GNN with node dimention {node_dim}...")
        # self-update
        modules = [
            nn.Linear(node_dim_in, nf_hidden),
            nn.ReLU(), 
            nn.BatchNorm1d(nf_hidden),
            nn.Linear(nf_hidden, node_dim)]
        self.update_state_self = BatchApply(nn.Sequential(*modules))
        if not self.recurent_dyn:
            interaction_input_dim = node_dim_in + node_dim_in
        else: 
            interaction_input_dim = node_dim + node_dim
        
        # interaction value
        modules = [
            nn.Linear(interaction_input_dim, nf_hidden),
            nn.ReLU(),
            nn.BatchNorm1d(nf_hidden),
            nn.Linear(nf_hidden, node_dim)]
        self.update_state_relational = BatchApply(nn.Sequential(*modules))
        # interaction weights
        modules = [
            nn.Linear(interaction_input_dim, nf_hidden),
            nn.ReLU(),
            nn.BatchNorm1d(nf_hidden),
            nn.Linear(nf_hidden, self.n_edge_types)]
        self.update_state_weights = BatchApply(nn.Sequential(*modules))


        # prediction values
        modules = [
            nn.Linear(node_dim, nf_hidden),
            nn.ReLU(),
            nn.BatchNorm1d(nf_hidden),
            nn.Linear(nf_hidden, node_dim_out)]
        
        self.pred_values = BatchApply(nn.Sequential(*modules))
        if self.no_interaction:
            rnn_input_dim = node_dim 
        else:
            rnn_input_dim = node_dim + node_dim

        self.temporal_rnn = BatchApply(TrackerRNN(rnn_input_dim, node_dim))


        prior = np.array([0.05, 0.95]) #TODO: hard coded bias towards sparsity
        log_prior = torch.FloatTensor(np.log(prior))
        log_prior = torch.unsqueeze(log_prior, 0)
        log_prior = torch.unsqueeze(log_prior, 0)
        log_prior = Variable(log_prior)
        log_prior = log_prior.to(self.device)
        self.log_prior = log_prior   

    def forward(self, states, z_where, actions): 
        B = z_where.shape[0]
        updated_state, weights, kl_loss = self.update_state(states, z_where, actions)  
        z_where_next = self.pred_values(updated_state)
        return updated_state, z_where_next.reshape(B, -1), weights, kl_loss

    def update_state(self, states, z_where, actions):
        """
        Object interaction for gnn representation
        Args:
            z: z_t
                states: (B, N, D)
                z_where: (B, N, 2)
                actions: (B, 2)
        Returns:
            updated_states: (B, N, D), z_{t+1}
        """


        B, N, z_where_dim = z_where.size()
        # GT z_what ara one-hot vectors 
        z_what = torch.eye(N)[None].repeat(B, 1, 1).to(self.device)
        actions = actions.unsqueeze(1).repeat(1, N, 1)
        
        # The feature of one object include the following
        # (B, N, N+4)
        feat = torch.cat([z_where, z_what, actions], dim=-1) 
        
        # (B, N, D)
        enc_self = self.update_state_self(feat)
        

        if self.recurent_dyn:
            states_matrix_self = states[:, :, None].expand(B, N, N, self.node_dim)
            states_matrix_other = states[:, None].expand(B, N, N, self.node_dim).clone()
            states_matrix = torch.cat([states_matrix_self, states_matrix_other], dim=-1)
            feat_matrix = states_matrix # for RNN decoder we are using RNN states as features for edge prediction
        else: 
            # Compute weights based on gaussian
            # (B, N, 1, 2)
            z_shift_self = z_where[:, :, None]
            # (B, 1, N, 2)
            z_shift_other = z_where[:, None]
            # (B, N, N, 2)
            dist_matrix = z_shift_self - z_shift_other
            # (B, N, 1, D) -> (B, N, N, D)
            feat_self = feat[:, :, None]
            feat_matrix_self = feat_self.expand(B, N, N, self.node_dim_in)
            # (B, 1, N, D) -> (B, N, N, D)
            feat_other = feat[:, None]
            feat_matrix_other = feat_other.expand(B, N, N, self.node_dim_in)
            # Replace absolute positions with relative ones
            # Must clone. Otherwise there will be multiple write
            feat_matrix_other = feat_matrix_other.clone()
            feat_matrix_other[..., :z_where_dim] = dist_matrix
            feat_matrix = torch.cat([feat_matrix_self, feat_matrix_other], dim=-1)
        relational_matrix = self.update_state_relational(feat_matrix)
       
        # COMPUTE WEIGHTS
        # (B, N, N, 2)
        weight_matrix_logits = self.update_state_weights(feat_matrix)
        weight_matrix = gumbel_softmax(weight_matrix_logits, tau=self.tau, hard=self.hard)
        # (B, N, N, 1)
        weight_matrix = weight_matrix[..., 0:1]

        prob = my_softmax(weight_matrix_logits, -1)
        loss_kl = kl_categorical(prob, self.log_prior, N)

        # Self mask, set diagonal elements to zero.
        if self.self_mask:
            diag = weight_matrix.diagonal(dim1=1, dim2=2)
            diag *= 0.0

        # (B, N1, N2, D) -> (B, N1, D)
        enc_relational = torch.sum(weight_matrix * relational_matrix, dim=2)
     
        if self.recurent_dyn:
            if self.no_interaction:
                # (B, N, D) 
                rnn_input = enc_self
            else: 
                # (B, N, 2*D) 
                rnn_input = torch.cat([enc_self, enc_relational], dim=-1)
            updated_state = self.temporal_rnn(states, rnn_input)
        else:
            if self.no_interaction:
                # (B, N, D) 
                updated_state = enc_self
            else:
                # (B, N, D) 
                updated_state = enc_self + enc_relational

        # (B, N, D), (B, N, N, 1), (1)
        return updated_state, weight_matrix, loss_kl


### linear process without options 
# def gnn(states, z_where, actions):
#     states, z_where, actions = 0, 0, 0
#     node_dim = 0 
#     B, N, z_where_dim = z_where.size()

#     states_matrix_self = states[:, :, None].expand(B, N, N, node_dim)
#     states_matrix_other = states[:, None].expand(B, N, N, node_dim).clone()
#     states_matrix = torch.cat([states_matrix_self, states_matrix_other], dim=-1)
#     feat_matrix = states_matrix

#     relational_matrix = update_state_relational(feat_matrix)
#     weight_matrix_logits = update_state_weights(feat_matrix)
#     weight_matrix = gumbel_softmax(weight_matrix_logits, tau=tau, hard=hard)
#     # (B, N, N, 1)
#     weight_matrix = weight_matrix[..., 0:1]

#     diag = weight_matrix.diagonal(dim1=1, dim2=2)
#     diag *= 0.0
#     enc_relational = torch.sum(weight_matrix * relational_matrix, dim=2)

#     # GT z_what are one-hot vectors 
#     z_what = torch.eye(N)[None].repeat(B, 1, 1).to(device)
#     actions = actions.unsqueeze(1).repeat(1, N, 1)

#     # The feature of one object include the following
#     # (B, N, N+4)
#     feat = torch.cat([z_where, z_what, actions], dim=-1) 
#     # (B, N, D)
#     enc_self = update_state_self(feat)

#     rnn_input = torch.cat([enc_self, enc_relational], dim=-1)
#     updated_state = temporal_rnn(states, rnn_input)
#     return updated_state


class PropNet(nn.Module):

    def __init__(self, node_dim_in, nf_hidden, node_dim_out, state_gnn, use_gpu=True):

        super(PropNet, self).__init__()

        self.node_dim_in = node_dim_in
        self.nf_hidden = nf_hidden
        self.node_dim_out = node_dim_out
        self.state_gnn = state_gnn

        # self-update
        modules = [
            nn.Linear(node_dim_in, nf_hidden),
            nn.ReLU(), 
            nn.BatchNorm1d(nf_hidden),
            nn.Linear(nf_hidden, node_dim_out)]
        self.update_state_self = BatchApply(nn.Sequential(*modules))

        # interaction value
        modules = [
            nn.Linear(node_dim_in * 2, nf_hidden),
            nn.ReLU(),
            nn.BatchNorm1d(nf_hidden),
            nn.Linear(nf_hidden, node_dim_out)]
        self.update_state_relational = BatchApply(nn.Sequential(*modules))

        # interaction weights
        modules = [
            nn.Linear(node_dim_in * 2, nf_hidden),
            nn.ReLU(),
            nn.BatchNorm1d(nf_hidden),
            nn.Linear(nf_hidden, 1)]
        self.update_state_weights = BatchApply(nn.Sequential(*modules))



    def forward(self, z, state):
        if self.state_gnn:
            return self.update_state(z, state)
        else:
            return self.update_state_z_where(z, state)  

  
    def update_state(self, z, state):
        """
        Object interaction in relational scalor
        Args:
            z: z_t
                z_pres: (B, N, 1)
                z_depth: (B, N, 1)
                z_where: (B, N, 4)
                z_what: (B, N, D)
            state: (B, N, D), h_{t-1}
        Returns:
            updated_state: (B, N, D), h'_{t-1}
        """
        update_state_self = self.update_state_self
        update_state_relational = self.update_state_relational
        update_state_weights = self.update_state_weights
        
        z_pres, z_where, z_what, z_depth = z
        
        def get_dim(z_x):
            _ , _, z_x_dim = z_x.size()
            return z_x_dim
        z_pres_dim, z_where_dim, z_what_dim, z_depth_dim = [get_dim(z_x) for z_x in z]
        z_where_shift_dim = 2
        z_where_scale_dim = 2


        B, N, _ = z_pres.size()
        # The feature of one object include the following
        # (B, N, D)
        feat = torch.cat(z + (state,), dim=-1)
        # (B, N, D)
        enc_self = update_state_self(feat)
        
        # Compute weights based on gaussian
        # (B, N, 2)
        z_shift_prop = z_where[:, :, 2:]
        # (B, N, 1, 2)
        z_shift_self = z_shift_prop[:, :, None]
        # (B, 1, N, 2)
        z_shift_other = z_shift_prop[:, None]
        # (B, N, N, 2)
        dist_matrix = z_shift_self - z_shift_other
        # (B, N, 1, D) -> (B, N, N, D)
        # feat_self = enc_self[:, :, None]
        feat_self = feat[:, :, None]
        feat_matrix_self = feat_self.expand(B, N, N, self.node_dim_in)
        # (B, 1, N, D) -> (B, N, N, D)
        feat_other = feat[:, None]
        feat_matrix_other = feat_other.expand(B, N, N, self.node_dim_in)
        # Replace absolute positions with relative ones
        # Must clone. Otherwise there will be multiple write
        feat_matrix_other = feat_matrix_other.clone()
        offset = z_pres_dim + z_where_scale_dim 
        feat_matrix_other[..., offset:offset + z_where_shift_dim] = dist_matrix
        feat_matrix = torch.cat([feat_matrix_self, feat_matrix_other], dim=-1)
        # (B, N, N, D)
        relational_matrix = update_state_relational(feat_matrix)
        zeros = torch.zeros_like(relational_matrix[:,:,0:1,:])
        # (B, N, N+1, D)
        relational_matrix = torch.cat([relational_matrix, zeros], dim=2)

        zeros = torch.zeros_like(feat_matrix[:,:,0:1,:])
        feat_matrix = torch.cat([feat_matrix, zeros], dim=2)
        # COMPUTE WEIGHTS
        # (B, N, N+1, 1)
        weight_matrix = update_state_weights(feat_matrix)
        # (B, N, >N, 1)
        weight_matrix = weight_matrix.softmax(dim=2)
        # Times z_pres (B, N, 1)-> (B, 1, N, 1)
        zeros = torch.zeros_like(z_pres[:,0:1,:])
        z_pres = torch.cat([z_pres, zeros], dim=1)
        weight_matrix = weight_matrix * z_pres[:, None]
        # Self mask, set diagonal elements to zero. (B, >N, >N, 1)
        # weights.diagonal: (B, 1, N)
        diag = weight_matrix.diagonal(dim1=1, dim2=2)
        diag *= 0.0
        # Renormalize (B, N, >N, 1)
        weight_matrix = weight_matrix / (weight_matrix.sum(dim=2, keepdim=True) + 1e-4)
        
        # (B, N1, N2, D) -> (B, N1, D)
        enc_relational = torch.sum(weight_matrix * relational_matrix, dim=2)
        
        # (B, N, D)
        updated_state = enc_self + enc_relational
        
        weight = torch.zeros(B, 5, 5, 1)
        weight[:, :N, :N, :] = weight_matrix[:,:,:N,:] 
        # (B, N, D), (B, MAXN, MAXN, 1)
        return updated_state, weight


    def update_state_z_where(self, z, state):
        """
        Object interaction in relational scalor
        Args:
            z: z_t
                z_pres: (B, N, 1)
                z_where: (B, N, 4)
                z_where_bias: (B, N, 4)
            state: (B, N, D), h_{t-1}
        Returns:
            updated_state: (B, N, D), h'_{t-1}
            updated_state: (B, N, D), h'_{t-1}
        """
        update_state_self = self.update_state_self
        update_state_relational = self.update_state_relational
        update_state_weights = self.update_state_weights
        
        z_pres, z_where, z_where_bias  = z
        
        def get_dim(z_x):
            _ , _, z_x_dim = z_x.size()
            return z_x_dim
        z_pres_dim, z_where_dim, _ = [get_dim(z_x) for z_x in z]
        z_where_shift_dim = 2
        z_where_scale_dim = 2


        B, N, _ = z_pres.size()
        # The feature of one object include the following
        # (B, N, D)
        feat = torch.cat(z + (state,), dim=-1)
        # (B, N, D)
        enc_self = update_state_self(feat)
        
        # Compute weights based on gaussian
        # (B, N, 2)
        z_shift_prop = z_where[:, :, 2:]
        # (B, N, 1, 2)
        z_shift_self = z_shift_prop[:, :, None]
        # (B, 1, N, 2)
        z_shift_other = z_shift_prop[:, None]
        # (B, N, N, 2)
        dist_matrix = z_shift_self - z_shift_other
        # (B, N, 1, D) -> (B, N, N, D)
        # feat_self = enc_self[:, :, None]
        feat_self = feat[:, :, None]
        feat_matrix_self = feat_self.expand(B, N, N, self.node_dim_in)
        # (B, 1, N, D) -> (B, N, N, D)
        feat_other = feat[:, None]
        feat_matrix_other = feat_other.expand(B, N, N, self.node_dim_in)
        # Replace absolute positions with relative ones
        # Must clone. Otherwise there will be multiple write
        feat_matrix_other = feat_matrix_other.clone()
        offset = z_pres_dim + z_where_scale_dim 
        feat_matrix_other[..., offset:offset + z_where_shift_dim] = dist_matrix
        feat_matrix = torch.cat([feat_matrix_self, feat_matrix_other], dim=-1)
        # (B, N, N, D)
        relational_matrix = update_state_relational(feat_matrix)
        zeros = torch.zeros_like(relational_matrix[:,:,0:1,:])
        # (B, N, N+1, D)
        relational_matrix = torch.cat([relational_matrix, zeros], dim=2)

        zeros = torch.zeros_like(feat_matrix[:,:,0:1,:])
        feat_matrix = torch.cat([feat_matrix, zeros], dim=2)
        # COMPUTE WEIGHTS
        # (B, N, N+1, 1)
        weight_matrix = update_state_weights(feat_matrix)
        # (B, N, >N, 1)
        weight_matrix = weight_matrix.softmax(dim=2)
        # Times z_pres (B, N, 1)-> (B, 1, N, 1)
        zeros = torch.zeros_like(z_pres[:,0:1,:])
        z_pres = torch.cat([z_pres, zeros], dim=1)
        weight_matrix = weight_matrix * z_pres[:, None]
        # Self mask, set diagonal elements to zero. (B, >N, >N, 1)

        diag = weight_matrix.diagonal(dim1=1, dim2=2)
        diag *= 0.0

        # Renormalize (B, N, >N, 1)
        weight_matrix = weight_matrix / (weight_matrix.sum(dim=2, keepdim=True) + 1e-4)
        
        # (B, N1, N2, D) -> (B, N1, D)
        enc_relational = torch.sum(weight_matrix * relational_matrix, dim=2)
        
        # (B, N, D)
        updated_state = enc_self + enc_relational
        
        weight = torch.zeros(B, 5, 5, 1)
        weight[:, :N, :N, :] = weight_matrix[:, :, :N, :] 
        # (B, N, D), (B, MAXN, MAXN, 1)
        return updated_state, weight
