import os

import numpy as np
from einops import rearrange
import torch
from tensorboardX import SummaryWriter

from .utils import create_dir
from .gnn import PropGTNet


class DynNet(object):
    """
    A class for learning gnn dynamics in case when GT coordinates are avelable
    """

    def __init__(self, 
                 n_itr,
                 batch_size,
                 lr, 
                 data_dims, 
                 seq_size, 
                 force_teach_each,
                 recurent_dyn,
                 self_mask, 
                 no_interaction,
                 logdir="./results/",
                **kwargs):
 
        #atch params 
        self.seq_size = seq_size 
        self.force_teach_each = force_teach_each # if 1 then 1-step loss 
        self.node_dim = 20
        gnn_input_dim = data_dims["z_where_dim"] + data_dims["z_what_dim"] + data_dims["action_dim"]
        gnn_output_dim = data_dims["z_where_dim"] #only z_where as output
        gnn_hidden_dim = 100
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PropGTNet(gnn_input_dim, 
                               self.node_dim, 
                               gnn_hidden_dim, 
                               gnn_output_dim, 
                               recurent_dyn=recurent_dyn, 
                               self_mask=self_mask,  
                               no_interaction=no_interaction,
                               device=self.device)
        self.model.to(self.device)
        self.rnn_state = None
        
        # training params 
        self.n_itr = n_itr
        self.batch_size = batch_size
        self.lr = lr
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.lr)
        self.l2_loss = torch.nn.MSELoss(reduction='mean')
        self.alpha = 0.00001
        
        # visualization params  
        self.log_freq = 200
        self.save_epoch_freq = 10000
        self.eval_each = 10000
        self.global_step = 0

        self.summary_dir = os.path.join(logdir, "summary/events")     
        self.ckpt_dir = os.path.join(logdir, "checkpoints")
        create_dir(self.summary_dir)
        create_dir(self.ckpt_dir)
        self.writer = SummaryWriter(self.summary_dir)
    
    def train(self, states, actions, evaluate_data=None):
        self.model.train()
        for i in range(self.n_itr):
            states_batch, actions_batch = self._get_batch(states, actions, self.batch_size, self.seq_size)
            self._update_parameters(torch.Tensor(states_batch).to(self.device), torch.Tensor(actions_batch).to(self.device), i)
            if evaluate_data is not None and i % self.eval_each == 0:
                states_test, actions_test = evaluate_data
                self.evaluate(torch.Tensor(states_test).to(self.device), torch.Tensor(actions_test).to(self.device), global_step=i)

    def evaluate(self, z_wheres, actions, global_step):
        """evaluation on test data"""
        total_loss = 0.0
        total_loss_0 = 0.0
        B, T, Nd = z_wheres.size()
        N = int(Nd/2)
        objects_loss = N * [0.0]
        rnn_states = torch.zeros(B, N, self.node_dim).to(self.device)
        pred_z_wheres_all = torch.zeros(B, T-1, N * 2).to(self.device)
        weights = torch.zeros(B, T-1, N, N, 1)
        with torch.no_grad():
            for j in range(T-1):
                z_wheres_t = z_wheres[:, j, :].reshape(B, N, 2).detach()
                rnn_states, pred_z_wheres, weight, _ = self.model(rnn_states, z_wheres_t, actions[:, j+1, :].detach())
                pred_zeros = torch.zeros_like(pred_z_wheres)

                total_loss += self.l2_loss(pred_z_wheres,  (z_wheres[:, j+1, :] - z_wheres[:, j, :]).detach())
                total_loss_0 += self.l2_loss(pred_zeros,  (z_wheres[:, j+1, :] - z_wheres[:, j, :]).detach())
                for n, object_loss in enumerate(objects_loss):
                    objects_loss[n] += self.l2_loss(pred_z_wheres[..., (n * 2):((n+1) * 2)],  (z_wheres[:, j + 1, :] - z_wheres[:, j, :])[..., (n * 2):((n + 1) * 2)].detach())
                pred_z_wheres = z_wheres[:, j, :] + pred_z_wheres
                weights[:, j, :, :] = weight
                pred_z_wheres_all[:, j, :] = pred_z_wheres.clone()
            self.writer.add_scalar('test/total_loss', total_loss.item()/(T-1), global_step=global_step)
            self.writer.add_scalar('test/const_baseline', total_loss_0.item()/(T-1), global_step=global_step)
            for i, object_loss in enumerate(objects_loss):
                self.writer.add_scalar(f'test/object_{i}_loss', object_loss.item()/(T-1), global_step=global_step)
            print(total_loss.item()/(T-1), total_loss_0.item()/(T-1))
        return weights, pred_z_wheres_all, total_loss

    def _get_batch(self, states, actions, batch_size, seq_size):

        n_episodes = states.shape[0]
        rollout_length = states.shape[1]
        state_dim = states.shape[2]
        actions_dim = actions.shape[2]

        states = states.reshape(n_episodes * rollout_length, state_dim)
        actions = actions.reshape(n_episodes * rollout_length, actions_dim)
        idxs_ep = np.random.randint(0, n_episodes, size=batch_size)
        in_episode = np.random.randint(0, rollout_length - seq_size, size=batch_size) 
        idxs_start = idxs_ep * rollout_length + in_episode
        idxs_end = idxs_ep * rollout_length + in_episode + seq_size
        idxs = np.asarray(list(map(range, idxs_start, idxs_end)))
        return states[idxs] , actions[idxs]

    def _update_parameters(self, z_wheres, actions, global_step):
        """l2 reconsraction loss for gt dynamics"""
        total_loss = 0.0
        kl_loss_ = 0.0 
        self.global_step = global_step
        B, T, Nd = z_wheres.size()
        N = int(Nd/2)
        objects_loss = N * [0.0]
        rnn_states = torch.zeros(B, N, self.node_dim).to(self.device)
        weights = torch.zeros(B, self.seq_size-1, N, N, 1)
        
        next_z_where_pred = z_wheres[:, 0, :]
        for j in range(self.seq_size-1):
            
            if j % self.force_teach_each == 0:
                current_z_where = z_wheres[:, j, :]
            else:
                current_z_where = next_z_where_pred
            z_wheres_t = current_z_where.reshape(B, N, 2).detach()

            rnn_states, pred_z_wheres, weight, kl_loss = self.model(rnn_states, z_wheres_t, actions[:, j+1, :].detach())
            kl_loss_ += kl_loss
            delta_z_where = (z_wheres[:, j+1, :] - z_wheres[:, j, :]).detach()
            next_z_where_pred = current_z_where + pred_z_wheres


            total_loss += self.l2_loss(pred_z_wheres, delta_z_where) + self.alpha * kl_loss
            for n, object_loss in enumerate(objects_loss):
                objects_loss[n] += self.l2_loss(pred_z_wheres[..., (n * 2):((n+1) * 2)],  (z_wheres[:, j + 1, :] - z_wheres[:, j, :])[..., (n * 2):((n + 1) * 2)].detach())
            weights[:, j, :,:] = weight

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        if global_step % self.log_freq == 0:
            self.writer.add_scalar('train/kl_loss', kl_loss_.item(), global_step=global_step)
            self.writer.add_scalar('train/total_loss', total_loss.item()/(self.seq_size-1), global_step=global_step)
            for i, object_loss in enumerate(objects_loss):
                self.writer.add_scalar(f'train/object_{i}_loss', object_loss.item()/(self.seq_size-1), global_step=global_step)
        
        weights = weights[:3]
        weights = rearrange(weights, "b t h w d -> d (b h) (t w)")
        self.writer.add_image(f'attention_weights', weights, global_step)

        if global_step % self.save_epoch_freq == 0 or global_step == 1:
            ckpt_model_filename = f"ckpt_epoch_{global_step}.pth"
            path = os.path.join(self.ckpt_dir, ckpt_model_filename)
            self.save(path)

    def reset(self):
        self.rnn_state = None

    def encode(self, z_where, action):
        # TODO: NOT WORKING ON ENV data 
        if self.rnn_state is None:
            B, N, d = z_where.size()
            self.rnn_state = torch.zeros(B, N, self.node_dim).to(self.device)
        updated_state, pred_z_wheres, weights, _  = self.model(self.rnn_state, z_where, action)
        
        self.rnn_state = updated_state
        return weights, pred_z_wheres

    def to(self, device):
        self.model.to(device)

    def load(self, path):
        if os.path.isfile(path):
            print("=> loading checkpoint '{}'".format(path))
            checkpoint = torch.load(path, map_location=self.device)
            self.global_step = checkpoint['global_step']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['self.optimizer'])
            print("=> loaded checkpoint '{}' ".format(path))
        else:
            raise ValueError("No checkpoint!")

    def save(self, path):
        state = {
            'global_step': self.global_step,
            'state_dict': self.model.state_dict(),
            'self.optimizer': self.optimizer.state_dict(),
        }
        torch.save(state, path)
        print(f'{path:>2} has been successfully saved, global_step={self.global_step}')