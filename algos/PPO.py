import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import MultivariateNormal


# Reference implementations:
# https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO_continuous.py
# https://github.com/seungeunrho/minimalRL/blob/master/ppo-lstm.py


class ActorCritic(nn.Module):
    def __init__(
        self, lidar_state_dim, position_state_dim, lidar_feature_dim, action_dim, hidden_dim, max_action,
        policy_noise, is_recurrent=True
    ):
        super(ActorCritic, self).__init__()
        self.recurrent = is_recurrent
        self.action_dim = action_dim

        self.lidar_compress_net = nn.Sequential(
            nn.Linear(lidar_state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, lidar_feature_dim, nn.ReLU())
        )

        if self.recurrent:
            self.l1 = nn.LSTM(lidar_feature_dim, hidden_dim, batch_first=True)
        else:
            self.l1 = nn.Linear(lidar_feature_dim + position_state_dim, hidden_dim)

        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

        self.max_action = max_action
        self.policy_noise = policy_noise

    def forward(self, lidar_state, position_state):
        with torch.no_grad():
            lidar_feature = self.lidar_compress_net(lidar_state)  #1800 -》50
        state = torch.cat((lidar_feature, position_state), dim=-1)

        p = torch.tanh(self.l1(state))

        p = torch.tanh(self.l2(p.data))
        return p

    def act(self, lidar_state, position_state):
        p = self.forward(lidar_state, position_state)
        action = torch.tanh(self.actor(p))

        return action * self.max_action

    def evaluate(self, lidar_state, position_state, action):
        p = self.forward(lidar_state, position_state)
        action_mean, _ = self.act(lidar_state, position_state)

        cov_mat = torch.eye(self.action_dim).to(self.device) * self.policy_noise

        dist = MultivariateNormal(action_mean, cov_mat)
        _ = dist.sample()
        action_logprob = dist.log_prob(action)
        entropy = dist.entropy()
        values = self.critic(p)

        if self.recurrent:
            values = values[..., 0]
        else:
            action_logprob = action_logprob[..., None]

        return values, action_logprob, entropy


class PPO(object):
    def __init__(
        self,
        lidar_state_dim,
        position_state_dim,
        lidar_feature_dim,
        action_dim,
        max_action,
        hidden_dim,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        eps_clip=.2,
        lmbda=0.95,
        lr=3e-4,
        K_epochs=80,
        recurrent_actor=False,
        recurrent_critic=False,
        device = 'cpu'
    ):
        self.device = device
        self.on_policy = True
        self.recurrent = recurrent_actor
        self.actorcritic = ActorCritic(
            lidar_state_dim, position_state_dim, lidar_feature_dim, action_dim, hidden_dim, max_action, policy_noise,
            is_recurrent=recurrent_actor
        ).to(self.device)
        self.target = copy.deepcopy(self.actorcritic)
        self.optimizer = torch.optim.Adam(self.target.parameters())

        self.discount = discount
        self.lmbda = lmbda
        self.tau = tau
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.actor_loss_coeff = 1.
        self.critic_loss_coeff = 0.74
        self.entropy_loss_coeff = 0.01

    def get_initial_states(self):
        h_0, c_0 = None, None
        if self.actorcritic.recurrent:
            h_0 = torch.zeros((
                self.actorcritic.l1.num_layers,
                1,
                self.actorcritic.l1.hidden_size),
                dtype=torch.float)
            h_0 = h_0.to(device=self.device)

            c_0 = torch.zeros((
                self.actorcritic.l1.num_layers,
                1,
                self.actorcritic.l1.hidden_size),
                dtype=torch.float)
            c_0 = c_0.to(device=self.device)
        return (h_0, c_0)

    def select_action(self, lidar_state, position_state):
        lidar_state = torch.FloatTensor(
            lidar_state.reshape(1, -1)).to(self.device)[:, None, :]
        position_state = torch.FloatTensor(
            position_state.reshape(1, -1)).to(self.device)[:, None, :]

        action = self.actorcritic.act(lidar_state, position_state)
        return action.cpu().data.numpy().flatten()

    def train(self, replay_buffer):

        # Sample replay buffer
        lidar_state, position_state, action, next_lidar_state, next_position_state, reward, not_done = \
            replay_buffer.on_policy_sample()

        running_actor_loss = 0
        running_critic_loss = 0

        discounted_reward = 0
        rewards = []

        for r, is_terminal in zip(reversed(reward), reversed(1 - not_done)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = r + (self.discount * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        rewards = rewards[..., None]

        # log_prob of pi(a|s)
        _, prob_a, _ = self.actorcritic.evaluate(
            lidar_state, position_state, action)

        # TODO: PPO Update
        # PPO allows for multiple gradient steps on the same data
        for _ in range(self.K_epochs):

            # V_pi'(s) and pi'(a|s)
            v_s, logprob, dist_entropy = self.target.evaluate(
                lidar_state, position_state, action)

            assert rewards.size() == v_s.size(), \
                '{}, {}'.format(rewards.size(), v_s.size())
            # Finding Surrogate Loss:
            advantages = rewards - v_s

            # Ratio between probabilities of action according to policy and
            # target policies

            assert logprob.size() == prob_a.size(), \
                '{}, {}'.format(logprob.size(), prob_a.size())
            ratio = torch.exp(logprob - prob_a)

            # Surrogate policy loss
            assert ratio.size() == advantages.size(), \
                '{}, {}'.format(ratio.size(), advantages.size())

            surrogate_policy_loss_1 = ratio * advantages
            surrogate_policy_loss_2 = torch.clamp(
                ratio,
                1-self.eps_clip,
                1+self.eps_clip) * advantages
            # PPO "pessimistic" policy loss
            actor_loss = -torch.min(
                surrogate_policy_loss_1,
                surrogate_policy_loss_2)

            # Surrogate critic loss: MSE between "true" rewards and prediction
            # TODO: Investigate size mismatch
            assert(v_s.size() == rewards.size())

            surrogate_critic_loss_1 = F.mse_loss(
                v_s,
                rewards)
            surrogate_critic_loss_2 = torch.clamp(
                surrogate_critic_loss_1,
                -self.eps_clip,
                self.eps_clip
            )
            # PPO "pessimistic" critic loss
            critic_loss = torch.max(
                surrogate_critic_loss_1,
                surrogate_critic_loss_2)

            # Entropy "loss" to promote entropy in the policy
            entropy_loss = dist_entropy[..., None].mean()

            # Gradient step
            self.optimizer.zero_grad()
            loss = ((critic_loss * self.critic_loss_coeff) +
                    (self.actor_loss_coeff * actor_loss) -
                    (entropy_loss * self.entropy_loss_coeff))
            # print(loss.size(), loss)
            loss.mean().backward(retain_graph=True)
            # print([p.grad for p in self.target.parameters()])
            nn.utils.clip_grad_norm_(self.target.parameters(),
                                     0.5)
            self.optimizer.step()

            # Keep track of losses
            running_actor_loss += actor_loss.mean().cpu().detach().numpy()
            running_critic_loss += critic_loss.mean().cpu().detach().numpy()

        self.actorcritic.load_state_dict(self.target.state_dict())
        torch.cuda.empty_cache()

    def save(self, filename):
        torch.save(self.actorcritic.state_dict(), filename)
        torch.save(self.optimizer.state_dict(),
                   filename + "_optimizer")

    def load(self, filename):
        self.actorcritic.load_state_dict(torch.load(filename))
        self.optimizer.load_state_dict(
            torch.load(filename + "_optimizer"))

    def eval_mode(self):
        self.actorcritic.eval()

    def train_mode(self):
        self.actorcritic.train()
