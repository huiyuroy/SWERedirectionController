from torch.distributions import Normal

from drl_baseline.PPO.agent import PPOAgentBaseline
from rlrdw import *
from rlrdw.swerc.config import SWERCPPO_CONFIG, SWERCPPO_MEMORY_CONFIG, SWERCPPO_TRAIN_CONFIG, swerc_state_seq


class VecCommonHead(nn.Module):
    def __init__(self, state_dim, seq_dim, device='cpu'):
        super().__init__()
        self.seq_dim = seq_dim
        self.state_dim = state_dim
        l_out_dim = 256
        self.linear_in = nn.Sequential(
            nn.Linear(state_dim, l_out_dim),
            # nn.LayerNorm(l_out_dim),
            nn.ReLU()
            # nn.Linear(256, l_out_dim),
            # nn.ReLU()
        )
        self.gru_out_dim = l_out_dim
        self.gru = nn.GRU(input_size=l_out_dim,
                          hidden_size=self.gru_out_dim,
                          num_layers=1,
                          bias=False,
                          batch_first=True)

        self.h: torch.Tensor = torch.zeros((1, 1, self.gru_out_dim), device=device)  # (num_layers, batch, hidden_size)
        self.first = True
        self.device = device
        self.out_dim = self.gru_out_dim * seq_dim

    def init_gru_state(self, batch_size):
        self.h = torch.zeros((1, batch_size, self.gru_out_dim), device=self.device)
        self.first = False

    def forward(self, s):
        x = self.linear_in(s)
        if self.first:
            b_size = s.shape[0]
            self.init_gru_state(b_size)
            x, h = self.gru(x, self.h)
        else:
            x, h = self.gru(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(x)
        return x


class GaussianActor_mu(nn.Module):
    def __init__(self, head, action_dim, net_width, log_std=0):
        super().__init__()
        self.head: VecCommonHead = head
        self.state_dim = self.head.out_dim
        self.net = nn.Sequential(
            nn.Linear(self.state_dim, net_width),
            nn.Tanh(),
            nn.Linear(net_width, net_width),
            nn.Tanh(),
        )

        self.mu_head = nn.Linear(net_width, action_dim)
        self.mu_head.weight.data.mul_(0.1)
        self.mu_head.bias.data.mul_(0.0)
        self.action_log_std = nn.Parameter(torch.zeros(1, action_dim) * log_std)

    def forward(self, state):
        x = self.head(state)
        a = self.net(x)
        mu = F.tanh(self.mu_head(a))
        return mu

    def get_dist(self, state):
        mu = self.forward(state)
        action_log_std = self.action_log_std.expand_as(mu)
        std = torch.exp(action_log_std)

        dist = Normal(mu, std)
        return dist

    def deterministic_act(self, state):
        a = self.forward(state)
        return a


class Critic(nn.Module):
    def __init__(self, head, net_width):
        super(Critic, self).__init__()
        self.head: VecCommonHead = head
        self.state_dim = self.head.out_dim
        self.net = nn.Sequential(
            nn.Linear(self.state_dim, net_width),
            nn.Tanh(),
            nn.Linear(net_width, net_width),
            nn.Tanh(),
            nn.Linear(net_width, 1)
        )

    def forward(self, state):
        x = self.head(state)
        v = self.net(x)
        return v


class SWERCPPOAlg(PPOAgentBaseline):
    def __init__(self, agent_config=SWERCPPO_CONFIG,
                 memory_config=SWERCPPO_MEMORY_CONFIG,
                 train_config=SWERCPPO_TRAIN_CONFIG):
        super().__init__(agent_config=agent_config,
                         memory_config=memory_config,
                         train_config=train_config)
        self.im_state_dim = list(self.agent_config['env']['img_state_dim'])
        self.seq_dim = swerc_state_seq
        self.head: VecCommonHead = VecCommonHead(self.state_dim, self.seq_dim, self.device).to(self.device)
        self.actor: GaussianActor_mu = None
        self.critic: Critic = None

    def net_construct(self, **kwargs):
        net_width = kwargs['net_width']
        self.actor = GaussianActor_mu(self.head,
                                      action_dim=self.action_dim,
                                      net_width=net_width).to(self.device)
        # actor_params = list(self.actor.parameters()) + list(self.head.parameters())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)

        self.critic = Critic(self.head, net_width=net_width).to(self.device)
        q_critic_params = [param for name, param in self.critic.named_parameters() if "head" not in name]
        self.critic_optimizer = torch.optim.Adam(q_critic_params, lr=self.critic_lr)

    def choose_action(self, state, deterministic=False):
        # only used when interact with the env
        # s, im_s = state
        with torch.no_grad():
            s = torch.unsqueeze(torch.FloatTensor(state), dim=0).to(self.device)  # from [x,x,...,x] to [[x,x,...,x]]
            if deterministic:
                # only used when evaluate the policy. Making the performance more stable
                a = self.actor.deterministic_act(s)
                logprob_a = None
            else:
                # only used when interact with the env
                dist = self.actor.get_dist(s)
                a = dist.sample()
                a = torch.clamp(a, -1, 1)
                logprob_a = dist.log_prob(a).cpu().numpy().flatten()
            return a.cpu().numpy()[0], logprob_a  # both are in shape (adim, 0), a is within [-1,1]

    def learn(self):
        self.entropy_coef *= self.entropy_coef_decay
        '''Prepare PyTorch data from Numpy data'''
        batch = self.replay_memory.sample()

        s = torch.tensor(batch['s']).to(self.device)
        a = torch.tensor(batch['a']).to(self.device)
        a_logprob = torch.tensor(batch['a_logprob']).to(self.device)
        r = torch.tensor(batch['r']).to(self.device)
        s_ = torch.tensor(batch['s_']).to(self.device)
        done = torch.tensor(batch['done']).to(self.device)
        dw = torch.tensor(batch['dw']).to(self.device)

        ''' Use TD+GAE+LongTrajectory to compute Advantage and TD target'''
        with torch.no_grad():
            vs = self.critic(s)
            vs_ = self.critic(s_)

            '''dw for TD_target and Adv'''
            deltas = r + self.gamma * vs_ * (~dw) - vs
            deltas = deltas.cpu().flatten().numpy()
            adv = [0]

            '''done for GAE'''
            for dlt, mask in zip(deltas[::-1], done.cpu().flatten().numpy()[::-1]):
                advantage = dlt + self.gamma * self.gae_lambd * adv[-1] * (~mask)
                adv.append(advantage)
            adv.reverse()
            adv = copy.deepcopy(adv[0:-1])
            adv = torch.tensor(adv).unsqueeze(1).float().to(self.device)
            td_target = adv + vs
            adv = (adv - adv.mean()) / (adv.std() + 1e-4)  # sometimes helps

        """Slice long trajectopy into short trajectory and perform mini-batch PPO update"""
        a_optim_iter_num = int(math.ceil(s.shape[0] / self.actor_mini_batch))
        losses = []
        for i in range(self.k_epochs):
            # Shuffle the trajectory, Good for training
            perm = np.arange(s.shape[0])
            np.random.shuffle(perm)
            perm = torch.LongTensor(perm).to(self.device)
            s, a, td_target, adv, a_logprob = \
                s[perm].clone(), a[perm].clone(), td_target[perm].clone(), adv[perm].clone(), a_logprob[perm].clone()
            '''update the actor'''
            for i in range(a_optim_iter_num):
                index = slice(i * self.actor_mini_batch, min((i + 1) * self.actor_mini_batch, s.shape[0]))
                distribution = self.actor.get_dist(s[index])
                dist_entropy = distribution.entropy().sum(1, keepdim=True)
                logprob_a_now = distribution.log_prob(a[index])
                # a/b == exp(log(a)-log(b))
                ratio = torch.exp(logprob_a_now.sum(1, keepdim=True) - a_logprob[index].sum(1, keepdim=True))
                surr1 = ratio * adv[index]
                surr2 = torch.clamp(ratio, 1 - self.clip_rate, 1 + self.clip_rate) * adv[index]
                a_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy
                c_loss = F.mse_loss(self.critic(s[index]), td_target[index])
                t_loss = a_loss + c_loss * 0.5
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                t_loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 40)
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                losses.append(t_loss.detach().cpu().numpy())
        return np.array(losses).mean(), 0, 0
