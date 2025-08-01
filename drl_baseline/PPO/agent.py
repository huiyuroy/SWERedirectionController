from torch.distributions import Beta, Normal

from drl_baseline import *
from drl_baseline.PPO.baseline_cfg import *


class BetaActor(nn.Module):
    def __init__(self, state_dim, action_dim, net_width):
        super(BetaActor, self).__init__()

        self.l1 = nn.Linear(state_dim, net_width)
        self.l2 = nn.Linear(net_width, net_width)
        self.alpha_head = nn.Linear(net_width, action_dim)
        self.beta_head = nn.Linear(net_width, action_dim)

    def forward(self, state):
        a = torch.tanh(self.l1(state))
        a = torch.tanh(self.l2(a))

        alpha = F.softplus(self.alpha_head(a)) + 1.0
        beta = F.softplus(self.beta_head(a)) + 1.0

        return alpha, beta

    def get_dist(self, state):
        alpha, beta = self.forward(state)
        dist = Beta(alpha, beta)
        return dist

    def deterministic_act(self, state):
        alpha, beta = self.forward(state)
        mode = alpha / (alpha + beta)
        return mode


class GaussianActor_musigma(nn.Module):
    def __init__(self, state_dim, action_dim, net_width):
        super(GaussianActor_musigma, self).__init__()

        self.l1 = nn.Linear(state_dim, net_width)
        self.l2 = nn.Linear(net_width, net_width)
        self.mu_head = nn.Linear(net_width, action_dim)
        self.sigma_head = nn.Linear(net_width, action_dim)

    def forward(self, state):
        a = torch.tanh(self.l1(state))
        a = torch.tanh(self.l2(a))
        mu = torch.sigmoid(self.mu_head(a))
        sigma = F.softplus(self.sigma_head(a))  # >=0
        return mu, sigma

    def get_dist(self, state):
        mu, sigma = self.forward(state)
        dist = Normal(mu, sigma)
        return dist

    def deterministic_act(self, state):
        mu, sigma = self.forward(state)
        return mu


class GaussianActor_mu(nn.Module):
    def __init__(self, state_dim, action_dim, net_width, log_std=0):
        super().__init__()
        self.l1 = nn.Linear(state_dim, net_width)
        self.l2 = nn.Linear(net_width, net_width)
        self.mu_head = nn.Linear(net_width, action_dim)
        self.mu_head.weight.data.mul_(0.1)
        self.mu_head.bias.data.mul_(0.0)
        self.action_log_std = nn.Parameter(torch.zeros(1, action_dim) * log_std)

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
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
    def __init__(self, state_dim, net_width):
        super(Critic, self).__init__()

        self.C1 = nn.Linear(state_dim, net_width)
        self.C2 = nn.Linear(net_width, net_width)
        self.C3 = nn.Linear(net_width, 1)

    def forward(self, state):
        v = torch.tanh(self.C1(state))
        v = torch.tanh(self.C2(v))
        v = self.C3(v)
        return v


class PPOBuffer(ReplayMemory):
    def __init__(self,
                 max_size: int,
                 batch_size: int,
                 memory_info: dict,
                 use_latest,
                 device):
        super().__init__(max_size, batch_size, memory_info, use_latest, device)

    def sample(self):
        if self.mem_size < self.max_size:
            raise Exception('memory not fill')
        else:
            self.mem_size = 0
            return self.data


class PPOAgentBaseline(DRLAgent):
    def __init__(self,
                 agent_config=PPO_BASELINE_CONFIG,
                 memory_config=None,
                 train_config=None,
                 actor_distribution="GS_m"):
        """

        Args:
            agent_config:
            memory_config:
            train_config:
            actor_distribution: Beta, GS_m, GS_ms
        """
        super().__init__(agent_config=agent_config,
                         memory_config=memory_config,
                         train_config=train_config)

        self.k_epochs = self.agent_config['base']['k_epochs']
        self.gae_lambd = self.agent_config['base']['lambd']
        self.clip_rate = self.agent_config['base']['clip_rate']
        self.l2_reg = self.agent_config['base']['l2_reg']
        self.actor_lr = self.agent_config['base']['actor_lr']
        self.critic_lr = self.agent_config['base']['critic_lr']
        self.actor_mini_batch = self.agent_config['base']['actor_mini_batch']
        self.critic_mini_batch = self.agent_config['base']['critic_mini_batch']
        self.entropy_coef = self.agent_config['base']['entropy_coef']
        self.entropy_coef_decay = self.agent_config['base']['entropy_coef_decay']
        self.actor_distribution = actor_distribution

        self.state_dim = self.agent_config['env']['state_dim']
        self.action_dim = self.agent_config['env']['action_dim']

        self.replay_memory = PPOBuffer(max_size=self.agent_config['memory']['max_size'],
                                       batch_size=0,
                                       memory_info=self.memory_config,
                                       use_latest=False,
                                       device=self.device)
        self.replay_memory.reset_memory()
        self.actor = None
        self.actor_optimizer = None
        self.critic = None
        self.critic_optimizer = None

    def net_construct(self, **kwargs):
        net_width = kwargs['net_width']
        # Choose distribution for the actor
        if self.actor_distribution == 'Beta':
            self.actor = BetaActor(self.state_dim, self.action_dim, net_width).to(self.device)
        elif self.actor_distribution == 'GS_ms':
            self.actor = GaussianActor_musigma(self.state_dim, self.action_dim, net_width).to(self.device)
        elif self.actor_distribution == 'GS_m':
            self.actor = GaussianActor_mu(self.state_dim, self.action_dim, net_width).to(self.device)
        else:
            print('Dist Error')

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)

        # Build Critic
        self.critic = Critic(self.state_dim, net_width).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

    def choose_action(self, state, deterministic=False):
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            if deterministic:
                # only used when evaluate the policy. Making the performance more stable
                a = self.actor.deterministic_act(state)
                logprob_a = None
            else:
                # only used when interact with the env
                dist = self.actor.get_dist(state)
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
        # c_optim_iter_num = int(math.ceil(s.shape[0] / self.critic_mini_batch))
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
        return np.array(losses).mean()

    def save(self, path):
        torch.save(self.actor.state_dict(), path + 'actor.pth')
        torch.save(self.critic.state_dict(), path + 'critic.pth')

    def load(self, path):
        self.actor.load_state_dict(torch.load(path + 'actor.pth', map_location=self.device))
        self.critic.load_state_dict(torch.load(path + 'critic.pth', map_location=self.device))


class PPOTrainConfiger(Configer):

    def __init__(self):
        super().__init__()
        self.agent_config = copy.deepcopy(PPO_BASELINE_CONFIG)
        self.memory_config = copy.deepcopy(PPO_BASE_MEMORY_CONFIG)
        self.train_config = copy.deepcopy(PPO_BASE_TRAIN_CONFIG)

    def specify_config(self, env):
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        self.agent_config['env']['state_dim'] = state_dim
        self.agent_config['env']['action_dim'] = action_dim
        self.agent_config['memory']['max_size'] = 2048  # this is used to build ppo buffer
        self.agent_config['memory']['batch_size'] = 128

        # replay memory is meaningless for ppo
        self.memory_config['s'][0] = (state_dim,)
        self.memory_config['a'][0] = (action_dim,)
        self.memory_config['r'][0] = (1,)
        self.memory_config['s_'][0] = (state_dim,)
        self.memory_config['a_logprob'] = [(action_dim,), np.float32]
        self.memory_config['done'] = [(1,), np.bool_]
        self.memory_config['dw'][0] = (1,)

        self.train_config['max_epi_step'] = env._max_episode_steps
        self.train_config['max_train_step'] = int(5e7)
        self.train_config['save_step'] = int(5e5)
        self.train_config['eval_step'] = int(5e3)

        if self.train_config['enable_random_seed']:
            np.random.seed(self.train_config['random_seed'])
            torch.manual_seed(self.train_config['random_seed'])
            torch.cuda.manual_seed(self.train_config['random_seed'])
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        return self.agent_config, self.memory_config, self.train_config
