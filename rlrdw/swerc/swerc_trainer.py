from pyrdw import generator
from pyrdw.util.file import ExcelWriter

from rlrdw import *
from rlrdw.swerc.swerc_benchmark import SWERCRDWAgent
from rlrdw import rlrdw_generator as r_gen
from rlrdw.training_scenes import abbr_name


class SWERCRdwEnv(DRLRDWEnv):

    def __init__(self):
        super().__init__()
        self.trajs = []
        self.entropy_atte = 1
        self.align_atte = 1

    def train(self):
        pname = "mess" #mess diff
        vname = 'complex2000x2000' #complex5000x5000

        save_name = f'{abbr_name[pname]}_{abbr_name[vname]}'

        excel_writter = ExcelWriter()
        excel_writter.open_excel(rl_model_path + '\\swerc', save_name + '_log_info')
        excel_writter.update_sheet('train info',
                                   ['Epi', 'Step', 'Epi Reward', 'Avg Reward', 'Avg Q Loss', 'Avg A Loss',
                                    'Avg Alpha Loss', 'P Name', 'V Name', 'Resets', 'Avg Resets', 'Walked Dis',
                                    'Avg Dis bt Resets'])
        self.add_agent(r_gen.obtain_agent(agent_manager='swerc',
                                          inputer='traj',
                                          entropy_atte=self.entropy_atte,
                                          align_atte=self.align_atte,
                                          swerc_reset_type='normal'),
                       name='swerc')


        self.load_ui('swerc train')
        self.load_logger()

        path = data_path + '\\rl\\phy\\{}.json'.format(pname)
        p_scene = r_gen.load_scene(path, scene_class='swerc')
        p_scene.update_grids_runtime_attr()
        p_scene.calc_swe_energy()

        path = data_path + '\\rl\\vir\\{}.json'.format(vname)
        v_scene = r_gen.load_scene(path, load_extend=True)
        v_scene.update_grids_runtime_attr()

        self.set_scenes(v_scene, p_scene)
        self.trajs = generator.load_trajectories(data_path + '\\rl\\vir', self.v_scene.name)

        self.prepare()
        self.env_ui.render_mode(True)

        ag: SWERCRDWAgent = self.agents['swerc']
        agent_cfg, memory_cfg, train_cfg = ag.config()
        max_train_step = train_cfg['max_train_step']
        warmup_epi = 0
        learn_step = agent_cfg['memory']['max_size']
        save_step = train_cfg['save_step']
        eval_epi = train_cfg['eval_epi']
        total_steps = 0
        total_epi = 0
        total_rwd = 0
        total_reset = 0

        while total_steps < max_train_step:
            s_time = time.perf_counter()

            traj = self.trajs[np.random.randint(0, len(self.trajs))]
            traj.range_targets(start=np.random.randint(0, int(traj.tar_num / 3)))
            self.set_current_trajectory(traj)
            self.init_agents_state(p_loc=self.obtain_rand_p_initloc(), p_fwd=[1, 0], v_loc=[0, 1], v_fwd=[1, 0])
            done = False
            total_epi += 1
            epi_r = 0
            epi_q_loss = []
            epi_a_loss = []
            epi_alpha_loss = []
            epi_learn_step = 0
            r_max = 0
            r_min = float('inf')
            self.reset()
            while not done:
                s, a, a_logprob, r, s_, dw, tr, _ = ag.step_non_deterministic()
                self.render()
                self.record()
                done = (dw or tr)
                if r > 0:
                    if r < r_min:
                        r_min = r
                    if r > r_max:
                        r_max = r
                if s is not None:  # not in reset
                    ag.store_step_data(s=s,
                                       a=a,
                                       a_logprob=a_logprob,
                                       r=r,
                                       s_=s_,
                                       done=done,
                                       dw=dw)
                    total_steps += 1
                    epi_r += r

                    if total_epi >= warmup_epi and (total_steps % learn_step == 0):  # 预热若干个回合后开始训练
                        q_loss, a_loss, alpha_loss = ag.learn()
                        epi_q_loss.append(q_loss)
                        epi_a_loss.append(a_loss)
                        epi_alpha_loss.append(alpha_loss)
                        epi_learn_step += 1

                    if total_steps % save_step == 0:
                        ag.save(save_name + f'_{int(total_steps / 1000)}k')

            e_time = time.perf_counter()

            total_s = total_steps * 0.001
            total_rwd += epi_r
            avg_rwd = total_rwd / total_epi
            avg_q_loss = np.array(epi_q_loss).mean() if len(epi_q_loss) > 0 else 0
            avg_a_loss = np.array(epi_a_loss).mean() if len(epi_q_loss) > 0 else 0
            avg_alpha_loss = np.array(epi_alpha_loss).mean() if len(epi_q_loss) > 0 else 0
            walk_dis = ag.walked_dis * 0.01

            print("Epi:{}, Step:{:.3f}k, Epi Reward：{:.2f}, Avg Epi Reward:{:.2f}, "
                  "Avg loss: q {:.2f}, P:{}, walk dis {:.1f},"
                  "time cost:{:.2f}".format(total_epi,
                                            total_s,
                                            epi_r,
                                            avg_rwd,
                                            avg_q_loss,
                                            self.p_scene.name,
                                            ag.walked_dis * 0.01,
                                            e_time - s_time), end=' ')
            data = self.env_logs['swerc'].log_epi_data()
            reset = data['reset_num']
            total_reset += reset
            avg_reset = total_reset / total_epi
            avg_dis = walk_dis / reset
            print(' resets: {reset_num}/{avg_reset:.2f} dis bet reset: {dis:.2f}'.format(**data,
                                                                                         avg_reset=avg_reset,
                                                                                         dis=avg_dis))
            excel_writter.write_excel(total_epi, total_s, epi_r, avg_rwd,
                                      avg_q_loss, avg_a_loss, avg_alpha_loss,
                                      self.p_scene.name, self.v_scene.name,
                                      reset, avg_reset, walk_dis, avg_dis)
            excel_writter.save_excel()

            if total_epi % eval_epi == 0:
                self.evaluate()

    def evaluate(self, eval_turns=2):
        print("Evaluate: ")
        print('pscene:{}'.format(self.p_scene.name))

        for _ in range(eval_turns):
            traj = self.trajs[np.random.randint(0, len(self.trajs))]
            traj.range_targets(start=0, end=500)
            self.set_current_trajectory(traj)
            self.init_agents_state(p_loc=self.obtain_rand_p_initloc(), p_fwd=[1, 0], v_loc=[0, 1], v_fwd=[1, 0])

            done = False
            self.reset()
            while not done:
                done = self.step()
                self.render()
                self.record()
                if done:
                    all_data = self.output_epi_info()
                    for d in all_data:
                        print('{alg_name}-resets: {reset_num}'.format(**d), end=', ')
                    print()


if __name__ == '__main__':
    env = SWERCRdwEnv()
    env.train()
