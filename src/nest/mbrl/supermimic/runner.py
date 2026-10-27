"""Motion Force Adapter V3 supervised WM fitting and imagined BPTT updates.

The optimization equations/defaults are unchanged. Native environment, sensor,
actuator-memory and checkpoint plumbing are adapted to NEST.
"""
from dataclasses import asdict
from pathlib import Path
import copy
import json
import time
import torch
from torch.nn import functional as F
from .utils import frozen, lambda_returns
from .model import make_policy, policy_variant, WorldModel, set_precision
from .reference import networks as ref
from .replay import GPUReplay
from .adapter import Adapter


class ReferenceRunner:
    def __init__(self, env, cfg, run_dir, iterations=5000, save_every=1000, task_spec=None):
        self.env, self.cfg, self.run_dir = env.unwrapped, cfg, Path(run_dir)
        env = self.env
        self.iterations, self.save_every = iterations, save_every
        # Preserve the source's first-run clock; save its rate so extending a
        # resumed run cannot move the command curriculum backwards.
        self.curriculum_rate = iterations * 24 / max(iterations-cfg.model_only_iterations,1)
        self.adapter = Adapter(self.env)
        self.layout, self.mdp, self.context = self.adapter.layout, self.adapter.mdp, self.adapter.context
        L = self.layout
        set_precision(cfg.amp)
        self.variant = policy_variant(cfg.policy_profile)
        self.actor = make_policy(L.obs_dim, L.actions, cfg.policy_profile).to(env.device)
        self.history_length = self.actor.hist
        self.critic = ref.Critic(L.physical+self.context.width+L.delay_size+9).to(env.device)
        self.model = WorldModel(L, self.context.width, self.adapter.default, self.adapter.scale,
                                self.adapter.mg, env.step_dt, cfg).to(env.device)
        self.actor.eval()
        self.architecture = dict(variant=self.variant, history=self.history_length, layout=asdict(L),
            actor=dict(obs_dim=L.obs_dim, goal_dim=3, encoder=[2048,1024,512,256],
                       decoder=[2048,1024,512,256], latent=32),
            physical_hidden=list(cfg.model_widths), auxiliary_hidden=list(cfg.auxiliary_widths),
            critic_hidden=[256,128], context_fields=self.context.fields,
            joint_names=list(self.adapter.robot.joint_names), body_names=list(self.adapter.robot.body_names),
            action_joint_names=[self.adapter.robot.joint_names[i] for i in self.adapter.action_ids],
            task_spec=task_spec)
        if cfg.policy_profile == 'small_mlp':
            self.architecture['actor'] = dict(obs_dim=L.obs_dim, goal_dim=3,
                hidden=[512,256,128], input_dim=self.history_length*L.obs_dim+3,
                activation='ELU', packing='five complete NEST frames; current goal only')
        self.replay = GPUReplay(cfg.replay_steps, env.num_envs, self.context.width, env.device, L)
        self.opt_model = torch.optim.Adam(self.model.physical.parameters(), lr=cfg.learning_rate)
        self.opt_auxiliary = torch.optim.Adam(self.model.auxiliary.parameters(), lr=cfg.learning_rate)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=cfg.learning_rate, betas=(.9,.95))
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=cfg.learning_rate)
        self.start_iteration = self.policy_updates = self.sim_transitions = 0
        self.training_seconds = self.elapsed_before_resume = 0.
        self.wm_rollout, self.wm_error = self._wm_rollout, self._wm_error
        self.imag_step, self.reward_done, self.critic_input = self._imag_step, self.mdp.reward_and_done, self.mdp.critic_input
        if cfg.compile:
            self.wm_rollout = torch.compile(self._wm_rollout)
            self.wm_error = torch.compile(self._wm_error)
            self.imag_step = torch.compile(self._imag_step)
            self.reward_done = torch.compile(self.mdp.reward_and_done)
            self.critic_input = torch.compile(self.mdp.critic_input)
        frame = self.env.observation_manager.compute()['policy'][..., :L.obs_dim]
        self.history = frame[:,None].repeat(1,self.history_length,1)

    def _imag_step(self, s, hist, ap, c, g):
        a = self.actor(hist,g[...,:3])
        nxt,_,landing = self.model(s,a,c)
        gn = self.mdp.advance_goal(nxt,g)
        obs = self.mdp.add_noise(self.mdp.obs(nxt,a,c,gn))
        return nxt,torch.cat((hist[:,1:],obs[:,None]),1),a,gn,landing

    def _wm_rollout(self, s, a, c, lags):
        current=s[:,0]
        states,contacts,landings=[],[],[]
        for i in range(self.cfg.window):
            current,ct,ld=self.model(current,a[:,i],c[:,i],detach_physics_for_auxiliary=True,lags=lags[:,i])
            states.append(current);contacts.append(ct);landings.append(ld)
        return torch.stack(states,1),torch.stack(contacts,1),torch.stack(landings,1)

    def _wm_error(self, predictions, contacts, landings, target, landing_target, mask):
        physical=self.model.physical.error(predictions,target).square().mean(-1)
        auxiliary=((predictions[...,self.layout.force]-target[...,self.layout.force])/self.adapter.mg).square().mean(-1)
        auxiliary=auxiliary+((predictions[...,self.layout.peak]-target[...,self.layout.peak])/.1).square().mean(-1)
        auxiliary=auxiliary+((predictions[...,self.layout.age]-target[...,self.layout.age])/.1).square().mean(-1)
        auxiliary=auxiliary+F.binary_cross_entropy_with_logits(contacts,target[...,self.layout.contact],reduction='none').mean(-1)
        auxiliary=auxiliary+F.binary_cross_entropy_with_logits(landings,landing_target.float(),reduction='none').mean(-1)
        lp=(physical*mask).sum()/len(target)
        la=(auxiliary*mask).sum()/len(target)
        e8=(physical[:,-1]*mask[:,-1]).sum()/mask[:,-1].sum().clamp_min(1)
        return lp,la,e8

    def fit_model(self):
        losses,auxiliary_losses,errors=[],[],[]
        for group in self.opt_auxiliary.param_groups:
            group['lr']=self.opt_model.param_groups[0]['lr']
        for _ in range(self.cfg.model_batches):
            b=self.replay.windows(self.cfg.model_batch_size,self.cfg.window)
            prediction,contacts,landing=self.wm_rollout(b['s'],b['a'],b['c'],b['lags'])
            lp,la,e8=self.wm_error(prediction,contacts,landing,b['s1'],b['landing'],b['mask'])
            self.opt_model.zero_grad(set_to_none=True)
            self.opt_auxiliary.zero_grad(set_to_none=True)
            (lp+la).backward()
            torch.nn.utils.clip_grad_norm_(self.model.physical.parameters(),1.,error_if_nonfinite=True)
            torch.nn.utils.clip_grad_norm_(self.model.auxiliary.parameters(),1.,error_if_nonfinite=True)
            self.opt_model.step();self.opt_auxiliary.step()
            losses.append(lp.detach()/self.cfg.window)
            auxiliary_losses.append(la.detach()/self.cfg.window)
            errors.append(e8.detach())
        return dict(model_loss=float(torch.stack(losses).mean()),auxiliary_loss=float(torch.stack(auxiliary_losses).mean()),
                    model_error8=float(torch.stack(errors).mean()))

    def improve(self):
        start,h=self.replay.starts(self.cfg.imagination_batch_size,self.history_length)
        s,c,g,ap=start['s'],start['c'],start['g'],start['ap']
        with torch.no_grad():
            hist=self.mdp.add_noise(self.mdp.obs(h['s'],h['ap'],h['c'],h['g']))
        states,actions,previous_actions,goals,peaks,landing=[],[],[],[],[],[]
        initial_s,initial_g=s,g
        with frozen(self.model,self.critic):
            for _ in range(self.cfg.horizon):
                previous_actions.append(ap);goals.append(g);peaks.append(s[:,self.layout.peak])
                s,hist,ap,g,ld=self.imag_step(s,hist,ap,c,g)
                states.append(s);actions.append(ap);landing.append(ld)
            flatten=lambda xs:torch.stack(xs).flatten(0,1)
            cc=c[None].expand(self.cfg.horizon,-1,-1).flatten(0,1)
            reward,done=self.reward_done(flatten(states),flatten(actions),flatten(previous_actions),
                                         flatten(peaks),cc,flatten(goals),flatten(landing))
            reward,done=reward.reshape(self.cfg.horizon,-1),done.reshape(self.cfg.horizon,-1)
            alive=torch.cumprod(1-done,0)
            before=torch.cat((torch.ones_like(alive[:1]),alive[:-1]),0)
            discount=self.cfg.gamma**torch.arange(self.cfg.horizon,device=s.device,dtype=torch.float32)[:,None]
            objective=(discount*before*reward).sum(0)
            objective=objective+self.cfg.gamma**self.cfg.horizon*alive[-1]*self.critic(self.mdp.critic_input(s,c,g))
            loss=-objective.mean()
            self.opt_actor.zero_grad(set_to_none=True)
            loss.backward()
            grad=torch.nn.utils.clip_grad_norm_(self.actor.parameters(),1.,error_if_nonfinite=True)
            torch.nn.utils.clip_grad_value_(self.actor.parameters(),.1)
            self.opt_actor.step()
        with torch.no_grad():
            ss=torch.cat((initial_s[None],torch.stack([s.detach() for s in states])),0)
            gs=torch.cat((initial_g[None],torch.stack([self.mdp.advance_goal(st.detach(),go.detach())
                                                      for st,go in zip(states,goals)])),0)
            cx=c[None].expand(self.cfg.horizon+1,-1,-1)
            X=self.critic_input(ss.flatten(0,1),cx.flatten(0,1),gs.flatten(0,1)).reshape(self.cfg.horizon+1,len(c),-1)
            self.critic.norm.update(X)
            values=self.critic(X)
            targets=lambda_returns(reward.detach(),done.detach(),values,self.cfg.gamma,self.cfg.lam).flatten()
            X=X[:-1].flatten(0,1)
        losses=[]
        for _ in range(self.cfg.critic_epochs):
            for ids in torch.randperm(len(X),device=X.device).tensor_split(self.cfg.critic_minibatches):
                loss_v=(self.critic(X[ids])-targets[ids]).square().mean()
                self.opt_critic.zero_grad(set_to_none=True)
                loss_v.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(),1.,error_if_nonfinite=True)
                self.opt_critic.step();losses.append(loss_v.detach())
        self.policy_updates+=1
        return dict(actor_loss=float(loss.detach()),actor_grad_norm=float(grad),critic_loss=float(torch.stack(losses).mean()),
                    imagined_reward=float((reward.detach()*before.detach()).mean()),imagined_survival=float(alive[-1].detach().mean()))

    @torch.no_grad()
    def collect(self, steps):
        rewards, falls = [], 0
        for _ in range(steps):
            s, c, g = self.adapter.state().clone(), self.context.capture(), self.adapter.goal().clone()
            ap = self.env.action_manager.action.clone()
            ep = self.env.episode_length_buf.clone()
            self.actor.obs_norm.update(self.history[:, -1])
            self.actor.goal_norm.update(g[:, :3])
            a = (self.actor(self.history, g[:, :3])+self.cfg.action_noise*torch.randn_like(ap)).clamp(-20,20)
            self.adapter.begin_step()
            observation, reward, term, timeout, _ = self.env.step(a)
            done = term | timeout
            target = self.adapter.target
            lags = self.adapter.executed_lags()
            following = self.adapter.state()
            # Interval pushes happen after reward/terminal capture. They are
            # exogenous state jumps, not transitions for the action dynamics.
            cut = (following[:, :self.layout.physical]-target[:, :self.layout.physical]).abs().amax(-1) > 1e-4
            if not all(torch.isfinite(x).all() for x in (s, c, g, target, reward)):
                raise FloatingPointError('Nonfinite NEST transition')
            self.replay.add(dict(s=s, s1=target, a=a, ap=ap, c=c, g=g, ep=ep, r=reward,
                                 done=done, term=term, landing=self.adapter.landing > .5, lags=lags, cut=cut))
            self.model.update_normalization(s, target, a, c, lags)
            frame = observation['policy'][..., :self.layout.obs_dim]
            self.history = torch.cat((self.history[:,1:], frame[:,None]),1)
            self.history[done] = frame[done,None].expand(-1,self.history_length,-1)
            self.sim_transitions += self.env.num_envs
            rewards.append(reward.mean()); falls += int(term.sum())
        return float(torch.stack(rewards).mean()), falls

    def sync(self):
        if str(self.env.device).startswith('cuda'):
            torch.cuda.synchronize(self.env.device)

    def save(self, iteration, elapsed):
        path = self.run_dir / f'model_{iteration:06d}.pt'
        state = dict(variant=self.variant, architecture=self.architecture, config=asdict(self.cfg),
            actor_state_dict=self.actor.state_dict(), world_model=self.model.state_dict(),
            critic_state_dict=self.critic.state_dict(), mdp=self.mdp.state_dict(), iter=iteration,
            opt_model=self.opt_model.state_dict(), opt_auxiliary=self.opt_auxiliary.state_dict(),
            opt_actor=self.opt_actor.state_dict(), opt_critic=self.opt_critic.state_dict(),
            comparison=dict(method='bptt_v3', policy_updates=self.policy_updates,
                wm_only_iterations=self.cfg.model_only_iterations, wall_seconds=elapsed,
                sim_transitions=self.sim_transitions, curriculum_step=self.env.common_step_counter,
                replay_bytes=self.replay.bytes_used, curriculum_rate=self.curriculum_rate),
            rng_cpu=torch.get_rng_state(),
            rng_cuda=torch.cuda.get_rng_state(self.env.device) if str(self.env.device).startswith('cuda') else None)
        temporary = path.with_suffix('.tmp')
        torch.save(state, temporary); temporary.replace(path)
        (self.run_dir/'last_checkpoint.txt').write_text(str(path.resolve())+'\n')
        return path

    def load(self, path, optimizers=True):
        saved = torch.load(path, map_location=self.env.device, weights_only=False)
        if saved.get('variant') != self.variant or saved.get('architecture') != self.architecture:
            raise ValueError('Checkpoint robot/task/architecture differs from this NEST V3 run')
        # Resume preserves training schedule; changing batch/precision/compile is
        # allowed, but silently changing the warmup or optimization is not.
        for key in ('model_only_iterations','gamma','lam','learning_rate','decay_iterations','update_to_data_ratio'):
            if saved['config'][key] != getattr(self.cfg,key):
                raise ValueError(f'Resume requires the saved {key}')
        for module, key in ((self.actor,'actor_state_dict'),(self.model,'world_model'),
                            (self.critic,'critic_state_dict'),(self.mdp,'mdp')):
            module.load_state_dict(saved[key])
        if optimizers:
            for name in ('model','auxiliary','actor','critic'):
                getattr(self,'opt_'+name).load_state_dict(saved['opt_'+name])
        self.start_iteration = saved['iter']
        self.policy_updates = saved['comparison']['policy_updates']
        self.sim_transitions = saved['comparison']['sim_transitions']
        self.elapsed_before_resume = saved['comparison']['wall_seconds']
        self.env.common_step_counter = saved['comparison']['curriculum_step']
        self.curriculum_rate = saved['comparison'].get('curriculum_rate',
            saved['iter']*24/max(saved['iter']-self.cfg.model_only_iterations,1))
        from nest.tasks.velocity.mdp.curriculums import apply_velocity_stage_ranges
        curriculum = self.env.cfg.curriculum.command_vel
        if curriculum is not None:
            command = self.env.command_manager.get_term('base_velocity')
            apply_velocity_stage_ranges(command.cfg.ranges, curriculum.params['velocity_stages'], self.env.common_step_counter)
            command._update_command()
        torch.set_rng_state(saved['rng_cpu'].cpu())
        if saved['rng_cuda'] is not None and str(self.env.device).startswith('cuda'):
            torch.cuda.set_rng_state(saved['rng_cuda'].cpu(), self.env.device)
        return saved

    def export(self):
        # Stateless actor accepts (B,10,51) proprioception and (B,3) command.
        # The caller owns history/reset, so there is no batch-size-dependent
        # hidden state baked into the portable policy.
        actor = copy.deepcopy(self.actor).cpu().eval()
        old = ref.AMP_DTYPE
        ref.AMP_DTYPE = None
        try:
            example = (torch.zeros(1,self.history_length,self.layout.obs_dim), torch.zeros(1,3))
            with torch.no_grad():
                traced = torch.jit.trace(actor, example)
            dest = self.run_dir/'exports'
            dest.mkdir(exist_ok=True)
            traced.save(str(dest/'policy.pt'))
            (dest/'policy.json').write_text(json.dumps(dict(architecture=self.architecture,
                input_history=[self.history_length,self.layout.obs_dim], input_goal=3,
                frame_order=['root_COM_linear_velocity_body','root_angular_velocity_body','projected_gravity',
                             'controlled_joint_pos_relative','controlled_joint_vel_relative','previous_action'],
                action_scale=self.adapter.scale.cpu().tolist(), default_joint_pos=self.adapter.default.cpu().tolist(),
                history_order='oldest_to_newest', reset_history='repeat_current_frame',
                normalization='embedded', previous_action='normalized action before PD scaling'),indent=2))
        finally:
            ref.AMP_DTYPE = old
        return dest/'policy.pt'

    def learn(self):
        self.sync(); began = time.monotonic()
        # Like upstream, resume starts a fresh simulator and FIFO, with fresh
        # warmup data. It is an optimizer resume, not a bitwise trajectory resume.
        self.collect(self.cfg.warmup_steps)
        self.sync()
        with (self.run_dir/'metrics.jsonl').open('a') as stream:
            for iteration in range(self.start_iteration,self.iterations):
                # Match upstream's PPO-equivalent curriculum clock while keeping
                # NEST's actual velocity stage boundaries and command masks.
                self.env.common_step_counter = int(max(iteration-self.cfg.model_only_iterations,0)*self.curriculum_rate)
                from nest.tasks.velocity.mdp.curriculums import apply_velocity_stage_ranges
                cfg = self.env.cfg.curriculum.command_vel
                if cfg is not None:
                    command = self.env.command_manager.get_term('base_velocity')
                    apply_velocity_stage_ranges(command.cfg.ranges, cfg.params['velocity_stages'], self.env.common_step_counter)
                for opt in (self.opt_model,self.opt_actor,self.opt_critic):
                    for group in opt.param_groups:
                        group['lr'] = self.cfg.lr(iteration)
                self.sync(); t0 = time.monotonic()
                reward, falls = self.collect(self.cfg.collection_steps(iteration))
                self.sync(); t1 = time.monotonic()
                model_stats = self.fit_model()
                self.sync(); t2 = time.monotonic()
                policy_stats = self.improve() if iteration >= self.cfg.model_only_iterations else {}
                self.sync(); t3 = time.monotonic()
                elapsed = self.elapsed_before_resume+t3-began
                row = dict(iteration=iteration+1,policy_updates=self.policy_updates,wall_seconds=elapsed,
                    model_only=iteration<self.cfg.model_only_iterations,real_step_reward=reward,falls=falls,
                    collect_seconds=t1-t0,model_seconds=t2-t1,policy_critic_seconds=t3-t2,
                    sim_transitions=self.sim_transitions,curriculum_step=self.env.common_step_counter,
                    replay_bytes=self.replay.bytes_used,
                    peak_gpu_bytes=torch.cuda.max_memory_allocated(self.env.device) if str(self.env.device).startswith('cuda') else 0,
                    **model_stats,**policy_stats)
                stream.write(json.dumps(row)+'\n'); stream.flush()
                if (iteration+1)%25 == 0 or iteration == self.start_iteration:
                    (self.run_dir/'progress.json').write_text(json.dumps(row,indent=2))
                    print(f'it {iteration+1}/{self.iterations} policy {self.policy_updates} | WM {model_stats["model_loss"]:.4g} | reward {reward:.4f}',flush=True)
                if (iteration+1)%self.save_every == 0 or iteration+1 in (self.iterations,self.cfg.model_only_iterations):
                    self.save(iteration+1,elapsed)
            self.training_seconds = self.elapsed_before_resume+time.monotonic()-began
        self.export()
        return self.run_dir/f'model_{self.iterations:06d}.pt'
