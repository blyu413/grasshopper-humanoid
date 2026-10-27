#!/usr/bin/env python3
"""Train NEST Grasshopper with Motion Force Adapter's reference V3 WM+BPTT."""
from __future__ import annotations
import argparse
import contextlib
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault('MPLCONFIGDIR','/tmp/nest-wm-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/nest-wm-cache')
import warp as wp
wp.config.enable_backward = False
wp.config.kernel_cache_dir = '/tmp/nest-wm-warp-cache'
from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[1]
TASK = 'Nest-Walk-Flat-Grasshopper'


def validate(runner, steps):
    import torch
    @contextlib.contextmanager
    def full_precision_comparison():
        # Compare independent FK/export implementations without TF32 rounding
        # accumulating through the robot's many 3x3 rotation products.
        previous = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision('highest')
        try:
            yield
        finally:
            torch.set_float32_matmul_precision(previous)
    from nest.mbrl.supermimic.adapter import tensor
    errors, reward_errors, delay_errors = [], [], []
    for _ in range(steps):
        with torch.no_grad():
            s, c, g = runner.adapter.state(), runner.context.capture(), runner.adapter.goal()
            ap = runner.env.action_manager.action.clone()
            d, ids = runner.adapter.robot.data, runner.adapter.action_ids
            native = torch.cat((tensor(d.root_lin_vel_b),tensor(d.root_ang_vel_b),tensor(d.projected_gravity_b),
                                (tensor(d.joint_pos)-tensor(d.default_joint_pos))[:,ids],
                                (tensor(d.joint_vel)-tensor(d.default_joint_vel))[:,ids],ap),-1)
            errors.append((native-runner.mdp.obs(s,ap,c,g)).abs().max())
            runner.collect(1)
            a = runner.replay.data['a'][runner.replay.row(runner.replay.size-1)]
            target = runner.adapter.target
            with full_precision_comparison():
                analytical = runner.mdp.reward_terms(target,a,ap,runner.adapter.previous_peak,c,g,runner.adapter.landing)
            reward_errors.append((analytical-runner.adapter.reward_values).abs().amax(0))
            delayed,_,_ = runner.model.delay(s[:,runner.layout.delay_start:],a,runner.adapter.executed_lags())
            delay_errors.append((delayed-target[:,runner.layout.delay_start:]).abs().max())
    model_stats = runner.fit_model()
    frozen_before = {k:v.clone() for k,v in runner.model.state_dict().items()}
    actor_before = {k:v.clone() for k,v in runner.actor.named_parameters()}
    policy_stats = runner.improve()
    model_frozen = all(torch.equal(v,runner.model.state_dict()[k]) for k,v in frozen_before.items())
    branch_names = ('mlp',) if runner.cfg.policy_profile == 'small_mlp' else ('enc','dec')
    branches = {branch:any(not torch.equal(v,dict(runner.actor.named_parameters())[k])
                          for k,v in actor_before.items() if k.startswith(branch+'.')) for branch in branch_names}
    checkpoint = runner.save(1,0.)
    with torch.no_grad():
        obs = runner.history.clone(); goal = runner.adapter.goal()[:,:3]
        expected = runner.actor(obs,goal).clone()
        getattr(runner.actor, 'mlp' if runner.cfg.policy_profile == 'small_mlp' else 'dec')[-1].bias.add_(1.)
        runner.load(checkpoint)
        roundtrip = float((expected-runner.actor(obs,goal)).abs().max())
    export = runner.export()
    from nest.mbrl.supermimic import model
    old = model.ref.AMP_DTYPE
    model.ref.AMP_DTYPE = None
    with torch.no_grad(), full_precision_comparison():
        expected_export = runner.actor(obs,goal).cpu()
        exported = torch.jit.load(str(export))(obs.cpu(),goal.cpu())
    model.ref.AMP_DTYPE = old
    export_error = float((exported-expected_export).abs().max())
    result = dict(scope='one WM fit and H32 actor/critic update; not a trained locomotion policy',
        clean_observation_max_error=float(torch.stack(errors).max()),
        reward_max_abs_error_by_term=dict(zip(runner.mdp.names,torch.stack(reward_errors).amax(0).cpu().tolist())),
        delay_max_abs_error=float(torch.stack(delay_errors).max()),
        model_frozen_during_policy_update=model_frozen,actor_branches_changed=branches,
        checkpoint_roundtrip_max_error=roundtrip,export_max_abs_error=export_error,
        model=model_stats,policy=policy_stats,architecture=runner.architecture,replay_bytes=runner.replay.bytes_used,
        contacts_validated=runner.adapter.has_contact,
        comparison_float32_matmul_precision='highest; training updates retain configured precision')
    result['passed'] = (result['clean_observation_max_error']<3e-4
        and max(result['reward_max_abs_error_by_term'].values())<3e-4
        and result['delay_max_abs_error']<3e-5 and model_frozen and all(branches.values())
        and roundtrip==0 and export_error<3e-4
        and all(torch.isfinite(torch.tensor(x)) for x in (*model_stats.values(),*policy_stats.values())))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iterations',type=int,default=5000,help='Total iterations, including WM-only warmup; absolute endpoint on resume')
    parser.add_argument('--num-envs',type=int,default=4096)
    parser.add_argument('--policy-profile',choices=('reference_v3','small_mlp'),default='reference_v3',
                        help='small_mlp: V2-sized 512/256/128 actor with five NEST frames; same V3 WM/BPTT')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--log-dir',type=Path)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--save-interval',type=int,default=1000)
    parser.add_argument('--model-only-iterations',type=int,default=500)
    parser.add_argument('--replay-steps',type=int,default=1000)
    parser.add_argument('--amp',choices=('bf16','none'),default='bf16')
    parser.add_argument('--no-compile',action='store_true')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare-only',action='store_true')
    mode.add_argument('--validate-only',action='store_true')
    mode.add_argument('--cpu-smoke',action='store_true',help='One native MuJoCo env; contact sensor/rewards omitted; same V3 networks')
    parser.add_argument('--validation-steps',type=int,default=64)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    import gymnasium as gym
    import torch
    from isaaclab.app import launch_simulation
    from isaaclab.utils.io import dump_yaml
    from isaaclab_tasks.utils import resolve_task_config
    import nest.tasks
    from nest.mbrl.supermimic.config import Config
    from nest.mbrl.supermimic.runner import ReferenceRunner
    from nest.mbrl.supermimic.model import make_policy, policy_variant, Layout
    from nest.assets.robots.grasshopper import POLICY_JOINT_NAMES, JOINT_EFFORT_LIMITS, URDF_PATH, USD_PATH
    from nest.tasks.velocity.config.grasshopper.env_cfg import TRAINING_DESIGN, COMMAND_DESIGN, DOMAIN_RANDOMIZATION_DESIGN
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision('high')
    if args.iterations<1 or args.num_envs<1 or args.save_interval<1 or args.validation_steps<8:
        parser.error('Positive iterations/envs/save interval and >=8 validation steps required')
    if args.prepare_only and args.resume:
        parser.error('prepare-only writes a fresh actor; use a real run to resume a checkpoint')
    if args.cpu_smoke and args.resume:
        parser.error('CPU smoke cannot resume a full task checkpoint')
    cfg, _ = resolve_task_config(TASK,'rsl_rl_cfg_entry_point',overrides=['physics=newton_mjwarp'])
    cfg.scene.num_envs = 1 if args.cpu_smoke else args.num_envs
    cfg.seed = args.seed
    cfg.sim.device = 'cpu' if args.cpu_smoke else args.device or 'cuda:0'
    cfg.sim.visualizer_cfgs = []
    mc = Config(policy_profile=args.policy_profile,model_only_iterations=args.model_only_iterations,replay_steps=args.replay_steps,
                replay_device=cfg.sim.device,amp=args.amp,compile=not args.no_compile)
    if args.cpu_smoke:
        from isaaclab_newton.physics import NewtonManager, NewtonMJWarpManager
        class NewtonCpuWMSmokeManager(NewtonMJWarpManager):
            @classmethod
            def _initialize_contacts(cls):
                if cls._report_contacts:
                    raise RuntimeError('Native CPU smoke cannot export Newton contacts')
                NewtonManager._contacts = None
        cfg.scene.feet_contact = None
        cfg.scene.robot.spawn.activate_contact_sensors = False
        for name in ('foot_swing_height','foot_slip','soft_landing'):
            setattr(cfg.rewards,name,None)
        for name in ('foot_air_time','foot_contact','foot_contact_forces'):
            setattr(cfg.observations.critic,name,None)
        cfg.sim.physics.solver_cfg.use_mujoco_cpu = True
        cfg.sim.physics.solver_cfg.class_type = NewtonCpuWMSmokeManager
        cfg.sim.physics.class_type = NewtonCpuWMSmokeManager
        mc.amp, mc.compile = 'none', False
        mc.model_batches, mc.model_batch_size, mc.imagination_batch_size = 1, 4, 4
        mc.critic_epochs, mc.critic_minibatches, mc.replay_steps = 1, 1, max(64,args.validation_steps)
    elif not args.prepare_only and (not cfg.sim.device.startswith('cuda') or not torch.cuda.is_available()):
        parser.error('CUDA unavailable; use --prepare-only or --cpu-smoke')
    if not args.cpu_smoke and not args.validate_only and not 0<=mc.model_only_iterations<args.iterations:
        parser.error('Require 0 <= model-only-iterations < iterations')
    if mc.compile and not args.prepare_only:
        from nest.mbrl.supermimic.utils import ensure_compile_headers
        ensure_compile_headers()
    cfg.validate()
    log = (args.log_dir or ROOT/'logs/supermimic/grasshopper_walk'/datetime.now().strftime('%Y-%m-%d_%H-%M-%S')).resolve()
    log.mkdir(parents=True,exist_ok=False)
    dump_yaml(str(log/'params/env.yaml'),cfg)
    task_spec = dict(task=TASK,training_design=TRAINING_DESIGN,command_design=COMMAND_DESIGN,
                    domain_randomization_design=DOMAIN_RANDOMIZATION_DESIGN,cpu_smoke=args.cpu_smoke,
                    urdf_sha256=hashlib.sha256(URDF_PATH.read_bytes()).hexdigest(),
                    usd_sha256=hashlib.sha256(USD_PATH.read_bytes()).hexdigest(),
                    control_dt=cfg.sim.dt*cfg.decimation)
    manifest = dict(variant=policy_variant(mc.policy_profile),status='preparing',args=vars(args),task_spec=task_spec,
                    model_config=asdict(mc),source_sha256={},
                    resume_semantics='optimizer/network resume; fresh simulator, histories and replay',
                    curriculum='upstream PPO-equivalent clock; NEST velocity stage thresholds')
    paths = [Path(__file__),*sorted((ROOT/'src/nest').rglob('*.py')),
             ROOT/'src/nest/mbrl/supermimic/reference/LICENSE',
             ROOT/'src/nest/mbrl/supermimic/reference/ORIGIN.md',URDF_PATH,USD_PATH]
    for path in paths:
        rel = path.relative_to(ROOT); dest = log/'source'/rel
        dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(path,dest)
        manifest['source_sha256'][str(rel)] = hashlib.sha256(path.read_bytes()).hexdigest()
    def write():
        (log/'run.json').write_text(json.dumps(manifest,indent=2,default=str)+'\n')
    write()
    try:
        if args.prepare_only:
            layout = Layout(len(JOINT_EFFORT_LIMITS),len(POLICY_JOINT_NAMES))
            actor = make_policy(layout.obs_dim,layout.actions,mc.policy_profile)
            torch.save(actor.state_dict(),log/'initial_actor.pt')
            manifest.update(status='prepared',simulation_validated=False,actor_parameters=sum(p.numel() for p in actor.parameters()),
                            policy_observation_shape=[actor.hist,layout.obs_dim],goal_shape=[3],action_dim=layout.actions)
            write(); print(f'WM_PREPARED {log}',flush=True); return
        with launch_simulation(cfg,args):
            with contextlib.closing(gym.make(TASK,cfg=cfg)) as env:
                env.reset()
                runner = ReferenceRunner(env,mc,log,args.iterations,args.save_interval,task_spec)
                try:
                    if args.resume:
                        runner.load(args.resume)
                        if runner.start_iteration>=args.iterations:
                            raise ValueError('--iterations must exceed saved iteration')
                    manifest.update(status='validating' if args.validate_only or args.cpu_smoke else 'training',
                                    architecture=runner.architecture,replay_bytes=runner.replay.bytes_used)
                    write()
                    if args.validate_only or args.cpu_smoke:
                        result = validate(runner,args.validation_steps)
                        (log/'validation.json').write_text(json.dumps(result,indent=2))
                        if not result['passed']:
                            raise RuntimeError(f'V3 preflight failed: {result}')
                        manifest['status'] = 'cpu_smoke_complete' if args.cpu_smoke else 'validation_complete'
                    else:
                        runner.save(runner.start_iteration,runner.elapsed_before_resume)
                        manifest['checkpoint'] = str(runner.learn())
                        manifest['status'] = 'complete'
                    manifest['simulation_validated'] = True
                    write(); print(f'WM_{manifest["status"].upper()} {log}',flush=True)
                finally:
                    runner.adapter.close()
    except BaseException as error:
        manifest.update(status='failed',error=str(error)); write(); raise


if __name__=='__main__':
    main()
