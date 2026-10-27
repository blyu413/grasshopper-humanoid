"""V3 equivalence and robot-adapter invariants without Isaac Lab or a GPU."""
import unittest
from pathlib import Path

import mujoco
import numpy as np
import torch

from nest.mbrl.supermimic.model import Layout, Policy, SmallPolicy, make_policy, policy_variant, PhysicalWorldModel, WorldModel, MotorDelay, pack, set_precision
from nest.mbrl.supermimic.reference import networks as ref
from nest.mbrl.supermimic.reference.rotations import quat_to_mat, so3_exp
from nest.mbrl.supermimic.kinematics import Kinematics
from nest.mbrl.supermimic.config import Config
from nest.mbrl.supermimic.replay import GPUReplay
from nest.mbrl.supermimic.utils import frozen, lambda_returns
from nest.mbrl.supermimic.inference import HistoryPolicy

torch.set_num_threads(2)


class TestReferenceParity(unittest.TestCase):
    def setUp(self):
        set_precision('none')
        torch.manual_seed(42)

    def test_policy_exact_reference_when_dimensions_match(self):
        torch.manual_seed(10); original = ref.Policy(99,3,10)
        torch.manual_seed(10); native = Policy(99,29)
        for key,value in original.state_dict().items():
            torch.testing.assert_close(value,native.state_dict()[key],rtol=0,atol=0)
        obs,goal = torch.randn(2,10,99),torch.randn(2,3)
        torch.testing.assert_close(original(obs,goal),native(obs,goal),rtol=0,atol=0)
        self.assertEqual(Policy(51,14).dec[-1].out_features,14)

    def test_small_actor_gradients_export_and_history(self):
        actor = make_policy(51,14,'small_mlp')
        self.assertIsInstance(actor,SmallPolicy)
        self.assertNotEqual(policy_variant('small_mlp'),policy_variant('reference_v3'))
        obs,goal = torch.randn(3,5,51,requires_grad=True),torch.randn(3,3,requires_grad=True)
        actor.obs_norm.update(obs.detach());actor.goal_norm.update(goal.detach())
        output=actor(obs,goal)
        self.assertEqual(tuple(output.shape),(3,14))
        output.square().sum().backward()
        self.assertGreater(float(obs.grad.abs().sum()),0)
        self.assertGreater(float(goal.grad.abs().sum()),0)
        self.assertLess(sum(p.numel() for p in actor.parameters()),400000)
        traced=torch.jit.trace(actor,(torch.zeros(1,5,51),torch.zeros(1,3)))
        torch.testing.assert_close(traced(obs,goal),actor(obs,goal))
        wrapped=HistoryPolicy(traced,history=5)
        first,second=torch.randn(3,54),torch.randn(3,54)
        wrapped(first)
        history=torch.cat((first[:,:51,None].transpose(1,2).repeat(1,4,1),second[:,:51,None].transpose(1,2)),1)
        torch.testing.assert_close(wrapped(second),actor(history,second[:,51:]))
        wrapped.reset()
        torch.testing.assert_close(wrapped(first),actor(first[:,:51,None].transpose(1,2).repeat(1,5,1),first[:,51:]))

    def test_physics_values_and_action_jacobian_match_reference(self):
        L = Layout(29,29)
        torch.manual_seed(10); old = ref.WorldModel(hidden=(32,32))
        torch.manual_seed(10); new = PhysicalWorldModel(L,33,(32,32))
        new.load_state_dict(old.state_dict(),strict=False)
        with torch.no_grad():
            old.net[-1].weight.normal_(std=.01)
            new.net[-1].weight.copy_(old.net[-1].weight)
        s = torch.randn(2,71)*.1; s[:,3:7] = torch.tensor([1.,0,0,0])
        a = torch.randn(2,29,requires_grad=True); c = torch.randn(2,33)
        first,second = old.step(L.unpack(s),a,c),new.step(L.unpack(s),a,c)
        for x,y in zip(first,second): torch.testing.assert_close(x,y,rtol=0,atol=0)
        loss1,loss2 = sum(x.square().sum() for x in first),sum(x.square().sum() for x in second)
        torch.testing.assert_close(torch.autograd.grad(loss1,a)[0],torch.autograd.grad(loss2,a)[0])

    def test_auxiliary_does_not_drive_physics_or_physical_training_gradients(self):
        L=Layout(3,2,max_lag=1)
        cfg=Config(model_widths=(32,32),auxiliary_widths=(16,16),amp='none',compile=False)
        model=WorldModel(L,5,torch.zeros(2),torch.ones(2),100.,.02,cfg)
        s=torch.randn(2,L.size)*.01; s[:,3:7]=torch.tensor([1.,0,0,0]); s[:,-2:]=0; s[:,-1]=10
        a,c=torch.randn(2,2),torch.randn(2,5)
        changed=s.clone();changed[:,L.physical:L.delay_start]+=10
        with torch.no_grad():
            model.physical.net[-1].weight.normal_(std=.01)
            model.auxiliary.net[-1].weight.normal_(std=.01)
        torch.testing.assert_close(model(s,a,c)[0][:,:L.physical],model(changed,a,c)[0][:,:L.physical])
        output=model(s,a,c,detach_physics_for_auxiliary=True)[0]
        output[:,L.physical:L.delay_start].square().sum().backward()
        self.assertTrue(all(p.grad is None or not p.grad.any() for p in model.physical.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.any() for p in model.auxiliary.parameters()))
        model.zero_grad(set_to_none=True)
        a.requires_grad_(True)
        with frozen(model):
            model(s,a,c)[0].square().sum().backward()
        self.assertTrue(torch.isfinite(a.grad).all());self.assertGreater(float(a.grad.abs().sum()),0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_v3_defaults_and_returns(self):
        cfg=Config()
        self.assertEqual((cfg.window,cfg.horizon,cfg.model_batches,cfg.model_batch_size),(8,32,10,2048))
        self.assertEqual((cfg.warmup_steps,cfg.model_only_iterations),(64,500))
        self.assertEqual([cfg.collection_steps(i) for i in range(4)],[2,3,2,3])
        values=torch.tensor([[2.],[3.],[100.]])
        out=lambda_returns(torch.ones(2,1),torch.tensor([[0.],[1.]]),values,.9,.95)
        self.assertAlmostEqual(out[1].item(),1.)
        self.assertAlmostEqual(out[0].item(),1+.9*(.05*3+.95),places=5)


class TestRobotInterfaces(unittest.TestCase):
    def test_motor_delay_matches_native_ring_including_resampling_and_reset(self):
        from nest.tasks.velocity.mdp.actions import PhysicsStepDelay
        L=Layout(3,2,max_lag=4,delay_period=5)
        default=torch.tensor([.1,-.2]);scale=torch.tensor([.2,.4])
        neutral=default[None].repeat(3,1)
        native=PhysicsStepDelay(neutral,0,4,5)
        model=MotorDelay(L,default,scale)
        for step in range(6):
            if step==3: native.reset(neutral,torch.tensor([1]))
            rows=(torch.arange(5)+native.cursor)%5
            state=torch.cat((native.values[rows].permute(1,0,2).flatten(1),native.lags[:,None],((native.ticks+native.phases)%5)[:,None]),-1)
            a=torch.randn(3,2,requires_grad=True)
            targets=[];lags=[]
            for _ in range(4):
                targets.append(native.compute((default+a.detach()*scale)).clone());lags.append(native.lags.clone())
            actual=torch.stack(targets,1)
            next_state,executed,_=model(state,a,torch.stack(lags,1))
            torch.testing.assert_close(executed.reshape(3,4,2)*scale+default,actual)
            rows=(torch.arange(5)+native.cursor)%5
            torch.testing.assert_close(next_state[:,:-2],native.values[rows].permute(1,0,2).flatten(1))
            executed.sum().backward()
            self.assertTrue(torch.isfinite(a.grad).all())

    def test_fk_matches_mujoco_with_nonzero_anchor_and_passive_mimic(self):
        m=mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body name="PELVIS"><freejoint/>
          <inertial pos=".1 0 0" mass="1" diaginertia="1 1 1"/>
          <body name="first" pos=".2 .1 .3" quat=".9238795 0 .3826834 0"><joint name="driven" pos=".1 .05 0" axis="0 1 0"/>
            <geom size=".1" mass="1"/><body name="second" pos="0 0 .4"><joint name="mimic" axis="1 0 0"/><geom size=".1" mass="1"/></body>
          </body></body></worldbody><equality><joint joint1="mimic" joint2="driven" polycoef="0 -1 0 0 0"/></equality></mujoco>''')
        kin=Kinematics(m,['mimic','driven'],['second','PELVIS','first'])
        L=Layout(2,1)
        for seed in range(3):
            d=mujoco.MjData(m);rng=np.random.default_rng(seed)
            d.qpos[:3]=rng.normal(size=3);d.qpos[3:7]=rng.normal(size=4);d.qpos[3:7]/=np.linalg.norm(d.qpos[3:7])
            d.qpos[7:]=rng.normal(size=2);d.qvel[:]=rng.normal(size=8);mujoco.mj_forward(m,d)
            # qvel free angular velocity is local; our state stores world omega.
            R=d.xmat[1].reshape(3,3)
            s=torch.tensor(np.r_[d.qpos[:7],d.qvel[:3],R@d.qvel[3:6],d.qpos[[8,7]],d.qvel[[7,6]]],dtype=torch.float32)[None]
            s.requires_grad_(True)
            P,Q,V,W=kin(L.unpack(s))
            for i,b in enumerate([3,1,2]):
                velocity=np.zeros(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,b,velocity,0)
                # mj_objectVelocity returns velocity at COM, shift to link origin.
                link_v=velocity[3:]-np.cross(velocity[:3],d.xipos[b]-d.xpos[b])
                np.testing.assert_allclose(P[0,i].detach(),d.xpos[b],atol=1e-6)
                np.testing.assert_allclose(Q[0,i].detach(),d.xmat[b].reshape(3,3),atol=1e-6)
                np.testing.assert_allclose(V[0,i].detach(),link_v,atol=1e-6)
                np.testing.assert_allclose(W[0,i].detach(),velocity[:3],atol=1e-6)
            P.square().sum().backward();self.assertTrue(torch.isfinite(s.grad).all())

    def test_fifo_masks_terminal_and_exogenous_boundaries_and_history(self):
        L=Layout(2,1,max_lag=0)
        replay=GPUReplay(6,1,2,'cpu',L)
        for t in range(10):
            data={key:torch.zeros_like(value[0]) for key,value in replay.data.items()}
            data['s'].fill_(t);data['s1'].fill_(t+.5);data['ep'].fill_(t%3)
            data['done'].fill_(t%3==2);data['cut'].fill_(t==7)
            replay.add(data)
        torch.manual_seed(8)
        windows=replay.windows(256,3)
        stop=(windows['done']|windows['cut']).squeeze(-1)
        for i in range(256):
            alive=1
            for t in range(3):
                self.assertEqual(windows['mask'][i,t].item(),alive)
                alive*=int(not stop[i,t])
        start,h=replay.starts(256,10)
        for row in range(256):
            current=int(start['s'][row,0]);low=max(4,current-current%3)
            self.assertGreaterEqual(float(h['s'][row,:,0].min()),low)
            self.assertEqual(float(h['s'][row,-1,0]),current)

    def test_inference_history_reset_and_commands_not_stacked(self):
        class Spy(torch.nn.Module):
            def forward(self,h,g):
                self.h,self.g=h.clone(),g.clone()
                return h[:,-1,:14]
        spy=Spy();policy=HistoryPolicy(spy)
        first=torch.randn(2,54);second=torch.randn(2,54)
        policy(first);policy(second)
        torch.testing.assert_close(spy.h[:,-2],first[:,:51])
        torch.testing.assert_close(spy.h[:,-1],second[:,:51])
        torch.testing.assert_close(spy.g,second[:,51:])
        policy.reset();policy(first)
        torch.testing.assert_close(spy.h,first[:,:51,None].transpose(1,2).expand(-1,10,-1))


if __name__=='__main__':
    unittest.main()
