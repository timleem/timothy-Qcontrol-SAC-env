# %%
import gymnasium as gym
import numpy as np
from qutip import (Options, basis, expect, ket2dm, liouvillian, mesolve,
                   operator_to_vector, vector_to_operator, sigmax, sigmay, sigmaz)

opts = Options(atol=1e-11, rtol=1e-9, nsteps=int(1e6))
# opts = Options(atol=1e-13, rtol=1e-11, nsteps=int(1e6))
opts.normalize_output = True  # mesolve is x3 faster if this is False
# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

# %%
class threelevel_Qcontrol(gym.Env):
    """Λ-system environment"""

    def __init__(
        self,
        Ωmax=1.0,
        Ωmin=0.0,
        Δ=0.0,
        δp=0.0,
        T=10.0,
        n_steps=20,
        γ=0.0,
        reward_gain=1.0,
        seed=1,
        fid1=0.99,
    ):
        """Initializes a new Λ-system environment.
        Args:
          seed: random seed for the RNG.
        """
        self.qstate = [basis(4, i) for i in range(4)]
        self.sig = [
            [self.qstate[i] * self.qstate[j].dag() for j in range(4)] for i in range(4)
        ]
        self.up = (self.sig[0][1] + self.sig[1][0]) / 2
        self.us = (self.sig[1][2] + self.sig[2][1]) / 2
        self.ψ0 = ket2dm(self.qstate[0])
        self.target_state = self.sig[2][2]

        self.Ωmax = Ωmax
        self.Ωmin = Ωmin
        self.Δ = Δ
        self.δp = δp
        self.T = T
        self.n_steps = n_steps
        self.γ = γ
        self.reward_gain = reward_gain
        self.fid1 = fid1
        self.prev_reward = 0
        self.episode_tracker = 0        ##delete this once done investigating

        self.action_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(2,),
            dtype=np.float32,
            seed=1235711131
            )
        self.observation_space = gym.spaces.Box(
            low=np.append(
                np.zeros(3, dtype=np.float32), -1 * np.ones(6, dtype=np.float32)
            ),
            high=np.ones(9, dtype=np.float32),
            shape=(9,),
            dtype=np.float32
        )

        self.current_step = 0
        self.current_qstate = self.ψ0
        self._state = self._dm2state(self.ψ0)
        self.terminated = False
        self.truncated = False

        self.update()

    def update(self):
        self.H0 = self.Δ * self.sig[1][1] + self.δp * self.sig[2][2]
        self.tlist = np.linspace(0, self.T, self.n_steps + 1)
        self.Δt = self.tlist[1] - self.tlist[0]

    def _qstep(self, action, qstate):
        action_0 = (action[0]+1)/2*self.Ωmax
        action_1 = (action[1]+1)/2*self.Ωmax
        # if self.episode_tracker == 2610:
        # print(action_0, action_1)
        H = self.H0 + action_0 * self.up + action_1 * self.us
        L = (liouvillian(H, [np.sqrt(self.γ) * self.sig[3][1]]) * self.Δt).expm()
        return apply_superoperator(L, qstate)

    def _mesolvestep(self, action, qstate):
        H = self.H0 + action[0] * self.up + action[1] * self.us
        tlist = self.tlist[self.current_step : self.current_step + 2]
        result = mesolve(
            H, qstate, tlist, c_ops=[np.sqrt(self.γ) * self.sig[3][1]], options=opts
        )
        return result.states[-1]

    def _dm2state(self, dm):
        return np.append(
            dm.diag()[:-1],
            np.append(
                dm.full()[([0, 0, 1], [1, 2, 2])].real,
                dm.full()[([0, 0, 1], [1, 2, 2])].imag,
            ),
        ).astype(np.float32)
    
    def reset(self,seed=None):
        self._reset()
        self.position = self._state
        self.episode_tracker = self.episode_tracker + 1
        info = {}
        return self.position, info
    
    def _reset(self):
        # self._reset_next_step = False
        self.current_step = 0
        self.current_qstate = self.ψ0
        self._state = self._dm2state(self.ψ0)
        self.terminated = False
        self.truncated = False
    
    def step(self, action):
        info = {}
        self.position, reward = self._step(action)
        return self.position, reward, self.terminated, self.truncated, info

    def _step(self, action):
        """Updates the environment according to the action.""" 
        terminal = False
        truncated = False
        if self.current_step < self.n_steps:
            self.current_qstate = self._qstep(action, self.current_qstate)
            # self.current_qstate = self._mesolvestep(action, self.current_qstate)
            next_state = self._dm2state(self.current_qstate)
            reward = 0
            if expect(self.current_qstate, self.target_state) > self.fid1:
                    terminal = True
                    reward = (self.reward_gain *(expect(self.target_state,self.current_qstate)*(self.n_steps-self.current_step+1)))
            if self.current_step == self.n_steps - 1:
                reward = self.reward_gain * expect(
                    self.target_state, self.current_qstate
                )
                truncated = True
                # if self.current_step == self.n_steps -1:
                #     fid = expect(self.current_qstate, self.target_state)
                #     print("reward: {reward}, fidelity: {fid}".format(reward=reward, fid=fid))
        else:
            self.truncated = True
            reward = -1
            next_state = np.zeros(9)
            # print(self.truncated)
        self.current_step += 1

        # if self.episode_tracker > 1686:
        #     print(self.current_qstate)
        #     print(reward)
            
        if truncated:
            self.truncated = True

        if terminal:
            self.terminated = True
        return next_state, reward


    def run_evolution(self, amps):
        time_step = self.reset()
        time_step_list = [time_step]

        for i in range(self.n_steps):
            time_step = self.step(amps[i])
            time_step_list.append(time_step)

        assert time_step.is_last() == True

        state_list = np.array([x.observation for x in time_step_list])
        reward_list = np.array([x.reward for x in time_step_list])
        terminal_list = np.array([x.step_type for x in time_step_list])

        return state_list, reward_list, terminal_list

    def run_qstepevolution(self, amps):
        Ωp = amps[:, 0]
        Ωs = amps[:, 1]

        states = [self.ψ0]

        for i in range(self.n_steps):
            states.append(self._qstep([Ωp[i], Ωs[i]], states[-1]))

        return states

    def run_mesolvevolution(self, amps):
        Ωp = amps[:, 0]
        Ωs = amps[:, 1]

        fp = function_from_array(Ωp, self.tlist[:-1])
        fs = function_from_array(Ωs, self.tlist[:-1])

        H = [self.H0, [self.up, fp], [self.us, fs]]

        result = mesolve(
            H,
            self.ψ0,
            self.tlist,
            c_ops=[np.sqrt(self.γ) * self.sig[3][1]],
            options=opts,
        )
        return result

    def final_efficiency(self, amps):
        return self.reward_gain * expect(
            self.target_state, self.run_mesolvevolution(amps).states[-1]
        )

    def inefficiency(self, vals):
        amps = vals2amps(vals)
        return 1 - self.final_efficiency(amps)

    def final_qstepefficiency(self, amps):
        return self.reward_gain * expect(
            self.target_state, self.run_qstepevolution(amps)[-1]
        )
    
    def track_qstepefficiency(self, amps):
        states_list = self.run_qstepevolution(amps=amps)
        fidelity = []
        for state in states_list:
            fidelity.append(self.reward_gain * expect(
                self.target_state, state))
        return fidelity

    def qstepinefficiency(self, vals):
        amps = vals2amps(vals)
        return 1 - self.final_qstepefficiency(amps)
    
    def render(self):
        pass

def vals2amps(vals):
    message = "vals must be 1-D array with shape (n,) where n is even"
    assert len(vals.shape) == 1, message
    assert vals.shape[0] % 2 == 0, message
    return vals.reshape(-1, 2, order="F")


def function_from_array(y, x):
    """Return function given an array and time points."""

    if y.shape[0] != x.shape[0]:
        raise ValueError("y and x must have the same first dimension")

    yx = np.column_stack((y, x))
    yx = yx[yx[:, -1].argsort()]

    def func(t, args):
        idx = np.searchsorted(yx[1:, -1], t, side="right")
        return yx[idx, 0]

    return func

def apply_superoperator(L, ρ):
    return vector_to_operator(L * operator_to_vector(ρ))

def gaussian(x, T):
    return np.exp(-((x / T) ** 2))


def Omm1rand(Ω_, env_parameters):
    return (Ω_ - 1) + np.random.rand(2 * env_parameters["n_steps"])


def Om(Ω_, env_parameters):
    return Ω_ * np.ones(2 * env_parameters["n_steps"])


def Omgaussian(Ω_, env_parameters):
    T_ = env_parameters["T"]
    tlist = np.linspace(0, T_, env_parameters["n_steps"], endpoint=False)

    return Ω_ * np.append(
        gaussian(tlist - 0.7 * T_, T_ / 3), gaussian(tlist - 0.3 * T_, T_ / 3)
    )


def Omrand(Ω_, env_parameters):
    return Ω_ * np.random.rand(2 * env_parameters["n_steps"])