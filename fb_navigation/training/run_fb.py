import numpy as np
import torch
from tqdm import trange

from fb_navigation.environments.continuous_four_rooms import ContinuousFourRooms
from fb_navigation.agents.forward_backward import FBLearner, RadialBasisFunction2D

env = ContinuousFourRooms(include_stay_action=False)

learner = FBLearner(
    obs_dim=2,
    n_actions=len(env.get_action_space()),
    embed_dim=50,
    learning_rate=5e-4,
    discount_factor=0.99,
    reg_coef=1,
    polyak_coef=0.95,
    batch_size=128,
    hidden_dim=256,
    num_layers=2,
    buffer_size=1_000_000,
    device="cuda",
    buffer_device="cuda",
    temperature=200.0,
    featuriser=RadialBasisFunction2D(
        x_bounds=(0.0, 1.0), y_bounds=(0.0, 1.0), dim=21, sigma=0.05, device="cuda"
    ),
    zero_z_p=0.25,
)


loss_history = []

for epoch in (pbar := trange(800)):
    for _ in range(25):
        for _ in range(4):
            obs = env.reset()

            w = learner.sample_cauchy_ball(1)
            for t in range(30):
                with torch.no_grad():
                    action = learner.act_e_greedy(
                        torch.tensor(
                            np.array(obs).reshape(1, -1), dtype=torch.float32
                        ).cuda(),
                        w,
                        epsilon=1.0,
                        available_actions=env.get_available_actions(obs),
                    )

                obs_new, reward, done, info = env.step(int(action))

                learner.update(
                    obs,
                    action,
                    obs_new,
                    False,
                    env.get_available_actions(obs_new),
                )
                if done:
                    obs = env.reset()
                else:
                    obs = obs_new

        for _ in range(40):
            fb_loss = learner.update_network()
            loss_history.append(fb_loss)

        learner.soft_update_target_networks()

    pbar.set_description_str(f"loss {loss_history[-1]:8.3f}")

np.save("fb_fourrooms_loss.npy", np.array(loss_history))
learner.save(path="fb_fourrooms.pt")
