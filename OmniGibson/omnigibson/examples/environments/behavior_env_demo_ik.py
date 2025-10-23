import os

import yaml

import omnigibson as og
from omnigibson.macros import gm

import omnigibson.lazy as lazy
from omnigibson.utils.ui_utils import KeyboardRobotController, choose_from_options
# Make sure object states are enabled
gm.ENABLE_OBJECT_STATES = True
gm.USE_GPU_DYNAMICS = False


def main(random_selection=False, headless=False, short_exec=False):
    """
    Generates a BEHAVIOR Task environment in an online fashion.

    It steps the environment 100 times with random actions sampled from the action space,
    using the Gym interface, resetting it 10 times.
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

    # Load the pre-selected configuration and set the online_sampling flag
    config_filename = os.path.join(og.example_config_path, "r1pro_behavior_ik.yaml")
    cfg = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
    #
    # Load the environment
    env = og.Environment(configs=cfg)

    # Move camera to a good position
    og.sim.viewer_camera.set_position_orientation(
        position=[1.6, 6.15, 1.5], orientation=[-0.2322, 0.5895, 0.7199, -0.2835]
    )

    # Allow user to move camera more easily
    #og.sim.enable_viewer_camera_teleoperation()

    robot = env.robots[0]
    # Create teleop controller
    action_generator = KeyboardRobotController(robot=robot)

    # Register custom binding to reset the environment
    action_generator.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.R,
        description="Reset the robot",
        callback_fn=lambda: env.reset(),
    )

    # Print out relevant keyboard info if using keyboard teleop
    action_generator.print_keyboard_teleop_info()

    max_steps = -1 if not short_exec else 100
    step = 0

    while step != max_steps:
        action = action_generator.get_teleop_action()
        env.step(action=action)
        step += 1
    # # Run a simple loop and reset periodically
    # max_iterations = 10 if not short_exec else 1
    # for j in range(max_iterations):
    #     og.log.info("Resetting environment")
    #     env.reset()
    #     for i in range(100):
    #         action = env.robots[0].action_space.sample()
    #         state, reward, terminated, truncated, info = env.step(action * 0.1)
    #         if terminated or truncated:
    #             og.log.info("Episode finished after {} timesteps".format(i + 1))
    #             break

    # Always close the environment at the end
    og.shutdown()


if __name__ == "__main__":
    main()
