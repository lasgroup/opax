import copy

import numpy as np
from scipy.spatial.transform import Rotation as R

from typing import Optional
from gym import core, spaces

from elastica._calculus import _isnan_check
from elastica import *


from opax.models.environment_models.soft_arm_reward import SoftArmRewardModel, MuscleTorquesWithVaryingBetaSplines


# Set base simulator class
class BaseSimulator(BaseSystemCollection, Constraints, Connections, Forcing, Damping, CallBacks):
    pass


class SoftArmReachOrienEnv(core.Env):
    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(self, integration_freq: float = 1.0e-4, sim_interval: float = 0.01, matplot_lib_rendering: bool = True,
                 game_mode: int = 1, dim: float = 3.0, COLLECT_DATA_FOR_POSTPROCESSING: bool = False,
                 reward_model: SoftArmRewardModel = SoftArmRewardModel(), **kwargs):
        self.dim = dim
        # Integrator type
        self.StatefulStepper = PositionVerlet()

        self.n_elem = 20
        self.sim_dt = integration_freq
        self.time_step = self.sim_dt
        self.RL_update_interval = sim_interval  # This is 100 updates per second
        self.num_steps_per_update = np.rint(
            self.RL_update_interval / self.sim_dt
        ).astype(int)
        self.matplot_lib_rendering = matplot_lib_rendering
        self.youngs_modulus = 1e7
        self.poisson_ratio = 0.5
        self.shear_modulus = 0.5 * self.youngs_modulus / (self.poisson_ratio + 1.0)
        self.damping = kwargs.get("damping", 0.2)
        self.torque_scale = kwargs.get("torque_scale", 5e-5)

        self.max_episode_final_time = 5  # seconds

        self.eval_log_step = 0

        self.base_length = 1
        self.radius = 0.05
        self.rest_rod_position_collection = None

        self.alpha = self.torque_scale * self.radius * self.youngs_modulus
        # self.alpha = 75
        self.beta = 75

        self.number_of_control_points = 6
        # self.number_of_observation_segments = self.number_of_control_points

        self.rendering_fps = 30
        self.step_skip = np.rint(1.0 / (self.rendering_fps * self.sim_dt)).astype(int)

        self.max_rate_of_change_of_activation = np.infty
        self.target_v_scale = 0.1

        # target position
        self.mode = game_mode
        self.target_position = kwargs.get("target_position", np.array([0.3, 0.95, 0.0]))
        self.target_orientation = kwargs.get("target_orientation", np.array([0.5, 0.5, 0.5, 0.5]))

        self.time_tracker = np.float64(0.0)

        if self.dim == 2.0:
            # normal direction activation (2D)
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.number_of_control_points,),
                dtype=np.float64,
            )
            self.action = np.zeros(self.number_of_control_points)
        if self.dim == 3.0 or self.dim == 2.5:
            # normal and/or binormal direction activation (3D)
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(2 * self.number_of_control_points,),
                dtype=np.float64,
            )
            self.action = np.zeros(2 * self.number_of_control_points)
        if self.dim == 3.5:
            # normal, binormal and/or tangent direction activation (3D)
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(3 * self.number_of_control_points,),
                dtype=np.float64,
            )
            self.action = np.zeros(3 * self.number_of_control_points)

        self.obs_state_points = 4
        num_points = int(self.n_elem / self.obs_state_points)
        num_rod_state = len(np.ones(self.n_elem + 1)[0::num_points])
        self.num_rod_state = num_rod_state

        # no information about target,
        # for each obs node: 3 (position) 3 (velocity)
        # for each obs elem: 4 (orientation) 3 (angular velocity)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(num_rod_state * 6 + self.obs_state_points * 7,),
            dtype=np.float64,
        )

        if self.mode == 2:
            assert "boundary" in kwargs, "need to specify boundary in mode 2"
            self.boundary = kwargs.get("boundary", [-0.6, 0.6, 0.3, 0.9, -0.6, 0.6])

        if self.mode == 3:
            assert "target_v" in kwargs, "need to specify target_v in mode 3"
            self.target_v = kwargs["target_v"]

        if self.mode == 4:
            assert (
                    "boundary" and "target_v" in kwargs
            ), "need to specify boundary and target_v in mode 4"
            self.boundary = kwargs.get("boundary", [-0.6, 0.6, 0.3, 0.9, -0.6, 0.6])
            self.target_v = kwargs["target_v"]

        self.reward_model = reward_model

        # Rendering-related
        self.viewer = None
        self.renderer = None
        self.render_mode = "rgb_array"

        self.COLLECT_DATA_FOR_POSTPROCESSING = COLLECT_DATA_FOR_POSTPROCESSING

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            return_info: bool = False,
            options: Optional[dict] = None,
            step: Optional[int] = None,
    ):
        super().reset(seed=seed)
        self.simulator = BaseSimulator()
        self.eval_log_step = step

        ###--------------ADD ARM TO SIMULATION--------------###
        # setting up test params
        n_elem = self.n_elem
        start = np.zeros((3,))
        direction = np.array([0.0, 1.0, 0.0])  # rod direction: pointing upwards
        normal = np.array([0.0, 0.0, 1.0])
        binormal = np.cross(direction, normal)

        # Set the arm properties after defining rods

        # Set the arm properties after defining rods
        base_length = self.base_length  # rod base length
        radius_tip = self.radius  # radius of the arm at the tip
        radius_base = self.radius  # radius of the arm at the base
        radius_along_rod = np.linspace(radius_base, radius_tip, n_elem)

        # damping = self.youngs_modulus * 1e-7 * 1
        damping = self.damping

        # Arm is shearable Cosserat rod
        self.shearable_rod = CosseratRod.straight_rod(
            self.n_elem,
            start,
            direction,
            normal,
            base_length,
            base_radius=radius_along_rod,
            density=1000,
            # nu=damping,
            youngs_modulus=self.youngs_modulus,
            shear_modulus=self.shear_modulus,
        )

        # self.shearable_rod.dissipation_constant_for_torques *= (
        #     1e6  # accounts for the new g/mm/s units (compared to kg/m/s)
        # )

        self.simulator.append(
            self.shearable_rod
        )  # Now rod is ready for simulation, append rod to simulation
        self.simulator.constrain(self.shearable_rod).using(
            OneEndFixedRod, constrained_position_idx=(0,), constrained_director_idx=(0,)
        )
        # self.simulator.add_forcing_to(self.shearable_rod).using(GravityForces, acc_gravity=np.array([0.0, 9.81*1e3,
        # 0.0]))

        # add damping
        self.simulator.dampen(self.shearable_rod).using(
            AnalyticalLinearDamper,
            damping_constant=damping,
            time_step=self.sim_dt,
        )

        if self.rest_rod_position_collection is None:
            self.rest_rod_position_collection = self.shearable_rod.position_collection

        if self.mode != 2:
            # fixed target position to reach
            target_position = self.target_position

        if self.mode == 2 or self.mode == 4:
            # random target position to reach with boundary
            t_x = np.random.uniform(self.boundary[0], self.boundary[1])
            t_y = np.random.uniform(self.boundary[2], self.boundary[3])
            if self.dim == 2.0 or self.dim == 2.5:
                t_z = np.random.uniform(self.boundary[4], self.boundary[5]) * 0
            elif self.dim == 3.0 or self.dim == 3.5:
                t_z = np.random.uniform(self.boundary[4], self.boundary[5])

            print("Target position:", t_x, t_y, t_z)
            target_position = np.array([t_x, t_y, t_z])

        # initialize sphere
        self.sphere_radius = 0.05
        self.sphere = Sphere(
            center=target_position,  # initialize target position of the ball
            base_radius=self.sphere_radius,
            density=1000,
        )

        if self.mode == 3:
            self.dir_indicator = 1
            self.sphere_initial_velocity = self.target_v
            self.sphere.velocity_collection[..., 0] = [
                self.sphere_initial_velocity,
                0.0,
                0.0,
            ]

        if self.mode == 4:

            self.trajectory_iteration = 0  # for changing directions
            self.rand_direction_1 = np.pi * np.random.uniform(0, 2)
            if self.dim == 2.0 or self.dim == 2.5:
                self.rand_direction_2 = np.pi / 2.0
            elif self.dim == 3.0 or self.dim == 3.5:
                self.rand_direction_2 = np.pi * np.random.uniform(0, 2)

            self.v_x = (
                    self.target_v
                    * np.cos(self.rand_direction_1)
                    * np.sin(self.rand_direction_2)
            )
            self.v_y = (
                    self.target_v
                    * np.sin(self.rand_direction_1)
                    * np.sin(self.rand_direction_2)
            )
            self.v_z = self.target_v * np.cos(self.rand_direction_2)

            self.sphere.velocity_collection[..., 0] = [
                self.v_x,
                self.v_y,
                self.v_z,
            ]
            self.boundaries = np.array(self.boundary)

        # set the orientation of target sphere
        if self.mode == 1:
            targete_orientation = self.target_orientation
            rotation_matrix = R.from_quat(targete_orientation).as_matrix()
        if self.mode == 2 or self.mode == 4:
            theta_x = 0
            theta_y = np.random.uniform(-np.pi / 2, np.pi / 2)
            theta_z = 0
            theta = np.array([theta_x, theta_y, theta_z])
            # Convert to rotation matrix
            rotation_matrix = np.array(
                [
                    [
                        -np.sin(theta[1]),
                        np.sin(theta[0]) * np.cos(theta[1]),
                        np.cos(theta[0]) * np.cos(theta[1]),
                    ],
                    [
                        np.cos(theta[1]) * np.cos(theta[2]),
                        np.sin(theta[0]) * np.sin(theta[1]) * np.cos(theta[2])
                        - np.sin(theta[2]) * np.cos(theta[0]),
                        np.sin(theta[1]) * np.cos(theta[0]) * np.cos(theta[2])
                        + np.sin(theta[0]) * np.sin(theta[2]),
                    ],
                    [
                        np.sin(theta[2]) * np.cos(theta[1]),
                        np.sin(theta[0]) * np.sin(theta[1]) * np.sin(theta[2])
                        + np.cos(theta[0]) * np.cos(theta[2]),
                        np.sin(theta[1]) * np.sin(theta[2]) * np.cos(theta[0])
                        - np.sin(theta[0]) * np.cos(theta[2]),
                    ],
                ]
            )
            self.target_orientation = R.from_matrix(rotation_matrix).as_quat()

        self.sphere.director_collection[..., 0] = rotation_matrix
        self.simulator.append(self.sphere)

        class WallBoundaryForSphere(FreeRod):
            """

            This class generates a bounded space that sphere can move inside. If sphere
            hits one of the boundaries (walls) of this space, it is reflected in opposite direction
            with the same velocity magnitude.

            """

            def __init__(self, boundaries):
                self.x_boundary_low = boundaries[0]
                self.x_boundary_high = boundaries[1]
                self.y_boundary_low = boundaries[2]
                self.y_boundary_high = boundaries[3]
                self.z_boundary_low = boundaries[4]
                self.z_boundary_high = boundaries[5]

            def constrain_values(self, sphere, time):
                pos_x = sphere.position_collection[0]
                pos_y = sphere.position_collection[1]
                pos_z = sphere.position_collection[2]

                radius = sphere.radius

                vx = sphere.velocity_collection[0]
                vy = sphere.velocity_collection[1]
                vz = sphere.velocity_collection[2]

                if (pos_x - radius) < self.x_boundary_low:
                    sphere.velocity_collection[:] = np.array([-vx, vy, vz])

                if (pos_x + radius) > self.x_boundary_high:
                    sphere.velocity_collection[:] = np.array([-vx, vy, vz])

                if (pos_y - radius) < self.y_boundary_low:
                    sphere.velocity_collection[:] = np.array([vx, -vy, vz])

                if (pos_y + radius) > self.y_boundary_high:
                    sphere.velocity_collection[:] = np.array([vx, -vy, vz])

                if (pos_z - radius) < self.z_boundary_low:
                    sphere.velocity_collection[:] = np.array([vx, vy, -vz])

                if (pos_z + radius) > self.z_boundary_high:
                    sphere.velocity_collection[:] = np.array([vx, vy, -vz])

            def constrain_rates(self, sphere, time):
                pass

        if self.mode == 4:
            self.simulator.constrain(self.sphere).using(
                WallBoundaryForSphere, boundaries=self.boundaries
            )

        # Add muscle torques acting on the arm for actuation
        # MuscleTorquesWithVaryingBetaSplines uses the control points selected by RL to
        # generate torques along the arm.
        self.torque_profile_list_for_muscle_in_normal_dir = defaultdict(list)
        self.spline_points_func_array_normal_dir = []
        # Apply torques
        self.simulator.add_forcing_to(self.shearable_rod).using(
            MuscleTorquesWithVaryingBetaSplines,
            base_length=self.base_length,
            number_of_control_points=self.number_of_control_points,
            points_func_array=self.spline_points_func_array_normal_dir,
            muscle_torque_scale=self.alpha,
            direction=str("normal"),
            step_skip=self.step_skip,
            max_rate_of_change_of_activation=self.max_rate_of_change_of_activation,
            torque_profile_recorder=self.torque_profile_list_for_muscle_in_normal_dir,
        )

        self.torque_profile_list_for_muscle_in_binormal_dir = defaultdict(list)
        self.spline_points_func_array_binormal_dir = []
        # Apply torques
        self.simulator.add_forcing_to(self.shearable_rod).using(
            MuscleTorquesWithVaryingBetaSplines,
            base_length=self.base_length,
            number_of_control_points=self.number_of_control_points,
            points_func_array=self.spline_points_func_array_binormal_dir,
            muscle_torque_scale=self.alpha,
            direction=str("binormal"),
            step_skip=self.step_skip,
            max_rate_of_change_of_activation=self.max_rate_of_change_of_activation,
            torque_profile_recorder=self.torque_profile_list_for_muscle_in_binormal_dir,
        )

        self.torque_profile_list_for_muscle_in_twist_dir = defaultdict(list)
        self.spline_points_func_array_twist_dir = []
        # Apply torques
        self.simulator.add_forcing_to(self.shearable_rod).using(
            MuscleTorquesWithVaryingBetaSplines,
            base_length=self.base_length,
            number_of_control_points=self.number_of_control_points,
            points_func_array=self.spline_points_func_array_twist_dir,
            muscle_torque_scale=self.beta,
            direction=str("tangent"),
            step_skip=self.step_skip,
            max_rate_of_change_of_activation=self.max_rate_of_change_of_activation,
            torque_profile_recorder=self.torque_profile_list_for_muscle_in_twist_dir,
        )

        # Call back function to collect arm data from simulation
        class ArmMuscleBasisCallBack(CallBackBaseClass):
            """
            Call back function for Elastica rod
            """

            def __init__(
                    self, step_skip: int, callback_params: dict,
            ):
                CallBackBaseClass.__init__(self)
                self.every = step_skip
                self.callback_params = callback_params

            def make_callback(self, system, time, current_step: int):
                if current_step % self.every == 0:
                    self.callback_params["time"].append(time)
                    self.callback_params["step"].append(current_step)
                    self.callback_params["position"].append(
                        system.position_collection.copy()
                    )
                    self.callback_params["directors"].append(
                        system.director_collection.copy()
                    )
                    self.callback_params["radius"].append(system.radius.copy())
                    self.callback_params["com"].append(
                        system.compute_position_center_of_mass()
                    )

                    return

        # Call back function to collect target sphere data from simulation
        class RigidSphereCallBack(CallBackBaseClass):
            """
            Call back function for target sphere
            """

            def __init__(self, step_skip: int, callback_params: dict):
                CallBackBaseClass.__init__(self)
                self.every = step_skip
                self.callback_params = callback_params

            def make_callback(self, system, time, current_step: int):
                if current_step % self.every == 0:
                    self.callback_params["time"].append(time)
                    self.callback_params["step"].append(current_step)
                    self.callback_params["position"].append(
                        system.position_collection.copy()
                    )
                    self.callback_params["directors"].append(
                        system.director_collection.copy()
                    )
                    self.callback_params["radius"].append(copy.deepcopy(system.radius))
                    self.callback_params["com"].append(
                        system.compute_position_center_of_mass()
                    )

                    return

        if self.COLLECT_DATA_FOR_POSTPROCESSING:
            # Collect data using callback function for postprocessing
            self.post_processing_dict_rod = defaultdict(list)
            # List which collected data will be append
            # set the diagnostics for rod and collect data
            self.simulator.collect_diagnostics(self.shearable_rod).using(
                ArmMuscleBasisCallBack,
                step_skip=self.step_skip,
                callback_params=self.post_processing_dict_rod,
            )

            self.post_processing_dict_sphere = defaultdict(list)
            # List which collected data will be append
            # set the diagnostics for target sphere and collect data
            self.simulator.collect_diagnostics(self.sphere).using(
                RigidSphereCallBack,
                step_skip=self.step_skip,
                callback_params=self.post_processing_dict_sphere,
            )

        ###--------------FINALIZE SIMULATION--------------###
        # Finalize simulation environment. After finalize, you cannot add
        # any forcing, constrain or call back functions
        self.simulator.finalize()

        # do_step, stages_and_updates will be used in step function
        self.do_step, self.stages_and_updates = extend_stepper_interface(
            self.StatefulStepper, self.simulator
        )
        # set state
        state = self.get_state()

        # reset current_step
        self.current_step = 0
        # reset time_tracker
        self.time_tracker = np.float64(0.0)
        # reset previous_action
        self.previous_action = None

        self._target = self.sphere.position_collection[..., 0]

        return state, {}

    def get_state(self):
        """
        Returns current state of the system to the controller.

        Returns
        -------
        numpy.ndarray
            1D (number_of_states) array containing data with 'float' type.
            Size of the states depends on the problem.
        """
        num_points = int(self.n_elem / self.obs_state_points)

        # get full 3D position information
        rod_state = self.shearable_rod.position_collection
        rod_compact_state = rod_state.transpose()[0: len(rod_state[0]) + 1: num_points].flatten()

        # get full 3D velocity information
        rod_velocity = self.shearable_rod.velocity_collection
        rod_compact_velocity = rod_velocity.transpose()[0: len(rod_velocity[0]) + 1: num_points].flatten()

        # get full 3D euler angle information
        rod_director = self.shearable_rod.director_collection
        rod_quad_list = []
        for i in range(self.obs_state_points):
            director = rod_director[..., num_points * (i + 1) - 1]
            elem_quad = R.from_matrix(director).as_quat()
            rod_quad_list.append(elem_quad)
        rod_compact_quad = np.array(rod_quad_list).flatten()

        # get full 3D anguler velocity information
        rod_angular_velocity = self.shearable_rod.omega_collection
        rod_compact_angular_velocity = rod_angular_velocity.transpose()[
                                       num_points - 1: len(rod_angular_velocity[0]) + 1: num_points].flatten()

        state = np.concatenate(
            (
                # rod information
                rod_compact_state,
                rod_compact_velocity,
                rod_compact_quad,
                rod_compact_angular_velocity,
            )
        )

        return state

    def step(self, action):
        # action contains the control points for actuation torques in different directions in range [-1, 1]
        self.action = action

        # set binormal activations to 0 if solving 2D case
        if self.dim == 2.0:
            self.spline_points_func_array_normal_dir[:] = action[
                                                          : self.number_of_control_points
                                                          ]
            self.spline_points_func_array_binormal_dir[:] = (
                    action[: self.number_of_control_points] * 0.0
            )
            self.spline_points_func_array_twist_dir[:] = (
                    action[: self.number_of_control_points] * 0.0
            )
        elif self.dim == 2.5:
            self.spline_points_func_array_normal_dir[:] = action[
                                                          : self.number_of_control_points
                                                          ]
            self.spline_points_func_array_binormal_dir[:] = (
                    action[: self.number_of_control_points] * 0.0
            )
            self.spline_points_func_array_twist_dir[:] = action[
                                                         self.number_of_control_points:
                                                         ]
        # apply binormal activations if solving 3D case
        elif self.dim == 3.0:
            self.spline_points_func_array_normal_dir[:] = action[
                                                          : self.number_of_control_points
                                                          ]
            self.spline_points_func_array_binormal_dir[:] = action[
                                                            self.number_of_control_points:
                                                            ]
            self.spline_points_func_array_twist_dir[:] = (
                    action[: self.number_of_control_points] * 0.0
            )
        elif self.dim == 3.5:
            self.spline_points_func_array_normal_dir[:] = action[
                                                          : self.number_of_control_points
                                                          ]
            self.spline_points_func_array_binormal_dir[:] = action[
                                                            self.number_of_control_points: 2 * self.number_of_control_points
                                                            ]
            self.spline_points_func_array_twist_dir[:] = action[
                                                         2 * self.number_of_control_points:
                                                         ]

        # Do multiple time step of simulation for <one learning step>
        for _ in range(self.num_steps_per_update):
            self.time_tracker = self.do_step(
                self.StatefulStepper,
                self.stages_and_updates,
                self.simulator,
                self.time_tracker,
                self.time_step,
            )

        if self.mode == 3:
            ##### (+1, 0, 0) -> (0, -1, 0) -> (-1, 0, 0) -> (0, +1, 0) -> (+1, 0, 0) #####
            if (
                    self.current_step
                    % (1.0 / (self.h_time_step * self.num_steps_per_update))
                    == 0
            ):
                if self.dir_indicator == 1:
                    self.sphere.velocity_collection[..., 0] = [
                        0.0,
                        -self.sphere_initial_velocity,
                        0.0,
                    ]
                    self.dir_indicator = 2
                elif self.dir_indicator == 2:
                    self.sphere.velocity_collection[..., 0] = [
                        -self.sphere_initial_velocity,
                        0.0,
                        0.0,
                    ]
                    self.dir_indicator = 3
                elif self.dir_indicator == 3:
                    self.sphere.velocity_collection[..., 0] = [
                        0.0,
                        +self.sphere_initial_velocity,
                        0.0,
                    ]
                    self.dir_indicator = 4
                elif self.dir_indicator == 4:
                    self.sphere.velocity_collection[..., 0] = [
                        +self.sphere_initial_velocity,
                        0.0,
                        0.0,
                    ]
                    self.dir_indicator = 1
                else:
                    print("ERROR")

        if self.mode == 4:
            self.trajectory_iteration += 1
            if self.trajectory_iteration == 500:
                # print('changing direction')
                self.rand_direction_1 = np.pi * np.random.uniform(0, 2)
                if self.dim == 2.0 or self.dim == 2.5:
                    self.rand_direction_2 = np.pi / 2.0
                elif self.dim == 3.0 or self.dim == 3.5:
                    self.rand_direction_2 = np.pi * np.random.uniform(0, 2)

                self.v_x = (
                        self.target_v
                        * np.cos(self.rand_direction_1)
                        * np.sin(self.rand_direction_2)
                )
                self.v_y = (
                        self.target_v
                        * np.sin(self.rand_direction_1)
                        * np.sin(self.rand_direction_2)
                )
                self.v_z = self.target_v * np.cos(self.rand_direction_2)

                self.sphere.velocity_collection[..., 0] = [
                    self.v_x,
                    self.v_y,
                    self.v_z,
                ]
                self.trajectory_iteration = 0

        self.current_step += 1

        # observe current state: current as sensed signal
        state = self.get_state()

        """ Done is a boolean to reset the environment before episode is completed """
        terminate = False
        truncate = False

        reward = self.reward_model.predict(obs=state, action=action, next_obs=state)
        reward = reward.astype(float).item()

        # Position of the rod cannot be NaN, it is not valid, stop the simulation
        # self.shearable_rod.position_collection = self.rest_rod_position_collection
        invalid_values_condition = _isnan_check(self.shearable_rod.position_collection)

        if invalid_values_condition:
            print(" Nan detected, exiting simulation now")
            print(state)
            # self.shearable_rod.position_collection = np.zeros(
            #     self.shearable_rod.position_collection.shape
            # )
            self.shearable_rod.position_collection = self.rest_rod_position_collection
            action = np.zeros(
                action.shape
            )
            reward = -1000
            state = self.get_state()
            print('current step: ' + str(self.current_step))
            terminate = True

        """ Done is a boolean to reset the environment before episode is completed """

        self.previous_action = action
        self._target = self.sphere.position_collection[..., 0]
        return state, reward, terminate, truncate, {"ctime": self.time_tracker}

    def render(self):
        pass
        # if self.render_mode == 'rgb_array':
        #     maxwidth = 800
        #     aspect_ratio = 3 / 4
        #     if self.renderer is None:
        #         from mbse.envs.utils.matplotlib_renderer import Session

        #         assert issubclass(
        #             Session, BaseRenderer
        #         ), "Rendering module is not properly subclassed"
        #         assert issubclass(
        #             Session, BaseElasticaRendererSession
        #         ), "Rendering module is not properly subclassed"
        #         self.renderer = Session(width=maxwidth, height=int(maxwidth * aspect_ratio))
        #         self.renderer.add_rod(self.shearable_rod)
        #         self.renderer.add_point(self._target.tolist(), self.sphere_radius)

        #     state_image = self.renderer.render()
        #     self.renderer.close()
        #     self.renderer = None
        #     return state_image
        # else:
        #     raise NotImplementedError("Rendering mode is not implemented")

    def post_processing(self, video_dir, filename_video, SAVE_DATA=False, **kwargs):
        """
        Make video 3D of arm movement in time, and store the arm, target, obstacles, and actuation
        data.

        Parameters
        ----------
        filename_video : str
            Names of the videos to be made for post-processing.
        SAVE_DATA : boolean
            If true collected data in simulation saved.
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------

        """

        if self.COLLECT_DATA_FOR_POSTPROCESSING:
            # plot_video_with_sphere_2D(
            #     [self.post_processing_dict_rod],
            #     [self.post_processing_dict_sphere],
            #     video_name=video_dir + "/2d_" + filename_video + '.mp4',
            #     fps=self.rendering_fps,
            #     step=1,
            #     vis2D=False,
            #     **kwargs,
            # )

            # plot_video_with_sphere(
            #     [self.post_processing_dict_rod],
            #     [self.post_processing_dict_sphere],
            #     video_name=video_dir + "/3d_" + filename_video + '.mp4',
            #     fps=self.rendering_fps,
            #     step=1,
            #     vis2D=False,
            #     **kwargs,
            # )

            if SAVE_DATA == True:
                import os
                # Transform nodal to elemental positions
                position_rod = np.array(self.post_processing_dict_rod["position"])
                position_rod = 0.5 * (position_rod[..., 1:] + position_rod[..., :-1])

                np.savez(
                    os.path.join(video_dir, filename_video + "_arm_data.npz"),
                    position_rod=position_rod,
                    radii_rod=np.array(self.post_processing_dict_rod["radius"]),
                    n_elems_rod=self.shearable_rod.n_elems,
                    position_sphere=np.array(
                        self.post_processing_dict_sphere["position"]
                    ),
                    radii_sphere=np.array(self.post_processing_dict_sphere["radius"]),
                )

                np.savez(
                    os.path.join(video_dir, filename_video + "_arm_activation.npz"),
                    torque_mag=np.array(
                        self.torque_profile_list_for_muscle_in_normal_dir["torque_mag"]
                    ),
                    torque_muscle=np.array(
                        self.torque_profile_list_for_muscle_in_normal_dir["torque"]
                    ),
                )

        else:
            raise RuntimeError(
                "call back function is not called anytime during simulation, "
                "change COLLECT_DATA=True"
            )

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
        if self.renderer:
            self.renderer.close()
            self.renderer = None


if __name__ == '__main__':
    env = SoftArmReachOrienEnv(integration_freq=2.0e-4)
    obs, _ = env.reset(seed=0)

    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminate, truncate, info = env.step(action)
        if terminate:
            obs, _ = env.reset()
        env.render()


