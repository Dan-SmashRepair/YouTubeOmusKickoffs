from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket
import numpy as np

from agent_omus import Agent_Omus
from obs.advanced_obs import AdvancedObs
from action.discrete_act import DiscreteAction
from rlgym_compat import GameState


class Omus(BaseAgent):
    def __init__(self, name, team, index):
        super().__init__(name, team, index)

        self.omus_obs_builder = AdvancedObs()
        self.omus_act_parser = DiscreteAction()
        self.agent_omus = Agent_Omus()
        self.omus_tick_skip = 6

        self.game_state: GameState = None
        self.controls = None
        self.action = None
        self.update_action = True
        self.ticks = 0
        self.prev_time = 0
        self.gamemode ='fiftyfifty'
        print('Omus Ready - Index:', index)


    def initialize_agent(self):
        # Initialize the rlgym GameState object now that the game is active and the info is available
        self.game_state = GameState(self.get_field_info())
        self.ticks = self.omus_tick_skip  # So we take an action the first tick
        self.prev_time = 0
        self.controls = SimpleControllerState()
        self.action = np.zeros(8)
        self.update_action = True
        self.ko_diag_array = np.array([
        [1, 0, 0, 0, 0,0,1,0], #0
        [1, 0, 0, 0, 0,0,1,0],
        [1, 0, 0, 0, 0,0,1,0],
        [1, 0, 0, 0, 0,0,1,0],
        [1, 0,-1, 0, 1,0,1,0],
        [1, 0,-1, 0, 1,0,1,0],
        [1, 0,-1, 0, 1,0,1,0],
        [1,-1,-1, 0, 1,0,1,0],
        [1,-1,-1,-1, 1,1,1,0],
        [1, 0,-1,-1, 1,0,1,0],
        [1, 0,-1, 0, 1,1,1,0], #10
        [1, 0, 1, 0, 1,0,1,0],
        [1, 0, 1, 0, 1,0,1,0],
        [1, 0, 1, 0, 1,0,1,0],
        [1, 0, 1, 0, 1,0,1,0],
        [1, 0, 1, 0, 1,0,1,0],
        [1, 0, 1, 0, 1,0,1,0],
        [1, 0, 1, 0, 1,0,1,0],
        [1, 0, 1, 0, 1,0,1,0],
        [1, 0, 1, 0, 1,0,1,0],
        [1, 0, 1, 1, 1,0,1,0], #20
        [1, 0, 1, 1, 1,0,1,0],
        [1, 0, 1, 1, 1,0,1,0],
        [1, 0, 1, 1, 1,0,1,0],
        [1, 0, 1, 1, 1,0,0,0],
        [1, 0, 1, 1, 1,0,0,0],
        [1, 0, 1, 1, 1,0,0,0],
        [1, 0, 0, 0, 0,0,0,0],
        [1, 0, 0,-1, 0,0,0,0],
        [1,-1, 0,-1, 0,0,0,0],
        [1,-1, 0,-1, 0,0,0,0], #30
        [1,-1, 0, 0, 0,0,0,0],
        [1, 0, 0, 0, 0,0,0,0],
    ])
        self.kickoff_time = 0
        self.ticks2 = -1
        self.ko_spawn_pos = 'Center'


    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        # To add Omus kickoffs to your bot, add below code under a simple if statement like the following:
        # if packet.game_ball.physics.location.x == 0 and packet.game_ball.physics.location.y == 0:
        # else: 'your bot's code'
        cur_time = packet.game_info.seconds_elapsed
        delta = cur_time - self.prev_time
        self.prev_time = cur_time

        ticks_elapsed = round(delta * 120)
        self.ticks += ticks_elapsed
        self.game_state.decode(packet, ticks_elapsed)
        self.ticks2 += 1

        if self.update_action:
            self.update_action = False

            # FIXME Hey, botmaker. Verify that this is what you need for your agent
            # By default we treat every match as a 1v1 against a fixed opponent,
            # by doing this your bot can participate in 2v2 or 3v3 matches. Feel free to change this
            player = self.game_state.players[self.index]
            teammates = [p for p in self.game_state.players if p.team_num == self.team]
            opponents = [p for p in self.game_state.players if p.team_num != self.team]

            if len(opponents) == 0:
                # There's no opponent, we assume this model is 1v0
                self.game_state.players = [player]
            else:
                # Sort by distance to ball
                teammates.sort(key=lambda p: np.linalg.norm(self.game_state.ball.position - p.car_data.position))
                opponents.sort(key=lambda p: np.linalg.norm(self.game_state.ball.position - p.car_data.position))

                # Grab opponent in same "position" relative to it's teammates
                opponent = opponents[min(teammates.index(player), len(opponents) - 1)]

                self.game_state.players = [player, opponent]

            obs = self.omus_obs_builder.build_obs(player, self.game_state, self.action)
            self.action = self.omus_act_parser.parse_actions(self.agent_omus.act(obs, self.gamemode), self.game_state)[0]  # Dim is (N, 8)

        if self.ticks >= self.omus_tick_skip - 1:
            self.update_controls(self.action)

        if self.ticks >= self.omus_tick_skip:
            self.ticks = 0
            self.update_action = True
        
        # substitute fiftyfifty or kickoff model based on spawn
        if abs(self.game_state.players[self.team].car_data.position[0]) <= 2 and 998 <= abs(self.game_state.players[self.team].car_data.position[1]) <= 1002 and abs(self.game_state.players[self.team].car_data.linear_velocity[0]) <= 30:
            self.gamemode = 'fiftyfifty'
        if 2046 <= abs(self.game_state.players[self.team].car_data.position[0]) <= 2050 and 2558 <= abs(self.game_state.players[self.team].car_data.position[1]) <= 2562 and abs(self.game_state.players[self.team].car_data.linear_velocity[0]) <= 30:
            self.kickoff_time = self.ticks2
            self.gamemode = 'kickoff'
            if self.game_state.players[0].car_data.position[0] > 0:
                self.ko_spawn_pos = 'Diagonal L'
            elif self.game_state.players[0].car_data.position[0] < 0:
                self.ko_spawn_pos = 'Diagonal R'
        elif 254 <= abs(self.game_state.players[self.team].car_data.position[0]) <= 258 and 3838 <= abs(self.game_state.players[self.team].car_data.position[1]) <= 3842 and abs(self.game_state.players[self.team].car_data.linear_velocity[0]) <= 30:
            self.kickoff_time = self.ticks2
            self.gamemode = 'kickoff'
            if self.game_state.players[0].car_data.position[0] > 0:
                self.ko_spawn_pos = 'Offset L'
            elif self.game_state.players[0].car_data.position[0] < 0:
                self.ko_spawn_pos = 'Offset R'
        elif abs(self.game_state.players[self.team].car_data.position[0]) <= 2 and 4606 <= abs(self.game_state.players[self.team].car_data.position[1]) <= 4610 and abs(self.game_state.players[self.team].car_data.linear_velocity[0]) <= 30:
            self.kickoff_time = self.ticks2
            self.gamemode = 'kickoff'
            self.ko_spawn_pos = 'Center'
        # counter-fake kickoffs
        step_20hz = int(np.floor((self.ticks2-self.kickoff_time)/6))
        if self.ko_spawn_pos == 'Diagonal L':
            if step_20hz <= 30:
                self.update_controls(self.ko_diag_array[step_20hz])
        elif self.ko_spawn_pos == 'Center':
            if 25 <= step_20hz <= 35:
                self.controls.handbrake = 1
        if np.linalg.norm(self.game_state.ball.position - np.zeros(3)) < 1050:
            if (step_20hz <= 78 and (self.ko_spawn_pos == 'Diagonal L' or self.ko_spawn_pos == 'Diagonal R')) or\
                (step_20hz <= 85 and (self.ko_spawn_pos != 'Diagonal L' or self.ko_spawn_pos != 'Diagonal R')):
                if np.linalg.norm(self.game_state.ball.position - self.game_state.players[1-self.team].car_data.position) - np.linalg.norm(self.game_state.ball.position - self.game_state.players[self.team].car_data.position) > 400:
                    self.controls.boost = 0
                    if step_20hz >= 29:
                        self.gamemode = 'fiftyfifty'
                    if np.linalg.norm(self.game_state.ball.position - self.game_state.players[1-self.team].car_data.position) - np.linalg.norm(self.game_state.ball.position - self.game_state.players[self.team].car_data.position) > 800:
                        if 800 > np.linalg.norm(self.game_state.ball.position - self.game_state.players[self.team].car_data.position):
                            if abs(np.linalg.norm(self.game_state.players[self.team].car_data.linear_velocity)) > 700:
                                self.controls.throttle = -1
                        if abs(np.linalg.norm(self.game_state.players[self.team].car_data.linear_velocity)) < 500:
                            self.controls.throttle = 1
        return self.controls


    def update_controls(self, action):
        self.controls.throttle = action[0]
        self.controls.steer = action[1]
        self.controls.pitch = action[2]
        self.controls.yaw = action[3]
        self.controls.roll = action[4]
        self.controls.jump = action[5] > 0
        self.controls.boost = action[6] > 0
        self.controls.handbrake = action[7] > 0
