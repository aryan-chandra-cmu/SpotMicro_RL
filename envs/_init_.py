from gymnasium.envs.registration import register

register(id="SpotMicroWalkMJ-v0", entry_point="envs.mj_walk_spotmicro:SpotMicroWalkMJ")
register(id="SpotMicroTurnMJ-v0", entry_point="envs.mj_turn_spotmicro:SpotMicroTurnMJ")
self._mj_viewer = None