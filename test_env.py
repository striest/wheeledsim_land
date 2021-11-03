import pybullet as p

from wheeledSim.terrain.randomTerrain import RacetrackTerrain
from wheeledSim.rosSimController import rosSimController

if __name__ == '__main__':
    sim = rosSimController('configs/frontcam_racetrack.yaml')

    x = input('racetrack!')
