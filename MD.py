import numpy as np
import pandas as pd
from math import ceil
from tqdm import tqdm
import os

ELEMENTS = {
    'He': {
        'eps': 6.03,
        'sig': 2.63,
        'm': 6.67,
        'r': 1.4
    },
    'H2': {
        'eps': 29.2,
        'sig': 2.87,
        'm': 1.71,
        'r': 1.2
    },
    'Ne': {
        'eps': 35.6,
        'sig': 2.75,
        'm': 33.55,
        'r': 1.5
    },
    'N2': {
        'eps': 95.05,
        'sig': 3.69,
        'm': 46.43,
        'r': 1.6
    },
    'O2': {
        'eps': 99.2,
        'sig': 3.52,
        'm': 26.68,
        'r': 1.5
    }
}

class MD:
    def __init__(self, element, natoms, t_start, t_end, dt, filename):
        self.m = element['m']
        self.r = element['r']
        self.sig = element['sig']
        self.eps = element['eps']
        self.natoms = natoms
        self.t_start = t_start
        self.t_end = t_end
        self.dt = dt
        self.filename = filename
        self.pos = np.zeros((natoms, 2))
        self.vel = np.zeros((natoms, 2))
        self.force = np.zeros((natoms, 2))
        self.E_pot = np.zeros((natoms, 1))
        self.E_kin = np.zeros((natoms, 1))
        self.Rc = 2.5 * self.sig
        self.LJP_Rc = self.ljp(self.Rc)

    def dump(self, timestamp):
        with open(self.filename+'.dump', 'a') as fp:
            fp.write(f'''ITEM: TIMESTEP
{timestamp}
ITEM: NUMBER OF ATOMS
{self.natoms}
ITEM: BOX BOUNDS p p p
{-self.box_size} {self.box_size}
{-self.box_size} {self.box_size}
0 0
ITEM: ATOMS radius type x y v_x v_y
''')
            output = np.ones((self.natoms, 1)) * self.r
            output = np.hstack((output, np.ones((self.natoms, 1))))
            # output = np.hstack((output, np.vstack((np.ones((self.natoms//2, 1)), np.ones((self.natoms-self.natoms//2, 1)) + 1))))
            output = np.hstack((output, self.pos))
            output = np.hstack((output, self.vel))
            np.savetxt(fp, output.reshape((self.natoms, 6), order='F'))

    def csv(self, timestamp):
        data = pd.DataFrame(index = np.ones(self.natoms)*timestamp)
        data['id'] = range(self.natoms)
        data[['x', 'y']] = self.pos
        data[['vx', 'vy']] = self.vel
        data[['fx', 'fy']] = self.force
        data['E_pot'] = self.E_pot
        data['E_kin'] = self.E_kin
        if not os.path.exists(self.filename+'.csv'):
            data.to_csv(self.filename+'.csv')
        else:
            data.to_csv(self.filename+'.csv', mode='a', header=False)
    
    def generate_pos(self):
        natoms = self.natoms
        
        num_x = ceil(np.sqrt(natoms))
        num_y = ceil(natoms / num_x)

        box_size = self.sig * np.max([num_x, num_y]) * 0.75
        self.box_size = box_size
        box_size = self.box_size - self.r
        x = np.linspace(-box_size, box_size, num_x)
        y = np.linspace(-box_size, box_size, num_y)

        pos = np.stack(np.meshgrid(x, y), -1).reshape(-1, 2)[0:natoms]
        return pos

    def generate_vel(self):
        std_dev = np.sqrt(self.t_start/self.m)
        mean = 0
        vel = np.random.normal(loc=mean, scale=std_dev, size=(self.natoms, 2))
        return vel
    
    def compute_E_kin(self):
        self.E_kin = 0.5 * self.m * (self.vel[:, 0] ** 2 + self.vel[:, 1] ** 2)

    def walls(self):
        tlprt_dist = self.box_size * 2
        for i in range(2):
            self.pos[self.pos[:, i] <= -self.box_size, i] += tlprt_dist
            self.pos[self.pos[:, i] >=  self.box_size, i] -= tlprt_dist

    def ljp(self, r):
        return 4 * self.eps * ((self.sig/r)**12 - (self.sig/r)**6)

    def compute_force(self, k = 1):
        pos = self.pos
        self.force *= 0
        tlprt_dist = self.box_size * 2
        teleports = [
            np.array([          0,           0]),
            np.array([ tlprt_dist,           0]),
            np.array([-tlprt_dist,           0]),
            np.array([          0,  tlprt_dist]),
            np.array([          0, -tlprt_dist]),
        ]
        for i in range(self.natoms):
            for j in range(self.natoms):
                if i == j:
                    continue
                for tlprt in teleports:
                    Rij = pos[i, :] - (pos[j, :] + tlprt)
                    Rij_module = np.sqrt(np.dot(Rij, Rij))
                    if Rij_module < self.Rc:
                        E_pot = k * (self.ljp(Rij_module) - self.LJP_Rc)
                        self.E_pot[i] += E_pot
                        self.force[i] += E_pot * Rij / (Rij_module**2)

    def varlet(self, k):
        T = self.E_kin.sum() / self.natoms
        # half_vel = self.vel + (self.force * self.dt * 0.5 / self.m)
        half_vel = np.sqrt(self.t_end/T) * self.vel + (self.force * self.dt * 0.5 / self.m)
        self.pos += half_vel * self.dt
        self.compute_force(k)
        self.vel = half_vel + (self.force * self.dt * 0.5 / self.m)
        self.compute_E_kin()

    def run(self, steps, k):
        self.pos = self.generate_pos() # Generate positions
        self.walls() # Check walls
        self.vel = self.generate_vel() # Generate velocities according to Maxwell`s Distribution
        self.compute_E_kin() # Compute Kinetic Energy
        print(f"T_start_input = {self.t_start} | E_kin_fact = {self.E_kin.sum():0.2f} | T_start_fact = {self.E_kin.sum()/self.natoms:0.2f}")
        self.compute_force(k) # Compute Forces
        self.dump(0)
        self.csv(0)
        for step in tqdm(range(steps)):
            self.varlet(k)
            self.walls()
            self.dump(step+1)
            self.csv(step+1)

    def run_continue(self, steps, k, file):
        self.pos = self.generate_pos()
        df = pd.read_csv(file).rename(columns={'Unnamed: 0': 'time'})
        max_step = df['time'].max()
        df = df.loc[df['time'] == max_step]
        self.pos = np.array(df.loc[:, ['x', 'y']])
        self.vel = np.array(df.loc[:, ['vx', 'vy']])
        self.force = np.array(df.loc[:, ['fx', 'fy']])
        self.E_kin = np.array(df.loc[:, 'E_kin'])
        for step in tqdm(range(int(max_step), int(max_step) + steps)):
            self.varlet(k)
            self.walls()
            self.dump(step+1)
            self.csv(step+1)



if __name__ == '__main__':

    for key, element in ELEMENTS.items():
        for T in [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600]:
            print(f"{key}: T = {T}")
            filepath = f'output/{key}/{key}_T{T}'

            params = {
                'element': element,
                'natoms': 100,
                't_start': T,
                't_end': T,
                'dt': 0.001,
                'filename': filepath
            }

            dump = filepath+'.dump'
            csv = filepath+'.csv'
            k = 2
            md = MD(**params)

        # Start

            # if os.path.exists(dump):
            #     os.remove(dump)
            # if os.path.exists(csv):
            #     os.remove(csv)
            
            # md.run(steps = 500, k = k)

        # Continue

            md.run_continue(steps = 1000, k = k, file = csv)