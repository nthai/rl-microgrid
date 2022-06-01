import numpy as np
import matplotlib.pyplot as plt

class Battery():
    
    def __init__(self, action_size, scaler_energy, scaler_price):
        """
        P_rated - charge/ discharge rate (kW)
        E_rated - rated capacity (kWh)
        C_E - energy capital cost ($/kWh)
        LC - life cycle
        eta - efficiency
        DOD - depth of discharge
        wear_cost - wear & operation cost ($/kWh/operation)
        wear_cost = (C_E * E_rated) / (eta * E_rated * LC * DOD)
        a1, m1, m2_c, m2_d: multiplier for energy gain
        """ 
        self.P_rated = scaler_energy.transform(np.array([[1000]]))[0] # pu
        self.E_rated = scaler_energy.transform(np.array([[5000]]))[0] # pu 
        self.C_E = scaler_price.transform(np.array([[171]]))[0] # pu
        self.LC = 4996
        self.eta = 1.
        self.DOD = 1.
        self.wear_cost = self.C_E / self.eta / self.DOD / self.LC
        self.action_set = np.linspace(-1, 1, num = action_size, endpoint = True)
        self.initial_SOC = 0.
        self.target_SOC = 0.5 # Decide the backup energy required
        self.a1 = 4.
        self.m1 = 2
        self.m2_c = 0.9
        self.m2_d = 1.2
        self.m2_d2 = 1
        self.plot_multiplier()
    
    def compute(self, state, action):
        current_pv = state[0]
        current_load = state[1]
        current_price = state[2]
        current_SOC = state[3]
        average_price = state[4]

        infos = None
        if __debug__:
            infos = dict()

        # Update SOC level
        # Assign penalty if battery operates exceeding permissible limit
        delta_SOC = self.action_set[action] * self.P_rated / self.E_rated
        penalty = 0
        if self.action_set[action] <= 0:
            next_SOC = np.maximum(0., current_SOC + delta_SOC)
            if abs(delta_SOC) > current_SOC:
                penalty = 10
        else:
            next_SOC = np.minimum(1., current_SOC + delta_SOC)
            if abs(delta_SOC) > 1 - current_SOC:
                penalty = 10
    
        # Assign penalty to force containment of PV within the building
        penalty_pv = 0
        if current_pv > (current_load + (next_SOC - current_SOC) * self.E_rated):
            if next_SOC != 1:        
                penalty_pv = np.exp(2.5 * (1 - next_SOC)) - 1

        # Compute piecewise multiplier
        if next_SOC < self.target_SOC: # Before target SOC is met
            multiplier = 1 + self.a1 * np.exp(-(-np.log((self.m1 - 1) / self.a1)) * \
                next_SOC / self.target_SOC)      
        else: # After target SOC is met
            if delta_SOC >= 0: # Charge
                multiplier = np.exp(-(-np.log(self.m2_c)) * (next_SOC - self.target_SOC) \
                    / (1 - self.target_SOC))       
            else: # Discharge
                multiplier = self.m2_d * np.exp(-(-np.log(self.m2_d2 / self.m2_d)) * \
                    (next_SOC - self.target_SOC) / (1 - self.target_SOC))             
    
        # +ve - positive reward/ gain
        energy_gain = average_price * (next_SOC - current_SOC) * self.E_rated * multiplier
        trading_cost = current_price * (next_SOC - current_SOC) * self.E_rated
        wear_cost = self.wear_cost * np.abs((next_SOC - current_SOC) * self.E_rated)
        reward = energy_gain - trading_cost - wear_cost - penalty - penalty_pv

        if __debug__:
            infos['energy_gain'] = float(energy_gain)
            infos['trading_cost'] = float(trading_cost)
            infos['wear_cost'] = float(wear_cost)
            infos['penalty'] = float(penalty)
            infos['penalty_pv'] = float(penalty_pv)

        return next_SOC, reward, infos


    def plot_multiplier(self):
        SOC = np.linspace(0, 1, 100)
        multiplier_c = []
        multiplier_d = []
        
        # Compute piecewise multiplier
        for i in range(len(SOC)):
            if SOC[i] < self.target_SOC: # Before target SOC is met
                multiplier = 1 + self.a1 * np.exp(-(-np.log((self.m1 - 1) / self.a1)) * \
                    SOC[i] / self.target_SOC)      
                multiplier_c.append(multiplier)
                multiplier_d.append(multiplier)
            else: # After target SOC is met
                multiplier = np.exp(-(-np.log(self.m2_c)) * (SOC[i] - self.target_SOC) \
                    / (1 - self.target_SOC))
                multiplier_c.append(multiplier)
                
                multiplier = self.m2_d * np.exp(-(-np.log(self.m2_d2 / self.m2_d)) * \
                    (SOC[i] - self.target_SOC) / (1 - self.target_SOC))
                multiplier_d.append(multiplier)
    
        plt.plot(SOC, np.array(multiplier_c), "r-", label = "Charge")
        plt.plot(SOC, np.array(multiplier_d), "b-", label = "Discharge")
        
        plt.axvline(x  = self.target_SOC, linestyle = "--")
        plt.axhline(y  = self.m1, linestyle = "--")
        plt.axhline(y  = self.m2_d, linestyle = "--")
        plt.axhline(y  = self.m2_d2, linestyle = "--")
        plt.axhline(y  = self.m2_c, linestyle = "--")
        #plt.annotate("({}, {:.4f})".format(target_SOC, float(cum_prob_target)), 
        #         xy=(target_SOC - 0.25, cum_prob_target + 0.08))
        
        plt.xlabel("SOC")
        plt.ylabel("Multiplier")
        plt.legend()
        plt.show()