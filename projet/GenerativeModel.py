import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

"""Different classes for the simulation

    Not all classes are used for the project.
    
    Main classes are :
    - ComplexDependentSatisfaction
    - ImpactOnOverCrowding
    
    Categorical variables are ranging from 1 to 5
"""

class Simulator:
    """Base Class for the Simulation"""
    
    def __init__(self, n_customers : int):
        """Initialisation function
        
        Variables:
            self.n_customers -- number of customers that took the survey
            self.satisfaction -- satisfaction of the customers. Unsatisfied = 0, Satisfied = 1
            self.table -- Numpy array with a list of number ranging from 1 to 5 ([1, 2, 3, 4 ,5])
            self.price -- Customer's price per ticket (very low, low, normal, high, very high)
            self.punctuality -- 1 is almost never punctual, 5 is very punctual
            self.duration -- Customer's journey duration
            self.frequency -- Customer's frequency of train usage
            self.overcrowding -- Customer's feeling of crowd level, 1 is very low and 5 is very high
            self.age -- Customer's age
            self.gender -- Customer's gender
            self.income -- Customer's income per year
            self.remote_working_days -- Customer's number of days per week of remote work
            self.has_car -- If customers owns a car or not
        """
        
        self.n_customers = n_customers
        self.satisfaction = np.zeros(n_customers)
        self.table = np.arange(1, 6)
        self.price = np.random.choice(self.table, 
                                      p=[0.1, 0.2, 0.40, 0.2, 0.1],size=n_customers)
        self.punctuality = np.random.choice(self.table, 
                                            p=[0.03, 0.07, 0.1, 0.3, 0.5], size=n_customers)
        self.duration = np.random.choice(self.table,
                                         p=[0.15, 0.2, 0.30, 0.2, 0.15], size=n_customers)
        self.frequency = np.random.choice(self.table,
                                          p=[0.35, 0.25, 0.2, 0.15, 0.05], size=n_customers)
        self.overcrowding = np.random.choice(self.table, size=n_customers)
        self.age = np.random.choice(np.arange(15, 80), size=n_customers)
        self.gender = np.random.choice(['M', 'F'], size=n_customers)
        self.income = np.zeros(n_customers)
        self.remote_working_days = np.random.choice(5, size=n_customers)
        self.has_car = np.array([np.random.choice(['yes', 'no']) if self.age[i] >= 18 
                                 else 'no' for i in range(n_customers)])
    
    def generate_vars(self, n_customers : int):
        """Generate variables that depend on multiple factors

            self.income -> np.array[]
        """
        
        # generate income
        incomes = [1000, 50000, 74000, 99000, 200000] # less than 30k, 30k <= x < 60k, 60k <= x < 85k, 85k <= x 100k, x < 100k
        for i in range(n_customers):
            income = np.random.choice(incomes, p=[0.05, 0.1, 0.5, 0.25, 0.1])
            if self.age[i] > 35 and income <= 100000:
                income += 10000
            if self.gender[i] == 'F':
                income -= income * 0.18
            self.income[i] = income
    def generate_dependent_var(self, n_customers : int):
        pass
    
    def generate_data(self, n_customers):
        pass    

class TrainSatisfactionSimulator(Simulator):
    """Extension of the base class"""
    
    def __init__(self, n_customers : int):
        """Initialisation function based on Simulator class
        Varaibles:
            self.features_names -- names of the generated features
            self.data -- Numpy Array : all the data generated
            self.df -- data converted into a pandas DataFrame
        """
        
        super().__init__(n_customers)
        self.features_names = np.array(["Age", "Gender", "Income", "Remote Working Days", "Has Car",
                                        "Price", "Punctuality","Duration", 
                                        "Frequency", "Overcrowding", "Satisfaction"])
        self.data = self.generate_data(n_customers)
        self.df = pd.DataFrame(self.data, columns=self.features_names)
        
    def generate_data(self, n_customers):
        """Generate all the data of the simulation
        
            Return: Numpy array with each column being one feature of the data generated.
        """
        
        self.generate_vars(n_customers)
        self.generate_dependent_var(n_customers)
        return np.array([self.age.astype(int), self.gender, self.income.astype(int), self.remote_working_days.astype(int), self.has_car,
                         self.price.astype(int), self.punctuality.astype(int), self.duration.astype(int), 
                         self.frequency.astype(int), self.overcrowding.astype(int), self.satisfaction.astype(int)]).T


# Independent Satisfaction
class IndependentSatisfaction(TrainSatisfactionSimulator):
    def __init__(self, n_customers):
        super().__init__(n_customers)
    def generate_dependent_var(self, n_customers):
        for i in range (n_customers):
            d = np.random.choice(2)
            self.satisfaction[i] = d
        return

# Simple example of a satisfaction depending on prices
            
class SimpleDependentSatisfaction(TrainSatisfactionSimulator):
    def __init__(self, n_customers):
        super().__init__(n_customers)
        
    def generate_dependent_var(self, n_customers):
        prices = self.price
        for i in range(n_customers):
            if (prices[i] > 50):
                self.satisfaction[i] = 1
            else:
                self.satisfaction[i] = 0
        return
    
# More complex satisfaction "function"

class ComplexDependentSatisfaction(TrainSatisfactionSimulator):
    def __init__(self, n_customers):
        super().__init__(n_customers)
        
    def generate_vars(self, n_customers : int):
        super().generate_vars(n_customers)
        # generate price
        prices = np.arange(1, 6)
        for i in range(n_customers):
            if self.income[i] > 100000:
                self.price[i] = np.random.choice(prices, 
                                            p=[0.05, 0.05, 0.3, 0.5, 0.1])
            else:
                if self.age[i] < 25:
                    self.price[i] = np.random.choice(prices, p=[0.2, 0.5, 0.2, 0.09, 0.01])
                else:
                    self.price[i] = np.random.choice(prices, p=[0.1, 0.2, 0.5, 0.1, 0.1])
            if self.duration[i] > 3 and self.price[i] <= 4:
                self.price[i] += 1
       
        # overcrowding and frequency
        for i in range(n_customers):
            # frequency
            if self.remote_working_days[i] > 1 and self.frequency[i] <= 4:
                self.frequency[i] += 1
            if self.has_car[i] == 'yes' and self.frequency[i] > 1:
                self.frequency[i] -= 1
            # overcrowding
            if self.punctuality[i] < 3:
                self.overcrowding[i] = np.random.choice([3, 4, 5])
            else:
                self.overcrowding[i] = np.random.choice([1, 2, 3, 4, 5])
            if self.price[i] >= 4 and self.overcrowding[i] > 1:
                self.overcrowding[i] -= 1
        return
   
    def generate_dependent_var(self, n_customers):
        freq_table = [0.05, 0.1, 0.2, 0.25, 0.5]
        price_table = [0.1, 0.2, 0.25, 0.5, 1]
        dur_table = [0.2, 0.25, 0.5, 1, 1]
        overcrow_table = [0, -1, -5, -10, -20]
        punct_table = [-5, 0, 5, 10, 15]
        
        for i in range(n_customers):
            price_impact = 1 / price_table[self.price[i] - 1]
            punctuality_impact = punct_table[self.punctuality[i] - 1]
            duration_impact = 1 / dur_table[self.duration[i] - 1]
            frequency_impact = 1 / freq_table[self.frequency[i] - 1]
            overcrowding_impact = overcrow_table[self.overcrowding[i] - 1]
            
            satisfaction_score = price_impact + punctuality_impact + duration_impact \
                                 + frequency_impact + overcrowding_impact
            # max satisfaction score possible = 50
            if satisfaction_score < 0:
                self.satisfaction[i] = np.random.choice([0, 1], p=[0.95, 0.05])
            elif satisfaction_score >= 0 and satisfaction_score < 10:
                self.satisfaction[i] = np.random.choice([0, 1], p=[0.7, 0.3])
            elif satisfaction_score >= 10 and satisfaction_score < 25:
                self.satisfaction[i] = np.random.choice([0, 1])
            elif satisfaction_score >= 25 and satisfaction_score < 40:
                self.satisfaction[i] = np.random.choice([0, 1], p=[0.3, 0.7])
            elif satisfaction_score >= 40:
                self.satisfaction[i] = np.random.choice([0, 1], p=[0.1, 0.9])
        return
    
    
# Simulation sans revenu

class NoIncomeDependentSatisfaction(ComplexDependentSatisfaction):
    """Simulation Class without income"""
    
    def __init__(self, n_customers):
        TrainSatisfactionSimulator.__init__(self, n_customers)

    def generate_vars(self, n_customers : int):
        TrainSatisfactionSimulator.generate_vars(self,n_customers)
        # overcrowding and frequency
        for i in range(n_customers):
            # frequency
            if self.remote_working_days[i] > 1 and self.frequency[i] <= 4:
                self.frequency[i] += 1
            if self.has_car[i] == 'yes' and self.frequency[i] > 1:
                self.frequency[i] -= 1
            # overcrowding
            if self.punctuality[i] < 3:
                self.overcrowding[i] = np.random.choice([3, 4, 5])
            else:
                self.overcrowding[i] = np.random.choice([1, 2, 3, 4, 5])
            if self.price[i] >= 4 and self.overcrowding[i] > 1:
                self.overcrowding[i] -= 1
        return

class ImpactOnOvercrowding(ComplexDependentSatisfaction):
    """We add a feature (first class) that impacts the simulation
    
    
    Keyword arguments:
    argument -- description
    Return: return_description
    """
    
    def __init__(self, n_customers):
        Simulator.__init__(self, n_customers)
        self.features_names = np.array(["Age", "Gender", "Income", "Remote Working Days", 
                                        "Has Car", "First-Class",
                                        "Price", "Punctuality","Duration", 
                                        "Frequency", "Overcrowding", "Satisfaction"])
        self.first_class = np.random.choice([0, 1], size=n_customers)
        self.data = self.generate_data(n_customers)
        self.df = pd.DataFrame(self.data, columns=self.features_names)
        
    def generate_vars(self, n_customers):
        return super().generate_vars(n_customers)
    
    def generate_dependent_var(self, n_customers):
        return super().generate_dependent_var(n_customers)
    
    def generate_data(self, n_customers):
            self.generate_vars(n_customers)
            self.generate_dependent_var(n_customers)
            return np.array([self.age.astype(int), self.gender, 
                            self.income.astype(int), self.remote_working_days.astype(int), 
                            self.has_car, self.first_class.astype(int),
                            self.price.astype(int), self.punctuality.astype(int), 
                            self.duration.astype(int), self.frequency.astype(int), 
                            self.overcrowding.astype(int), self.satisfaction.astype(int)]).T

# Satisfaction depend de la Moyenne pondérée 
class PondDependentSatisfaction(TrainSatisfactionSimulator):
    def __init__(self, n_customers):
        super().__init__(n_customers)
        
    def generate_dependent_var(self, n_customers):
        """
        Calcule la satisfaction binaire (1 ou 0) en fonction des variables indépendantes
        pondérées par leur facteur d'importance.
        """
        # Facteurs d'importance
        i_price = 0.2
        i_punctuality = 0.3
        i_duration = 0.1
        i_frequency = 0.2
        i_overcrowding = 0.2

        # Crée un tableau de données et applique une normalisation
        data = np.array([self.price, self.punctuality, self.duration, 
                         self.frequency, self.overcrowding]).T
        scaler = MinMaxScaler()
        data = scaler.fit_transform(data)  # Met à l'échelle entre [0, 1]

        # Applique les facteurs d'importance pour chaque variable
        weighted_sum = (i_price * data[:, 0] +
                        i_punctuality * data[:, 1] +
                        i_duration * data[:, 2] +
                        i_frequency * data[:, 3] +
                        i_overcrowding * data[:, 4])

        # Calcule la satisfaction : 0 si la somme pondérée > 0.5, sinon 1
        self.satisfaction = np.where(weighted_sum > 0.5, 0, 1)  # Utilise un seuil de 0.5 après normalisation
        return



# Test       
if __name__ == "__main__":
    gen = ComplexDependentSatisfaction(20)
    print(gen.df)