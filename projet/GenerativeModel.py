import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

#TODO : 
# Faire une simulation sans le revenu

# Base class with continuous variables (may change)

class TrainSatisfactionSimulator:
    def __init__(self, n_customers : int):
        self.n_customers = n_customers
        self.satisfaction = np.zeros(n_customers)
        self.features_names = np.array(["Age", "Gender", "Income", "Remote Working Days", "Has Car",
                                        "Price", "Punctuality","Duration", 
                                        "Frequency", "Overcrowding", "Satisfaction"])
        self.price = np.zeros(n_customers)
        self.punctuality = np.random.choice([1, 2, 3, 4, 5], 
                                            p=[0.03, 0.07, 0.1, 0.3, 0.5], 
                                            size=n_customers) # 5 globalement très ponctuel, 1 globalement très en retard
        self.duration = None
        self.frequency = None
        self.overcrowding = np.zeros(n_customers)
        self.age = np.random.choice(np.arange(15, 80), size=n_customers)
        self.gender = np.random.choice(['M', 'F'], size=n_customers) # Male, female or other
        self.income = np.zeros(n_customers) # 0 to 1'000'000
        self.remote_working_days = np.random.choice(5, size=n_customers) # 0 to 5 per week
        self.has_car = np.array([np.random.choice(['yes', 'no']) if self.age[i] >= 18 
                                 else 'no' for i in range(n_customers)]) # Yes or No
        self.data = self.generate_data(n_customers)
        self.df = pd.DataFrame(self.data, columns=self.features_names)
        
    def generate_vars(self, n_customers : int):
        # generate duration
        duration_table = np.arange(1, 5)
        len_dur = len(duration_table)
        p_duration = np.random.normal(2.5, 1, size=len_dur) # moyenne par jour
        p_duration /= np.sum(p_duration)
        self.duration = np.random.choice(duration_table, p=p_duration, size=n_customers)

        # generate income
        incomes = np.arange(0, 500000, 1000)
        p_income = np.random.normal(85702, 2,size=len(incomes))
        p_income /= np.sum(p_income)
        for i in range(n_customers):
            income = np.random.choice(incomes, p=p_income)
            if self.age[i] > 35 and income <= 100000:
                income += 10000
            if self.gender[i] == 'F':
                income -= income * 0.18
            self.income[i] = income

        # generate price
        prices = np.arange(1, 5)
        prices = np.arange(1, 5)
        p_prices = np.random.normal(2.5, 1,size=len(prices))
        p_prices /= np.sum(p_prices)
        self.price = np.random.choice(prices, p=p_prices, size=n_customers)
        # frequency       
        freq_table = np.arange(1, 5)
        len_freq = len(freq_table)
        p_freq = np.random.normal(2, 1, size=len_freq)
        p_freq /= np.sum(p_freq)
        
        self.frequency = np.random.choice(freq_table, p=p_freq, size=n_customers) # par an (entre 1 et 260)
        # overcrowding and frequency
        self.overcrowding = np.random.choice([1, 2, 3, 4, 5], size=n_customers)
          
    def generate_dependent_var(self, n_customers : int):
        pass
    
    def generate_data(self, n_customers):
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
        prices = np.arange(1, 5)
        len_p = len(prices)
        p_prices = np.random.normal(2.5, 1,size=len_p)
        p_prices /= np.sum(p_prices)
        p_prices_under25 = np.random.exponential(size=len_p)
        p_prices_under25 /= np.sum(p_prices_under25)
        p_prices_high_income = np.random.power(5, len_p)
        p_prices_high_income /= np.sum(p_prices_high_income)
        for i in range(n_customers):
            if self.income[i] > 100000:
                self.price[i] = np.random.choice(prices, p=p_prices_high_income)
            else:
                if self.age[i] < 25:
                    self.price[i] = np.random.choice(prices, p=p_prices_under25)
                else:
                    self.price[i] = np.random.choice(prices, p=p_prices)
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
   
    def generate_dependent_var(self, n_customers):
        data = np.array([self.price, self.punctuality, self.duration, 
                             self.frequency, self.overcrowding]).T     
        i_price = 1
        i_dur = 0.1
        i_freq = 0.05
        i_punct = 0.9
        i_overcrow = 0.8
        
        n_features = data.shape[1]
        uniform = 1 / n_features # here uniform does not mean that they all have same probability.
        prices = self.price / n_features * uniform
        punctuality = self.punctuality / n_features * uniform
        duration = self.duration / n_features * uniform
        frequency =  self.frequency / n_features * uniform
        overcrowding = self.overcrowding / n_features * uniform
        
        for i in range(n_customers):
            score = ((i_price * prices[i]) +  (i_dur * duration[i])
            + (i_freq * frequency[i]) + (i_punct * punctuality[i]) 
            + (i_overcrow * overcrowding[i]))
            if (score >= 0.4):
                self.satisfaction[i] = 1
            else:
                self.satisfaction[i] = 0
        return
    
    
# Simulation sans revenu

class NoIncomeDependentSatisfaction(ComplexDependentSatisfaction):
    """Simulation Class without income"""
    
    def __init__(self, n_customers):
        super(TrainSatisfactionSimulator).__init__(n_customers)
        
    def generate_vars(self, n_customers : int):
        super(TrainSatisfactionSimulator).generate_vars(n_customers)
        # generate price
        prices = np.arange(1, 5)
        prices = np.arange(1, 5)
        p_prices = np.random.normal(2.5, 1,size=len(prices))
        p_prices /= np.sum(p_prices)
        self.price = np.random.choice(prices, p=p_prices, size=n_customers) 
       
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
    
    def generate_data(self, n_customers):
        self.generate_vars(n_customers)
        super().generate_dependent_var(n_customers)
        return np.array([self.age.astype(int), self.gender, self.remote_working_days.astype(int), self.has_car,
                         self.price.astype(int), self.punctuality.astype(int), self.duration.astype(int), 
                         self.frequency.astype(int), self.overcrowding.astype(int), self.satisfaction.astype(int)]).T


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