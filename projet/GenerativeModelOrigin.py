import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# TODO : change logic for computing de satisfaction. Best to take a score like more than 10 or something like this.


# Base class with continuous variables (may change)

class TrainSatisfactionSimulator:
    def __init__(self, n_customers : int):
        self.n_customers = n_customers
        self.satisfaction = np.zeros(n_customers)
        self.features_names = np.array(["Price", "Punctuality","Duration", "Frequency", "Overcrwoding", "Satisfaction"])
        self.price = None
        self.punctuality = None
        self.duration = None
        self.frequency = None
        self.overcrowding = None
        self.data = self.generate_data(n_customers)
        self.df = pd.DataFrame(self.data, columns=self.features_names)
    def generate_independent_vars(self, n_customers : int):
        prices = np.arange(1, 5)
        p_prices = np.random.normal(2.5, 1,size=len(prices))
        p_prices /= np.sum(p_prices)
        self.price = np.random.choice(prices, p=p_prices, size=n_customers)
        
        self.punctuality = np.random.choice([1, 2, 3, 4, 5], p=[0.03, 0.07, 0.1, 0.3, 0.5], size=n_customers) # 5 globalement très ponctuel, 1 globalement très en retard
        
        duration_table = np.arange(1, 5)
        len_dur = len(duration_table)
        p_duration = np.random.normal(2.5, 1, size=len_dur) # moyenne par jour
        p_duration /= np.sum(p_duration)
        self.duration = np.random.choice(duration_table, p=p_duration, size=n_customers)
        
        freq_table = np.arange(1, 5)
        len_freq = len(freq_table)
        p_freq = np.random.normal(2, 1, size=len_freq)
        p_freq /= np.sum(p_freq)
        self.frequency = np.random.choice(freq_table, p=p_freq, size=n_customers) # par an (entre 1 et 260)
        
        self.overcrowding = np.random.choice([1, 2, 3, 4, 5], size=n_customers)
        
    def generate_dependent_var(self, n_customers : int):
        pass
    
    def generate_data(self, n_customers):
        self.generate_independent_vars(n_customers)
        self.generate_dependent_var(n_customers)
        return np.array([self.price.astype(int), self.punctuality.astype(int), self.duration.astype(int), 
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

# model avec des variables catégoriques 

class TrainSatisfactionSimulator:
    def __init__(self, n_customers):
        self.n_customers = n_customers
        self.price = None
        self.punctuality = None
        self.duration = None
        self.frequency = None
        self.overcrowding = None
        self.satisfaction = np.zeros(n_customers)  # Satisfaction binaire (1 ou 0)
        self.data_matrix = None  # Matrice pour stocker les données complètes
        
        self.generate_independent_vars()
        self.generate_satisfaction()
        self.create_data_matrix()

    def generate_independent_vars(self):
        """
        Génère les valeurs pour price, punctuality, duration, frequency et overcrowding.
        Chaque variable suit une distribution normale entre 1 et 5.
        """
        mean, std_dev = 3, 1  # Moyenne centrée sur 3 pour rester dans [1,5]

        # Génération des variables indépendantes en suivant une distribution normale
        self.price = np.clip(np.round(np.random.normal(mean, std_dev, self.n_customers)), 1, 5)
        self.punctuality = np.clip(np.round(np.random.normal(mean, std_dev, self.n_customers)), 1, 5)
        self.duration = np.clip(np.round(np.random.normal(mean, std_dev, self.n_customers)), 1, 5)
        self.frequency = np.clip(np.round(np.random.normal(mean, std_dev, self.n_customers)), 1, 5)
        self.overcrowding = np.clip(np.round(np.random.normal(mean, std_dev, self.n_customers)), 1, 5)

    def generate_satisfaction(self):
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

    def create_data_matrix(self):
        """
        Crée une matrice de données combinant toutes les variables indépendantes et la satisfaction.
        """
        # Combine les variables et la satisfaction dans une matrice
        self.data_matrix = np.column_stack((self.price, self.punctuality, self.duration,
                                            self.frequency, self.overcrowding, self.satisfaction))
    
    def display_data_matrix(self):
        """
        Affiche la matrice des données pour chaque client.
        """
        print("Data Matrix (price, punctuality, duration, frequency, overcrowding, satisfaction):")
        print(self.data_matrix)
        
if __name__ == "__main__":
    gen = ComplexDependentSatisfaction(20)
    print(gen.df)