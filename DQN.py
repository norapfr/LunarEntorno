import numpy as np
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf

import torch.optim as optim

from collections import deque, namedtuple

from lunar import LunarLanderEnv

print(torch.__version__)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Lecturas interesantes: 
# https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf (Playing atari with DQN)
# https://www.nature.com/articles/nature14236 (Human level control through RL)
# https://www.lesswrong.com/posts/kyvCNgx9oAwJCuevo/deep-q-networks-explained


#DQN necesita redes q sean porfundas, entrenen rapido y no sufran de gradientes debiles
class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(DQN,self).__init__()
        #definimos la red feedforward
        self.fc1=nn.Linear(state_size,hidden_size) #capa de entrada->tiene tantos nodos como estado tiene el problema
        self.fc2=nn.Linear(hidden_size,hidden_size) #primera capa oculta->64
        self.fc3=nn.Linear(hidden_size,hidden_size) # segunda capa oculta->64
        self.out= nn.Linear(hidden_size,action_size) #capa de salida-> nodos son el numero de acciones que tiene el problema
    
    #esto lo añado yo
    #indica como se conectan las capas definidas anteriormente
    # usamos relu pq mantiene un gradiente constante, lo q facilita el entrenamiento de redes profundas al evitar q los gradientes se vuelvan insignificantes durante la retropropagacion
    #ademas es simple y requiere menos recursos-> se usa en dqn para modelar la funcion de valor
    def forward(self,state):
        x=torch.relu(self.fc1(state))
        x=torch.relu(self.fc2(x))
        x=torch.relu(self.fc3(x))
        return self.out(x)
    #puede requerir mas funciones segun la libreria escogida.
    

 #almacena experiencias pasadas del agente   
class ReplayBuffer():
    def __init__(self, buffer_size=10000):
        self.buffer = deque(maxlen=buffer_size) # deque es una doble cola que permite añadir y quitar elementos de ambos extremos
    #    self.memory= deque(maxlen=buffer_size)
      #  self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
       
    def push(self, state, action, reward, next_state, done):
        # insert into buffer
        self.buffer.append((state,action,reward,next_state,done))
        
        
    def sample(self, batch_size):
        # get a batch of experiences from the buffer
        #pillamos una muestra random del buffer-> cada elemento es una tupla de experiencia
        batch = random.sample(self.buffer, batch_size)
    
        states = np.array([experience[0] for experience in batch]) #pilla todos los states de la muestra random
        actions = np.array([experience[1] for experience in batch])
        rewards = np.array([experience[2] for experience in batch])
        next_states = np.array([experience[3] for experience in batch])
        dones = np.array([experience[4] for experience in batch])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)
    
class DQNAgent():
    def __init__(self, lunar: LunarLanderEnv, gamma=0.99, 
                epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                learning_rate=0.001, batch_size=64, 
                memory_size=10000, episodes=1500, 
                target_network_update_freq=10,
                replays_per_episode=1000):
        """
        Initialize the DQN agent with the given parameters.
        
        Parameters:
        lunar (LunarLanderEnv): The Lunar Lander environment instance.
        gamma (float): Discount factor for future rewards.
        epsilon (float): Initial exploration rate.
        epsilon_decay (float): Decay rate for exploration rate.
        epsilon_min (float): Minimum exploration rate.
        learning_rate (float): Learning rate for the optimizer.
        batch_size (int): Size of the batch for experience replay.
        memory_size (int): Number of experiences stored on the replay memory.
        episodes (int): Number of episodes to train the agent.
        target_network_update_freq (int): Frequency of updating the target network.
        """
        
        # Initialize hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.episodes = episodes
        
        self.target_updt_freq = target_network_update_freq
        self.replays_per_episode = replays_per_episode
        
        # Initialize replay memory
        # a deque is a double sided queue that allows us to append and pop elements from both ends
        self.memory = ReplayBuffer(memory_size)
        
        # Initialize the environment
        self.lunar = lunar
        
        observation_space = lunar.env.observation_space
        action_space = lunar.env.action_space

        self.action_size = action_space.n

        
        # La red neuronal debe tener un numero de parametros
        # de entrada igual al espacio de observaciones
        # y un numero de salida igual al espacio de acciones.
        # Asi como un numero de capas intermedias adecuadas.
        self.q_network = DQN(
            state_size=observation_space.shape[0],
            action_size=action_space.n,
            hidden_size=64#elegir un tamaño de capa oculta
        ).to(device)
        
        self.target_network = DQN(
            state_size=observation_space.shape[0],
            action_size=action_space.n,
            hidden_size=64 #elegir un tamaño de capa oculta
        ).to(device)
        
        # Set weights of target network to be the same as those of the q network
        self.target_network.load_state_dict(self.q_network.state_dict())

       # Set target network to evaluation mode
        self.target_network.eval()
        #se pone en modo evaluacion para q no se actualice grandiantes
        # la red target solo se actualiza cada x tiempo para estabilizar el aprendizaje-> se usa para hacer predicciones o para entrenar 
      
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        # depende del framework que uses (tf o pytorch)
        
        print(f"QNetwork:\n {self.q_network}")

 
          
    def act(self):
        """
        This function takes an action based on the current state of the environment.
        it can be randomly sampled from the action space (based on epsilon) or
        it can be the action with the highest Q-value from the model.
        """
        state= self.lunar.state
        if np.random.rand()<self.epsilon:
            action = random.choice(np.arange(self.action_size)) #accion aleatoria->exploracion
        else:
            #pasamos el estado por la red para obtener los valores Q y escogemos la accion con el Q mas alto
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            with torch.no_grad():
                q_value= self.q_network(state_tensor)
            self.q_network.train()
            action = np.argmax(q_value.cpu().data.numpy())
        next_state, reward, done = self.lunar.take_action(action, verbose=False)
        
        return next_state, reward, done, action
    
    def update_model(self):
        """
        Perform experience replay to train the model.
        Samples a batch of experiences from memory, computes target Q-values,
        and updates the model using the computed loss.
        """
       
      
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = torch.from_numpy(states).float().to(device)
        actions = torch.from_numpy(np.array(actions)).long().to(device)
        rewards = torch.from_numpy(np.array(rewards)).float().to(device)
        next_states = torch.from_numpy(next_states).float().to(device)
        dones = torch.from_numpy(np.array(dones).astype(np.uint8)).float().to(device)

        q_values = self.q_network(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        #gather seleciona la Q para las acciones tomadas

        next_q_values = self.target_network(next_states).max(1)[0].detach()
        # obtiene el maximo q para los siguinetes estado de la red objetivo y usamos detach para q no se calcule el gradiente
        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = F.mse_loss(q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        
        return loss.item()
        
    def update_target_network(self):
        # copiar los pesos de la red q a la red objetivo
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    def save_model(self, path):
        """
        Save the model weights to a file.
        Parameters:
        path (str): The path to save the model weights.
        Returns:
        None
        """
        # guardar el modelo en el path indicado
        torch.save(self.q_network.state_dict(),path)
    
    def load_model(self, path):
        """
        Load the model weights from a file.
        Parameters:
        path (str): The path to load the model weights from.
        Returns:
        None
        """
        
        self.q_network.load_state_dict(torch.load(path))
        self.q_network.eval()  # Modo evaluación
           
    def train(self):
        """
        Train the DQN agent on the given environment for a specified number of episodes.
        The agent will interact with the environment, store experiences in memory, and learn from them.
        The target network will be updated periodically based on the update freq parameter.
        The agent will also decay the exploration rate (epsilon) over time.
        The training process MUST be logged to the console.    
        Returns:
        None
        """
        success_count=0
        for episode in range(self.episodes):
            state = self.lunar.reset()
            total_reward=0
            done=False
           
            while not done:
                #eligumos una accion y la ejecutamos
                next_state, reward, done, action = self.act()
                self.memory.push(state, action, reward, next_state, done) #se guarda la experiencia en memoria

               
                state = next_state
                total_reward += reward

                if len(self.memory) >= self.batch_size:
                    loss = self.update_model()
                else:
                    loss = 0.0
   

            # de cada episodio, se hace el replay extra:
            #se hace un "repaso del episodio" -> hace q aprenda mejor y se prepara para el siguiente
            for _ in range(self.replays_per_episode):
                if len(self.memory) >= self.batch_size:
                    loss = self.update_model()

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            if episode % self.target_updt_freq == 0:
             self.update_target_network()
            print(f"Ep {episode:4d} | Reward: {total_reward:8.2f} | Eps: {self.epsilon:.3f} | Loss: {loss:.4f}")
           # Verifica si se resolvió el entorno
            if total_reward >= 200:
                 success_count += 1
                 print(success_count)
        success_rate = success_count / self.episodes
        print(success_rate)
       
                    