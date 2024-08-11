import os
import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset,DataLoader
from sklearn.preprocessing import StandardScaler,MinMaxScaler

seed_value=1337
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
# thermonet ...working ... trained on entropy and energy and free energy(ofcourse the condition F=U-TS has to be satisifed)
class ANNBranch1(nn.Module):
    def __init__(self, input_dim,wht,s, activation):
        super(ANNBranch1, self).__init__()

        self.shared_layers = nn.Sequential(
             nn.Linear(input_dim, int(input_dim * wht)),
             activation,
            nn.Linear(int(input_dim * wht), int(input_dim *wht * wht)),
            activation,
            nn.Linear(int(input_dim *wht*wht), s),
            activation,
        )

        self.output_layer = nn.Sequential(
            nn.Linear(s, 1),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x = self.shared_layers(x)
        x = self.output_layer(x)
        return x


class ANN_thermo1(nn.Module):
    def __init__(self, input_dim,wht,s, lr, activation, epochs, batch_size, loss,
         w1,w2,w3,  # weights of the network have to give individually not array becuase hypopt will be easier :(
         Temperature):
        super(ANN_thermo1, self).__init__()
        # hyperparameters of the network 
        self.input_dim = input_dim
        self.wht = wht
        self.s=s
       # self.s1=s1
        self.activation = self.get_activation(activation)
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss_function = self.get_loss(loss)
        self.lr = lr
        self.w1=w1  # weight for loss of energy 
        self.w2=w2  # weight for loss of entropy        
        self.w3=w3  # thermodynamic loss for 
        self.Temperature=Temperature

        self.branch_entropy = ANNBranch1(input_dim,wht,s, self.activation)
        self.branch_energy = ANNBranch1(input_dim,wht,s, self.activation)
        self.branch_feng = ANNBranch1(input_dim,wht,s, self.activation)

    def forward(self, x):
        output_entropy = self.branch_entropy(x)
        output_energy = self.branch_energy(x)
        output_feng = self.branch_feng(x)
        return output_entropy, output_energy,output_feng

    def get_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'prelu':
            return nn.PReLU(num_parameters=1) 
        elif activation == 'linear':
            return nn.Identity()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU()
        elif activation == 'rrelu':
            return nn.RReLU()
        else:
            raise NotImplementedError("Activation function {} not implemented".format(activation))

    def get_loss(self, loss):
        if loss == 'mse':
            return nn.MSELoss()
        elif loss == 'mae':
            return nn.L1Loss()
        else:
            raise NotImplementedError("Loss function {} not implemented".format(loss))

    def custom_loss(self,output_entropy, output_energy,output_feng,target_entropy, target_energy, target_feng):
        loss_entropy = self.loss_function(output_entropy, target_entropy)
        loss_energy = self.loss_function(output_energy, target_energy)
        loss_feng = self.loss_function(output_feng, target_feng)


        # expected using outputs
        # free energy equation
        expected_feng = output_energy/6.8 - (self.Temperature / 1000) * output_entropy
        thermodynamic_loss_feng = self.loss_function(expected_feng, target_feng)
        # entropy
        # add all the losses together
        # total_loss = (loss_energy+thermodynamic_loss_energy
        #             + loss_entropy+thermodynamic_loss_entropy+
        #              loss_feng + thermodynamic_loss_feng)
        reg_loss = (loss_energy+loss_entropy+loss_feng)
        #thermodynamic_loss=(thermodynamic_loss_feng+thermodynamic_loss_energy+thermodynamic_loss_entropy)
        # w1=1;w2=1;w3=0;w4=15;w5=0;w6=0

        total_loss=(self.w1*loss_energy+self.w2*loss_entropy+self.w3*thermodynamic_loss_feng) # the weights can be hyperparamters now
        #print(thermodynamic_loss_energy)
        return total_loss

    def train_predict(self, X_train, y_train_entropy, y_train_energy, X_test, free_eng_train):
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        y_train_entropy = np.array(y_train_entropy)
        y_train_energy = np.array(y_train_energy)
        y_train_feng = np.array(free_eng_train)

        X_train_tensor = torch.FloatTensor(X_train_scaled)
        y_train_entropy_tensor = torch.FloatTensor(y_train_entropy).view(-1, 1)
        y_train_energy_tensor = torch.FloatTensor(y_train_energy).view(-1, 1)
        y_train_feng_tensor = torch.FloatTensor(y_train_feng).view(-1, 1)
        X_test_tensor = torch.FloatTensor(X_test_scaled)

        train_data = TensorDataset(X_train_tensor, y_train_entropy_tensor, y_train_energy_tensor, y_train_feng_tensor)
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        optimizer = optim.Adam(self.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            for inputs, target_entropy, target_energy, target_feng in train_loader:
                optimizer.zero_grad()
                output_entropy, output_energy,output_feng = self(inputs)
                total_loss = self.custom_loss(output_entropy, output_energy,output_feng,target_entropy, target_energy, target_feng)
                total_loss.backward()
                optimizer.step()
            #print(epoch, total_loss.item())

        predictions = self(X_test_tensor)
        entropy_pred, energy_pred,feng_pred = predictions[0].detach().numpy(), predictions[1].detach().numpy(),predictions[2].detach().numpy()
        return entropy_pred, energy_pred,feng_pred



#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# thermonet ...working ... trained on entropy and energy and the condition F=U-TS has to be satisifed)
class ANNBranch2(nn.Module):
    def __init__(self, input_dim, activation):
        super(ANNBranch2, self).__init__()

        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, int(input_dim * 0.5)),
            activation,
            nn.Linear(int(input_dim * 0.5), int(input_dim * 0.5 * 0.5)),
            activation,
            nn.Linear(int(input_dim *0.5*0.5), 2),
            activation,
        )

        self.output_layer = nn.Sequential(
            nn.Linear(2, 1),
            activation
        )

    def forward(self, x):
        x = self.shared_layers(x)
        x = self.output_layer(x)
        return x


class ANN_thermo2(nn.Module):
    def __init__(self, input_dim, lr, activation, epochs, batch_size, loss,
         w1,w2,w3,  # weights of the network have to give individually not array becuase hypopt will be easier :(
         Temperature):
        super(ANN_thermo2, self).__init__()
        # hyperparameters of the network
        self.input_dim = input_dim
        self.activation = self.get_activation(activation)
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss_function = self.get_loss(loss)
        self.lr = lr
        self.w1=w1  # weight for loss of energy
        self.w2=w2  # weight for loss of entropy
        self.w3=w3  # thermodynamic loss for
        self.Temperature=Temperature

        self.branch_entropy = ANNBranch2(input_dim, self.activation)
        self.branch_energy = ANNBranch2(input_dim, self.activation)



    def forward(self, x):
        output_entropy = self.branch_entropy(x)
        output_energy = self.branch_energy(x)
        return output_entropy, output_energy

    def get_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'linear':
            return nn.Identity()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU()
        else:
            raise NotImplementedError("Activation function {} not implemented".format(activation))

    def get_loss(self, loss):
        if loss == 'mse':
            return nn.MSELoss()
        elif loss == 'mae':
            return nn.L1Loss()
        else:
            raise NotImplementedError("Loss function {} not implemented".format(loss))

    def custom_loss(self,output_entropy, output_energy,target_entropy, target_energy, target_feng):
        loss_entropy = self.loss_function(output_entropy, target_entropy)
        loss_energy = self.loss_function(output_energy, target_energy)
        # expected using outputs
        # expected_energy = output_feng + (Temperature / 1000) * output_entropy
        # thermodynamic_loss_energy = self.loss_function(expected_energy, target_energy)
        # free energy equation
        expected_feng = output_energy - (self.Temperature / 1000) * output_entropy
        thermodynamic_loss_feng = self.loss_function(expected_feng, target_feng)
        # entropy
        # expected_entropy = (output_energy-output_feng)*(1000/Temperature)
        # thermodynamic_loss_entropy = self.loss_function(expected_entropy, target_entropy)
        # w1=1;w2=1;w3=15;
        total_loss=(self.w1*loss_energy+self.w2*loss_entropy+self.w3*thermodynamic_loss_feng)
        # the weights are hyperparameters of the network
        #print(thermodynamic_loss_energy)
        return total_loss

    def train_predict(self, X_train, y_train_entropy, y_train_energy, X_test, free_eng_train):
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        y_train_entropy = np.array(y_train_entropy)
        y_train_energy = np.array(y_train_energy)
        y_train_feng = np.array(free_eng_train)

        X_train_tensor = torch.FloatTensor(X_train_scaled)
        y_train_entropy_tensor = torch.FloatTensor(y_train_entropy).view(-1, 1)
        y_train_energy_tensor = torch.FloatTensor(y_train_energy).view(-1, 1)
        y_train_feng_tensor = torch.FloatTensor(y_train_feng).view(-1, 1)
        X_test_tensor = torch.FloatTensor(X_test_scaled)

        train_data = TensorDataset(X_train_tensor, y_train_entropy_tensor, y_train_energy_tensor, y_train_feng_tensor)
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        optimizer = optim.Adam(self.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            for inputs, target_entropy, target_energy, target_feng in train_loader:
                optimizer.zero_grad()
                output_entropy, output_energy= self(inputs)
                total_loss = self.custom_loss(output_entropy, output_energy,target_entropy, target_energy, target_feng)
                total_loss.backward()
                optimizer.step()
            #print(epoch, total_loss.item())

        predictions = self(X_test_tensor)
        entropy_pred, energy_pred= predictions[0].detach().numpy(), predictions[1].detach().numpy()
        return entropy_pred, energy_pred

#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx

# Normal neural network
class ANN_torch(nn.Module):
    def __init__(self, input_dim, lr, activation, epochs, batch_size, loss):
        super(ANN_torch, self).__init__()

        self.input_dim = input_dim
        self.activation = self.get_activation(activation)
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss = self.get_loss(loss)
        self.lr = lr
        self.ann = self.build_model()

    def build_model(self):
        model = nn.Sequential(
            nn.Linear(self.input_dim, int(self.input_dim * 0.5)),
            self.activation,
            nn.Linear(int(self.input_dim * 0.5), int(self.input_dim * 0.5 * 0.5)),
            self.activation,
            nn.Linear(int(self.input_dim * 0.5 * 0.5), 2),
            self.activation,
            nn.Linear(2, 1),
            self.activation
        )
        return model

    def forward(self, x):
        return self.ann(x)

    def get_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'linear':
            return nn.Identity()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU()
        else:
            raise NotImplementedError("Activation function {} not implemented".format(activation))

    def get_loss(self, loss):
        if loss == 'mse':
            return nn.MSELoss()
        elif loss == 'mae':
            return nn.L1Loss()
        else:
            raise NotImplementedError("Loss function {} not implemented".format(loss))
    def train_predict(self, X_train, y_train, X_test):
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        y_train = np.array(y_train)
        # Convert data to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled)
        y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
        X_test_tensor = torch.FloatTensor(X_test_scaled)

        # Create DataLoader for training data
        train_data = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)

        # Define optimizer
        optimizer = optim.Adam(self.ann.parameters(), lr=self.lr)

        # Training loop
        for epoch in range(self.epochs):
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = self.ann(inputs)
                loss = self.loss(outputs, targets)
                loss.backward()
                optimizer.step()
            print(epoch,loss)
        # Evaluation (prediction)
        predictions = self(X_test_tensor).detach().numpy()

        return predictions
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

