import torch.nn as nn
from connectivity import *
from torch.utils.data import TensorDataset, DataLoader


class LatentNet(torch.nn.Module):
    '''
    Pytorch module for implementing a latent circuit model
    '''
    def __init__(self, n, N, n_trials, sigma_rec=.15, input_size=6, output_size=2, device='cpu'):
        super(LatentNet, self).__init__()
        self.alpha = .2
        self.sigma_rec = torch.tensor(sigma_rec)
        self.n = n
        self.N = N
        self.n_trials = n_trials
        self.input_size = input_size
        self.output_size = output_size
        self.activation = torch.nn.ReLU()
        self.device = device

        # Initialize connectivity layers
        self.recurrent_layer = nn.Linear(self.n, self.n, bias=True)
        self.recurrent_layer.weight.data.normal_(mean=0., std=0.025).to(device=self.device)
        self.recurrent_layer.bias.data.normal_(mean=0.2, std=0).to(device=device)
        self.recurrent_layer.bias.requires_grad = False
        self.input_layer = nn.Linear(self.input_size, self.n, bias=False)
        self.input_layer.weight.data.normal_(mean=0.2, std=.01).to(device=self.device)
        self.output_layer = nn.Linear(self.n, self.output_size, bias=False)
        self.output_layer.weight.data.normal_(mean=.2, std=0.01).to(device=self.device)

        # Initialize embedding matrix, q
        self.a = torch.nn.Parameter(torch.rand(self.N, self.N, device=self.device), requires_grad=True)
        self.q = self.cayley_transform(self.a)

        # Apply connectivity masks to initialized connectivity
        self.connectivity_masks()

    def connectivity_masks(self):
        # These masks are applied after each gradient update during training.
        # Input mask
        input_mask = torch.zeros_like(self.input_layer.weight.data)
        input_mask[:self.input_size, :self.input_size] = torch.eye(self.input_size)
        self.input_layer.weight.data = input_mask * torch.relu(self.input_layer.weight.data)

        # Output mask
        output_mask = torch.zeros_like(self.output_layer.weight.data)
        output_mask[-self.output_size:, -self.output_size:] = torch.eye(self.output_size)
        self.output_layer.weight.data = output_mask * torch.relu(self.output_layer.weight.data)


    def cayley_transform(self, a):
        # Tranform square matrix, a, into orthonormal matrix, q.
        skew = (a - a.t()) / 2
        skew = skew.to(device=self.device)
        eye = torch.eye(self.N).to(device=self.device)
        o = (eye - skew) @ torch.inverse(eye + skew)
        return o[:self.n, :]

    def forward(self, u):
        # total timesteps
        t = u.shape[1]

        # Initialize hidden states to zero
        states = torch.zeros(u.shape[0], 1, self.n, device=self.device)
        batch_size = states.shape[0]

        # Noise to be added at each time step
        noise = torch.sqrt(2 * self.alpha * self.sigma_rec ** 2) * torch.empty(batch_size, t, self.n).normal_(mean=0,
                                                                                                              std=1).to(
            device=self.device)

        # Loop over time steps
        for i in range(t - 1):
            state_new = (1 - self.alpha) * states[:, i, :] + self.alpha * (
                self.activation(
                    self.recurrent_layer(states[:, i, :]) + self.input_layer(u[:, i, :]) + noise[:, i, :]))
            states = torch.cat((states, state_new.unsqueeze_(1)), 1)
        return states

    def loss_function(self, x, z, y, l_y, y_mask):
        # Loss function penalizes errors in outputs and hidden states.
        return self.mse_z(x, z)+ l_y * self.nmse_y(y,x,y_mask)

    def mse_z(self, x, z):

        # Mean squared error for task performance.
        return torch.sum((( (self.output_layer(x) - z)) )**2 ) / x.shape[0] / x.shape[1]

    def nmse_x(self, y, x,mask):
        mask = mask[:x.shape[0], :]
        # Normalized mean squared error between projected trajectories (hidden states) and latents.
        mse = nn.MSELoss(reduction='mean')
        y_bar = y - torch.mean(y, dim=[0, 1], keepdim=True)
        x = (x - torch.mean(x, dim=[0, 1], keepdim=True)) / torch.std(x, dim=[0, 1], keepdim=True)
        return mse((y * mask) @ self.q.t(), (mask*(x @ self.q))@ self.q.t()) / mse(mask * y_bar, torch.zeros_like(mask * y_bar))

    def nmse_q(self, y):
        # Normalized mean squared error between trajectories and projected trajectories. This error can be interpreted
        # as variance not explained by the subspace spanned by the columns of Q.
        mse = nn.MSELoss(reduction='mean')
        y_bar = y - torch.mean(y, dim=[0, 1], keepdim=True)
        return mse(y @ self.q.t() @ self.q, y) / mse(y_bar, torch.zeros_like(y_bar))

    def nmse_y(self, y,x, mask):
        mask = mask[:x.shape[0], :]
        # Mean squared error between trajectories and embedded trajectories of the latent circuit model.
        mse = nn.MSELoss(reduction='mean')
        y_bar = y - torch.mean(y, dim=[0, 1], keepdim=True)

        # z-score x
        x = (x - torch.mean(x,dim=[0, 1], keepdim=True)) / torch.std(x, dim=[0, 1], keepdim=True)
        return mse(mask*(x @ self.q), mask * y) / mse(mask * y_bar, torch.zeros_like(mask * y_bar))

    # Function for fitting latent circuit model.
    def fit(self, u, z, y, epochs, lr,l_y,weight_decay, y_mask):

        # Initialize optimizer and wrap training data as PyTorch dataset
        optimizer = torch.optim.Adam(self.parameters(), lr=lr,weight_decay = weight_decay)
        my_dataset = TensorDataset(u, z, y)  #
        my_dataloader = DataLoader(my_dataset, batch_size=128)

        # Training loop
        loss_history = []
        for i in range(epochs):
            epoch_loss = 0
            for batch_idx, (u_batch, z_batch, y_batch) in enumerate(my_dataloader):
                optimizer.zero_grad()
                x_batch = self.forward(u_batch)
                loss = self.loss_function(x_batch, z_batch, y_batch,l_y,y_mask)
                epoch_loss += loss.item() / epochs
                loss.backward()
                optimizer.step()

                # Re-calculate embedding matrix q after matrix a is updated.
                self.q = self.cayley_transform(self.a)

                # Re-apply connectivity masks after each gradient step.
                self.connectivity_masks()

            if i % 10 == 0:
                x = self.forward(u)
                print('Epoch: {}/{}.............'.format(i, epochs), end=' ')
                print("mse_z: {:.4f}".format(self.mse_z(x, z).item()), end=' ')
                print("nmse_y: {:.4f}".format(self.nmse_y(y,x,y_mask).item()))
                loss_history.append(epoch_loss)
        return loss_history