import torch.nn as nn
from connectivity import *
from torch.utils.data import TensorDataset, DataLoader


class LatentNet(torch.nn.Module):
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

        self.recurrent_layer = nn.Linear(self.n, self.n, bias=False)
        self.recurrent_layer.weight.data.normal_(mean=0., std=0.025).to(device=device)
        #self.recurrent_layer.bias.data.normal_(mean=0.2, std=0).to(device=device)
        #self.recurrent_layer.bias.requires_grad = False

        self.input_layer = nn.Linear(self.input_size, self.n, bias=False)
        self.input_layer.weight.data.normal_(mean=0.2, std=.1).to(device=device)

        self.output_layer = nn.Linear(self.n, self.output_size, bias=False)
        self.output_layer.weight.data.normal_(mean=.2, std=0.1).to(device=device)

        self.init_hidden = torch.nn.Parameter(torch.zeros(self.n_trials, 1, self.n, device=device), requires_grad=False)
        self.device = device
        self.a = torch.nn.Parameter(torch.rand(self.N, self.N, device=device), requires_grad=True)
        self.q = self.cayley_transform(self.a)
        self.connectivity_masks()
    #
    # def init_connectivity(self):
    #     w_rec, w_in, w_out = generate_connectivity(self.n, self.input_size, self.output_size, self.device, radius=1.5)
    #     self.recurrent_layer.weight.data = w_rec
    #     self.input_layer.weight.data = w_in
    #     self.output_layer.weight.data = w_out

    def connectivity_masks(self):
        # Input mask
        input_mask = torch.zeros_like(self.input_layer.weight.data)
        input_mask[:self.input_size, :self.input_size] = torch.eye(self.input_size)
        self.input_layer.weight.data = input_mask * torch.relu(self.input_layer.weight.data)

        # Output mask
        output_mask = torch.zeros_like(self.output_layer.weight.data)
        output_mask[-self.output_size:, -self.output_size:] = torch.eye(self.output_size)
        self.output_layer.weight.data = output_mask * torch.relu(self.output_layer.weight.data)


    def cayley_transform(self, a):
        skew = (a - a.t()) / 2
        skew = skew.to(device=self.device)
        eye = torch.eye(self.N).to(device=self.device)
        o = (eye - skew) @ torch.inverse(eye + skew)
        return o[:self.n, :]

    def forward(self, u):
        t = u.shape[1]
        states = torch.zeros(u.shape[0], 1, self.n, device=self.device)
        batch_size = states.shape[0]

        noise = torch.sqrt(2 * self.alpha * self.sigma_rec ** 2) * torch.empty(batch_size, t, self.n).normal_(mean=0,
                                                                                                              std=1).to(
            device=self.device)

        for i in range(t - 1):
            state_new = (1 - self.alpha) * states[:, i, :] + self.alpha * (
                self.activation(
                    self.recurrent_layer(states[:, i, :]) + self.input_layer(u[:, i, :]) + noise[:, i, :]))
            states = torch.cat((states, state_new.unsqueeze_(1)), 1)

        return states

    def loss_function(self, x, z, y, l_y):
        return self.mse_z(x, z)+ l_y * self.nmse_y(y,x)

    def mse_z(self, x, z):


        return torch.sum(((self.output_layer(x) - z) )**2 ) / x.shape[0] / x.shape[1]

    def nmse_x(self, y, x):
        mse = nn.MSELoss(reduction='mean')
        y_bar = y - torch.mean(y, dim=[0, 1], keepdim=True)

        return mse(y @ self.q.t(), x) / mse(y_bar, torch.zeros_like(y_bar))

    def nmse_q(self, y):
        mse = nn.MSELoss(reduction='mean')
        y_bar = y - torch.mean(y, dim=[0, 1], keepdim=True)

        return mse(y @ self.q.t() @ self.q, y) / mse(y_bar, torch.zeros_like(y_bar))

    def nmse_y(self, y,x):
        mse = nn.MSELoss(reduction='mean')
        y_bar = y - torch.mean(y, dim=[0, 1], keepdim=True)
        return mse(x @ self.q, y) / mse(y_bar, torch.zeros_like(y_bar))



    def fit(self, u, z, y, epochs, lr,l_y,weight_decay):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr,weight_decay = weight_decay)
        my_dataset = TensorDataset(u, z, y)  # create your datset
        my_dataloader = DataLoader(my_dataset, batch_size=128)

        loss_history = []
        for i in range(epochs):
            # calculating loss as in the beginning of an epoch and storing it
            epoch_loss = 0
            for batch_idx, (u_batch, z_batch, y_batch) in enumerate(my_dataloader):
                optimizer.zero_grad()
                x_batch = self.forward(u_batch)
                loss = self.loss_function(x_batch, z_batch, y_batch,l_y)
                epoch_loss += loss.item() / epochs
                loss.backward()
                optimizer.step()
                self.q = self.cayley_transform(self.a)
                self.connectivity_masks()


            if i % 10 == 0:
                x = self.forward(u)
                print('Epoch: {}/{}.............'.format(i, epochs), end=' ')
                print("mse_z: {:.4f}".format(self.mse_z(x, z).item()), end=' ')
                print("nmse_y: {:.4f}".format(self.nmse_y(y,x).item()))
                loss_history.append(epoch_loss)
        return loss_history