import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
import torch_geometric.nn as geo_nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def get_dataset(save_path):
	'''
	read data from .npy file 
	no need to modify this function
	'''
	raw_data = np.load(save_path, allow_pickle=True)
	dataset = []
	for i, (node_f, edge_index, edge_attr, y)in enumerate(raw_data):
		sample = Data(
		x=torch.tensor(node_f, dtype=torch.float),
		y=torch.tensor([y], dtype=torch.float),
		edge_index=torch.tensor(edge_index, dtype=torch.long),
		edge_attr=torch.tensor(edge_attr, dtype=torch.float)
		)
		dataset.append(sample)
	return dataset


class GraphNet(nn.Module):
	'''
	Graph Neural Network class
	'''
	def __init__(self, n_features, lr=1e-3):
		'''
		n_features: number of features from dataset, should be 37
		'''
		super(GraphNet, self).__init__()
		# define your GNN model here
		self.conv1 = geo_nn.GCNConv(n_features, 64, cached=False, normalize=True)
		self.conv2 = geo_nn.GCNConv(64, 128, cached=False, normalize=True)
		self.conv3 = geo_nn.GCNConv(128, 256, cached=False, normalize=True)

		self.linear1 = nn.Linear(256,128)
		self.bn1 = nn.BatchNorm1d(128)
		self.linear2 = nn.Linear(128,1)
		

		self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
		
	def forward(self, x):
		# define the forward pass here
		x, edge_index, batch = x.x, x.edge_index, x.batch
		x = F.relu(self.conv1(x, edge_index))
		x = F.dropout(x, training=self.training)
		x = F.relu(self.conv2(x, edge_index))
		x = F.dropout(x, training=self.training)
		x = F.relu(self.conv3(x, edge_index))


		x = F.relu(self.linear1(geo_nn.global_max_pool(x, batch)))
		x = self.linear2(self.bn1(x))
		return x.view(-1)


def main():
	# load data and build the data loader
	train_set = get_dataset('train_set.npy')
	test_set = get_dataset('test_set.npy')
	train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
	test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

	# number of features in the dataset
	# no need to change the value
	n_features = 37

	# build your GNN model
	model = GraphNet(n_features)

	# define your loss and optimizer
	# loss_func = ...
	loss_func = nn.MSELoss()

	print(model)

	hist = {"train_loss":[], "test_loss":[]}
	num_epoch = 50
	for epoch in range(1, 1+num_epoch):
		model.train()
		loss_all = 0
		for data in train_loader:
			# your codes for training the model

			output = model(data)

			loss = loss_func(output, data.y)
			loss_all += loss.item() * data.num_graphs * len(data)

			model.optimizer.zero_grad()
			loss.backward()
			model.optimizer.step()

		train_loss = loss_all / len(train_set)

		with torch.no_grad():
			loss_all = 0
			for data in test_loader:
				# your codes for validation on test set
				# ...
				validation_out = model(data)
				loss = loss_func(validation_out, data.y)
				loss_all += loss.item() * data.num_graphs * len(data)
			test_loss = loss_all / len(test_set)

			hist["train_loss"].append(train_loss)
			hist["test_loss"].append(test_loss)
			print(f'Epoch: {epoch}, Train loss: {train_loss:.3}, Test loss: {test_loss:.3}')

			if epoch % 9 == 0:
				torch.save({"model": model.state_dict(),}, "p1_model.ckpt")
			# test on test set to get prediction 
	with torch.no_grad():
		prediction = np.zeros(len(test_set))
		label = np.zeros(len(test_set))
		idx = 0
		for data in test_loader:
			data = data.to(device)
			output = model(data)
			prediction[idx:idx+len(output)] = output.squeeze().detach().numpy()
			label[idx:idx+len(output)] = data.y.detach().numpy()
			idx += len(output)
		prediction = np.array(prediction).squeeze()
		label = np.array(label).squeeze()
		ms_sum_error = np.sum((prediction-label)**2)
		ms_error = np.mean((prediction-label)**2)
		
		print("ms_error: {}".format(ms_error))
		print("ms_sum_error: {}".format(ms_sum_error))
	# visualization
	# plot loss function
	ax = plt.subplot(1,1,1)
	ax.plot([e for e in range(1,1+num_epoch)], hist["train_loss"], label="train loss")
	ax.plot([e for e in range(1,1+num_epoch)], hist["test_loss"], label="test loss")
	plt.xlabel("epoch")
	plt.ylabel("loss")
	ax.legend()
	plt.savefig("q1_loss.png")
	plt.show()
	plt.close()
		
	# plot prediction vs. label
	x = np.linspace(np.min(label), np.max(label))
	y = np.linspace(np.min(label), np.max(label))
	ax = plt.subplot(1,1,1)
	ax.scatter(prediction, label, marker='+', c='red')
	ax.plot(x, y, '--')
	plt.xlabel("prediction")
	plt.ylabel("label")
	plt.savefig("q1_preds")
	plt.show()
	plt.close()

if __name__ == "__main__":
	main()
