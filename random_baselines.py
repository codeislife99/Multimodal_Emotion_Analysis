from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from gt_mosei_dataloader import gt_mosei
import os

class Random_Net(nn.Module):
	def __init__(self):
		super(Random_Net, self).__init__()
		self.x = Parameter(torch.FloatTensor([0,0,0,0,0,0]))

	def forward(self,y):
		return self.x.unsqueeze(0).expand(y.size(0),y.size(1))
'----------------------------------------------------------------------------------------------------------------------'
Random = Random_Net()
Random = Random.cuda()
# criterion = nn.MSELoss(size_average = False) # MSE
# criterion = nn.L1Loss(size_average = False) # MAE
criterion = nn.SmoothL1Loss(size_average=False) # Huber Loss
params =  list(Random.parameters())
optimizer = torch.optim.Adam(params, lr = 0.001)
num_workers = 0
test_mode = True
val_mode = False
train_mode = False
use_pretrained = True
batch_size = 100
no_of_epochs = 1000

'----------------------------------------------------------------------------------------------------------------------'
if train_mode:
	train_dataset = gt_mosei(mode= "train")
	data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                        batch_size=batch_size,
                                        shuffle=True,num_workers = num_workers)
elif val_mode:
	val_dataset = gt_mosei(mode = "val")
	data_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                        batch_size=batch_size,
                                        shuffle=False,num_workers = num_workers)
	no_of_epochs = 1
else:
	test_dataset = gt_mosei(mode = "test")
	data_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                        batch_size=batch_size,
                                        shuffle=False,num_workers = num_workers)
	no_of_epochs = 1

'------------------------------------------Saving Intermediate Models--------------------------------------------------'

def save_checkpoint(state, filename = 'Huber_Loss_sum'):
	filename = filename + '.pth.tar'
	os.system("mkdir -p random_models")
	torch.save(state, './random_models/'+filename)
'----------------------------------------------------------------------------------------------------------------------'

# y = Variable(torch.FloatTensor([1,2,3,4,5,6])).cuda()
epoch = 0
while epoch < no_of_epochs:

	if use_pretrained:
		# pretrained_file = './DAN/dual_attention_net_iter_8000_0.pth.tar'
		pretrained_file = './random_models/Huber_Loss_sum.pth.tar'

		checkpoint = torch.load(pretrained_file)
		Random.load_state_dict(checkpoint['Random'])
		use_pretrained = False
		if train_mode:
			epoch = checkpoint['epoch']+1
			optimizer.load_state_dict(checkpoint['optimizer'])

	running_loss = 0.0 
	K = 0
	for i,gt in enumerate(data_loader):
		gt = Variable(gt.float()).cuda()
		x = Random(gt)
		x = torch.clamp(x,0,3)
		loss = criterion(x,gt)
		if train_mode:
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()

		running_loss += loss.data[0]
		K+=batch_size
		average_loss = running_loss/K
	if train_mode:			
		print('Training -- Epoch [%d], Sample [%d], Average Loss: %.4f'
		% (epoch+1, K, average_loss))
	elif val_mode:
		print('Validating -- Epoch [%d], Sample [%d], Average Loss: %.4f'
		% (epoch+1, K, average_loss))
	else:
		print('Testing -- Epoch [%d], Sample [%d], Average Loss: %.4f'
		 % (epoch+1, K, average_loss))
	# print(x)
	if train_mode:
		save_checkpoint({
			'epoch': epoch,
			'loss' : average_loss,
			'Random' : Random.state_dict(),
			'optimizer': optimizer.state_dict(),
		})
	epoch+=1