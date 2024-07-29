import torch
from torch.utils.data import Dataset, DataLoader
from transformer_lens.hook_points import HookedRootModule


class RandomDataset(Dataset):
    def __init__(self, num_samples, input_size):
        self.num_samples = num_samples
        self.input_size = input_size
        self.data = torch.randn(num_samples, input_size)
        self.labels = torch.randint(0, 2, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# 创建数据集实例
dataset = RandomDataset(num_samples=1000, input_size=10)
# 创建数据加载器实例
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
import torch.nn as nn
import torch.optim as optim


class SimpleModel(HookedRootModule):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        return self.fc(x)


model = SimpleModel()  # 实例化模型
model = torch.nn.DataParallel(model)  # 并行化模型
model = model.cuda()  # 将模型移动到 GPU

# 定义损失函数和优化器
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# 训练模型
num_epochs = 5
for epoch in range(num_epochs):
    for inputs, targets in data_loader:
        # inputs, targets = inputs.cuda(), targets.cuda()  # 将数据移动到 GPU
        outputs = model(inputs)  # 前向传播
        loss = loss_fn(outputs, targets)  # 计算损失
        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 后向传播
        optimizer.step()  # 更新参数

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')
