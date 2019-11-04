import torch
import torch.nn as nn
import math

output = torch.randn(1, 5, requires_grad = True) #假设是网络的最后一层，5分类
label = torch.empty(1, dtype=torch.long).random_(5) # 0 - 4， 任意选取一个分类

loss = nn.CrossEntropyLoss()

# torch loss 1
print ('pytorch loss = ', loss(output, label))

# my loss
score = output [0,label.item()].item() # label对应的class的logits（得分）
print ('Score for the ground truth class = ', label)
first = - score
second = 0
for i in range(5):
    second += math.exp(output[0,i])
second = math.log(second)
loss = first + second
print ('-' * 20)
print ('my loss = ', loss)

# np test
np_output = output.detach().numpy()
np_label = label.numpy()

print ('Network Output is: ', np_output, np_output.shape)
print ('Ground Truth Label is: ', np_label, np_label.shape)

output = torch.from_numpy(np_output)
label = torch.from_numpy(np_label)

print ('pytorch loss = ', loss(output, label))