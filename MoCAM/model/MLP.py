
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class MlpNet(nn.Module):
    def __init__(self, n_inputs, n_outputs, dropout=0.2):
        super(MlpNet, self).__init__()
        self.fc1 = nn.Linear(n_inputs, n_inputs, bias=True) # 입력층(13) -> 은닉층1(50)으로 가는 연산
        self.fc2 = nn.Linear(n_inputs, int(n_inputs/2), bias=True) # 은닉층1(50) -> 은닉층2(30)으로 가는 연산
        self.fc3 = nn.Linear(int(n_inputs/2), int(n_inputs/4), bias=True) # 은닉층1(50) -> 은닉층2(30)으로 가는 연산
        self.fc4 = nn.Linear(int(n_inputs/4), n_outputs, bias=True) # 은닉층2(30) -> 출력층(1)으로 가는 연산
        self.dropout = nn.Dropout(0.2) # 연산이 될 때마다 20%의 비율로 랜덤하게 노드를 없앤다.        
   
    def forward(self, x):
        x = F.relu(self.fc1(x)) # Linear 계산 후 활성화 함수 ReLU를 적용한다.  
        x = F.relu(self.fc2(x)) # Linear 계산 후 활성화 함수 ReLU를 적용한다.  
        x = self.dropout(F.relu(self.fc3(x))) # 은닉층2에서 드랍아웃을 적용한다.(즉, 30개의 20%인 6개의 노드가 계산에서 제외된다.)
        x = F.relu(self.fc4(x)) 
        return x
