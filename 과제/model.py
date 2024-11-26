import torch.nn as nn

#클래스 정의
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        #CNN 계층 정리
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    # foward 함수
    def forward(self, x):
        # 입력 이미지 처리
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        # 특성을 1차원으로 펼침 (Flattening)
        x = x.view(-1, 64 * 8 * 8)
        # 완전 연결 레이어 (Fully Connected Layers)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        # 최종 출력
        output = self.fc2(x)

        return output
