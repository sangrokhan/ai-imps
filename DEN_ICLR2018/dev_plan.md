📑 Implementation Guide: Dynamically Expandable Networks (DEN)
이 문서는 태스크별 가중치 관리, 가로 확장, 그리고 모든 레이어에 대한 Split & Duplication 기능을 포함한 DEN의 PyTorch 구현 가이드입니다.
1. 아키텍처 설계 원칙
 * Heterogeneous Task Handling: nn.ModuleDict를 활용하여 태스크마다 다른 입력(D_{in}) 및 출력(D_{out}) 차원을 처리하는 독립적인 Head와 Tail을 생성합니다.
 * Modular Training: 학습 로직을 모델 클래스와 분리하여 데이터, 손실 함수, 하이퍼파라미터를 유연하게 주입할 수 있도록 설계합니다.
 * Dynamic Weight Management: 가중치 저장 경로를 사용자가 지정하며, 태스크별로 별도의 가중치 파일(.pt)을 관리합니다.
2. 주요 하이퍼파라미터 설정 (Config)
학습 시 인자로 전달하거나 별도의 YAML/JSON으로 관리할 하이퍼파라미터 항목입니다.
| 항목 | 설명 | 권장 초기값 |
|---|---|---|
| l1_lambda | Selective Retraining 시 Sparse 연결을 유도하는 L_1 계수 | 10^{-3} \sim 10^{-4} |
| expansion_threshold | 네트워크 확장을 결정하는 손실(Loss) 임계값 | 사용자 지정 |
| drift_threshold (\epsilon) | 뉴런 분할(Split)을 결정하는 가중치 변화량 임계값 | 0.01 \sim 0.1 |
| new_nodes_k | 확장 시 레이어당 추가할 뉴런 수 | 10 \sim 20 |
3. 단계별 구현 가이드
[Step 1] 인프라 및 가중치 관리 로직
가중치 저장 위치를 동적으로 설정하고, 태스크별 입출력 층을 생성합니다.
import torch
import torch.nn as nn
import os

class DEN(nn.Module):
    def __init__(self, hidden_dims):
        super().__init__()
        self.hidden_dims = hidden_dims # List of hidden units per layer
        self.shared_layers = nn.ModuleList() # Hidden layers
        self.task_inputs = nn.ModuleDict()  # Task-specific input heads
        self.task_outputs = nn.ModuleDict() # Task-specific output tails

    def add_task_layer(self, task_id, in_dim, out_dim):
        """태스크별 입출력 층 동적 생성"""
        if task_id not in self.task_inputs:
            self.task_inputs[task_id] = nn.Linear(in_dim, self.hidden_dims[0])
            self.task_outputs[task_id] = nn.Linear(self.hidden_dims[-1], out_dim)

    def save_weights(self, task_id, save_dir):
        """지정된 경로에 태스크별 가중치 저장"""
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"{task_id}.pt")
        torch.save({
            'state_dict': self.state_dict(),
            'hidden_dims': self.hidden_dims
        }, path)

    def load_weights(self, task_id, save_dir):
        path = os.path.join(save_dir, f"{task_id}.pt")
        if os.path.exists(path):
            checkpoint = torch.load(path)
            # 레이어 크기가 변했을 수 있으므로 strict=False 사용 또는 구조 재구성 필요
            self.load_state_dict(checkpoint['state_dict'], strict=False)

[Step 2] 별도 학습 함수 및 Selective Retraining
모델 외부에서 학습을 제어하며 L_1 정규화를 적용합니다.
def train_den_step(model, train_loader, criterion, optimizer_fn, config):
    """
    model: DEN 모델 객체
    criterion: Loss 함수 (예: nn.CrossEntropyLoss)
    optimizer_fn: 옵티마이저 생성 팩토리 (예: lambda p: torch.optim.Adam(p, lr=1e-3))
    config: 하이퍼파라미터 딕셔너리
    """
    model.train()
    # 1. Selective Retraining을 위한 파라미터 필터링 (현재 태스크 관련만)
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optimizer_fn(params)

    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        
        # 기본 Loss + L1 Regularization
        l1_reg = sum(p.abs().sum() for p in model.parameters())
        loss = criterion(output, target) + config['l1_lambda'] * l1_reg
        
        loss.backward()
        optimizer.step()
    
    return loss.item()

[Step 3] Dynamic Expansion (가로 확장)
모든 히든 레이어의 차원을 가로로 확장합니다.
def expand_network(model, k):
    """모든 히든 레이어에 k개의 뉴런 추가"""
    with torch.no_grad():
        for i, layer in enumerate(model.shared_layers):
            # 1. 현재 레이어의 Output 차원 확장
            # 2. 다음 레이어의 Input 차원 확장 (연쇄 적용)
            # Linear 레이어의 weight 파라미터를 torch.cat으로 확장된 새 파라미터로 교체
            pass 
    # 확장 후 파라미터 수가 변하므로 옵티마이저는 반드시 새로 생성해야 합니다.

[Step 4] Split & Duplication (전체 레이어 적용)
학습 전후 가중치를 비교하여 지식 손상을 방지합니다.
 * 학습 전 old_state = model.state_dict()를 복사합니다.
 * 학습 후 각 뉴런(Row)별로 유클리드 거리를 계산합니다.
 * 임계값(\epsilon)을 넘는 뉴런 인덱스를 찾아 expand_network와 유사한 방식으로 해당 뉴런만 복제하여 레이어 크기를 키웁니다.
 * 복제된 뉴런 중 하나에는 old_state의 가중치를 할당하여 이전 지식을 보존합니다.
4. 구현 시 주의사항
 * Weight Mapping: 레이어가 확장되면 이전 태스크의 뉴런 인덱스가 달라질 수 있습니다. 각 태스크가 어떤 뉴런 인덱스를 사용하는지 맵핑 테이블(dict)을 가중치와 함께 저장하는 것이 필수적입니다.
 * Zero-Grad Masking: Selective Retraining 시 선택되지 않은 뉴런의 가중치가 변하지 않도록 grad를 강제로 0으로 만드는 마스킹 로직을 optimizer.step() 직전에 추가하세요.
