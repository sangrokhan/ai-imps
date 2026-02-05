import torch
import torch.nn as nn
import torch.nn.functional as F
from core.base_model import BaseModel

class DENModel(BaseModel):
    """
    DER: Dynamically Expandable Representation for Class Incremental Learning (CVPR 2021)
    논문의 핵심 아키텍처를 구현한 모델 클래스입니다.
    사용자가 'DEN'으로 명칭하였으나, 제공된 논문 제목에 따라 DER의 동적 확장 구조를 따릅니다.
    """
    def __init__(self, config):
        super(DENModel, self).__init__(config)
        self.input_dim = config.get("input_dim", 784)
        self.hidden_dim = config.get("hidden_dim", 400)
        self.output_dim = config.get("output_dim", 10) # Initial output dim
        
        # 현재 활성화된 피처 추출기 리스트 (Task별로 확장됨)
        self.feature_extractors = nn.ModuleList([
            self._make_extractor(self.input_dim, self.hidden_dim)
        ])
        
        # 각 Task별 분류기 (마지막에 모든 피처를 통합하여 분류)
        self.classifier = nn.Linear(self.hidden_dim, self.output_dim)
        
        self.current_task = 0

    def _make_extractor(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.ReLU()
        )

    def expand_model(self, new_classes):
        """새로운 태스크 발생 시 모델 확장"""
        self.current_task += 1
        
        # 1. 이전 피처 추출기 동결
        for param in self.feature_extractors.parameters():
            param.requires_grad = False
            
        # 2. 새로운 피처 추출기 추가
        self.feature_extractors.append(self._make_extractor(self.input_dim, self.hidden_dim))
        
        # 3. 분류기 확장 (기존 지식 + 새로운 지식 결합)
        old_out_dim = self.classifier.out_features
        new_out_dim = old_out_dim + new_classes
        total_feature_dim = len(self.feature_extractors) * self.hidden_dim
        
        # 실제 논문에서는 채널 프루닝 등을 수행하지만, 여기서는 기본 확장 로직만 구현
        self.classifier = nn.Linear(total_feature_dim, new_out_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        
        # 모든 피처 추출기의 출력을 결합 (Concatenation)
        features = []
        for extractor in self.feature_extractors:
            features.append(extractor(x))
        
        combined_features = torch.cat(features, dim=1)
        
        # 확장된 분류기를 통해 최종 출력 계산
        logits = self.classifier(combined_features)
        return logits

    def compute_loss(self, outputs, targets):
        # 기본 CrossEntropy + 논문 권장 Auxiliary Loss (생략 가능)
        return F.cross_entropy(outputs, targets)
