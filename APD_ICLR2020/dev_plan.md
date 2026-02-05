Part 1. 논문 핵심 아이디어 분석 (Key Insights)
이 논문은 연속 학습(Continual Learning)의 고질적인 문제인 **파국적 망각(Catastrophic Forgetting)**과 **학습 순서 민감성(Order-sensitivity)**을 해결하기 위해 파라미터 분해(Parameter Decomposition) 기법을 제안합니다. 사용자님이 관심 가지셨던 **LoRA(Low-Rank Adaptation)**와 개념적으로 유사한 "Additive(가산적)" 접근 방식을 취합니다.
1. Additive Parameter Decomposition (APD)
기존 모델의 가중치(\mathbf{W})를 **공유 파라미터(Shared)**와 **태스크 별 파라미터(Task-specific)**의 합으로 정의합니다.

 * \mathbf{W}_{shared}: 모든 태스크에서 공유되는 지식 (General Knowledge).
 * \mathbf{W}_{task}^{(t)}: 특정 태스크 t에서만 필요한 지식 (Specific Knowledge).
 * \mathbf{M} (Masks): 각 파라미터가 공유될지, 태스크 전용일지를 결정하는 희소(Sparse) 마스크입니다.
2. Scalability (확장성)
모든 파라미터를 태스크마다 복사하는 것이 아니라, Sparse Mask와 작은 차분의 Task Parameter만 저장하므로 메모리 효율성이 높습니다. 이는 사용자님이 고민하시던 "대규모 모델의 효율적 관리"와 맥락이 닿아 있습니다.
3. Order-Robustness (순서 강건성)
기존 방식들은 Task A → Task B 순서로 학습할 때와 B → A로 학습할 때 성능 차이가 컸습니다. APD는 공유 지식과 태스크 지식을 명확히 분리함으로써, 학습 순서가 바뀌어도 최종 성능(Accuracy)의 변동폭이 매우 적습니다.
Part 2. 에이전트 위임용 개발 명세서 (Agent Task Specification)
이 문서를 복사하여 사용자님의 'Coding Agent' 또는 'AI Specialist Agent'에게 전달하시면 됩니다. PyTorch 환경에서의 구체적인 구현 가이드라인을 포함하고 있습니다.
[System Prompt]
당신은 PyTorch 기반의 Deep Learning Model Implementation Expert입니다. 아래의 요구사항(PRD)에 따라 논문의 아이디어를 코드로 구현하고, 검증 스크립트를 작성하십시오.
[Task Description]
"Scalable and order-robust continual learning with additive parameter decomposition" 논문의 핵심 알고리즘인 APD(Additive Parameter Decomposition) 레이어를 구현하고, Split-CIFAR100 벤치마크에서 연속 학습 성능을 테스트하십시오.
[Requirements Details]
1. Custom Layer Implementation (apd_layers.py)
기존 nn.Linear와 nn.Conv2d를 대체할 수 있는 APD 전용 레이어 클래스를 작성하십시오.
 * 클래스 구조: APDLinear(nn.Module), APDConv2d(nn.Module)
 * 파라미터 정의:
   * weight_shared: nn.Parameter, 모든 태스크 공유.
   * weight_task: nn.ParameterDict 또는 nn.ParameterList, 태스크 ID를 키(Key)로 가짐.
   * task_mask: 태스크별 희소 마스크 (Binary or Continuous).
 * Forward Logic:
   * 입력으로 task_id를 받습니다.
   * 수식: y = (W_{shared} \cdot M_s + W_{task}[id] \cdot M_t) x + b
   * Masking 연산을 통해 최종 Weight를 동적으로 구성한 후 연산을 수행합니다.
2. Model Architecture (model.py)
 * Backbone: ResNet-18 (축소된 버전 사용 가능) 또는 AlexNet.
 * 적용: 모델의 모든 Conv 및 Linear 레이어를 위에서 정의한 APDConv2d, APDLinear로 교체하십시오.
 * 유틸리티: 현재 task_id를 모델 전체에 전파(Broadcast)할 수 있는 메커니즘을 구현하십시오.
3. Training Strategy (Continual Learning Loop)
 * Dataset: CIFAR-100을 10개의 태스크(각 10 클래스)로 분할 (Split-CIFAR100).
 * Optimization:
   * 각 태스크 학습 시, 이전 태스크의 weight_task는 고정(Freeze)합니다.
   * weight_shared는 정규화(Regularization)를 강하게 적용하여 급격한 변화를 막습니다.
   * Mask에 대해 sparsity penalty(L1 norm 등)를 적용하여 저장 용량을 최적화하십시오.
4. Evaluation Metrics
학습 완료 후 다음 지표를 출력하는 함수를 작성하십시오.
 * Average Accuracy: 모든 태스크 학습 후, 각 태스크별 Test Set 정확도의 평균.
 * Forgetting Measure: F_{ij} (태스크 i 학습 직후 성능 - 태스크 j까지 학습 후 태스크 i의 성능)의 평균.
[Deliverables]
 * apd_layer.py: 커스텀 레이어 코드.
 * train_apd.py: 학습 및 평가 루프가 포함된 실행 스크립트.
 * requirements.txt: 필요한 라이브러리 목록.
