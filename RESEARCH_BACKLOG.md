# Research Backlog: Reinforcement Learning Implementations

주인님(Sangrok)의 지시에 따라 순차적으로 구현할 강화학습 논문 리스트입니다. 모든 모델은 Atari 환경 학습을 기본으로 가정합니다.

| 순번 | 논문 / 알고리즘 | 연도 | 상태 | 비고 |
| :--- | :--- | :--- | :--- | :--- |
| 1 | **Q-Learning** (Technical Note: Q-Learning) | 1992 | ✅ 완료 | Watkins & Dayan |
| 2 | **REINFORCE** (Simple Statistical Gradient-Following...) | 1992 | ✅ 완료 | Williams |
| 3 | **SARSA** (Online Q-Learning using Connectionist Systems) | 1994 | ✅ 완료 | Rummery & Niranjan |
| 4 | **DQN** (Human-level control through deep RL) | 2015 | ⏳ 대기 | Mnih et al. |
| 5 | **DDPG** (Continuous control with deep RL) | 2015 | ⏳ 대기 | Lillicrap et al. |
| 6 | **TRPO** (Trust Region Policy Optimization) | 2015 | ⏳ 대기 | Schulman et al. |
| 7 | **A3C** (Asynchronous Methods for Deep RL) | 2016 | ⏳ 대기 | Mnih et al. |
| 8 | **PPO** (Proximal Policy Optimization Algorithms) | 2017 | ⏳ 대기 | Schulman et al. |
| 9 | **SAC** (Soft Actor-Critic) | 2018 | ⏳ 대기 | Haarnoja et al. |
| 10 | **TD3** (Addressing Function Approximation Error...) | 2018 | ⏳ 대기 | Fujimoto et al. |

## 작업 지침
- 모든 구현은 `agent_guide.md`를 준수한다.
- `models/implementation/[논문명]/` 구조를 유지한다.
- Atari 환경(`gymnasium[atari]`)을 기본 환경으로 사용한다.
- TensorBoard를 통해 학습 과정(Reward, Loss, 가중치 분포 등)을 상세히 기록한다.
