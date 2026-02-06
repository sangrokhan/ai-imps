import pytest
import torch
from core.registry import MODEL_REGISTRY
import implementations

@pytest.mark.parametrize("model_name", ["gcn", "graphsage", "gat", "graphormer"])
def test_gnn_registration(model_name):
    """GNN 모델들이 레지스트리에 정상 등록되었는지 확인"""
    assert model_name in MODEL_REGISTRY._registry

def test_gcn_forward():
    """GCN 모델 인스턴스화 및 Forward Pass 테스트"""
    config = {"input_dim": 16, "hidden_dim": 8, "output_dim": 2}
    model_cls = MODEL_REGISTRY.get("gcn")
    model = model_cls(config)
    
    N = 10
    x = torch.randn(N, 16)
    adj = torch.eye(N) # Dummy normalized adj
    
    out = model((x, adj))
    assert out.shape == (N, 2)

def test_graphsage_forward():
    """GraphSAGE 모델 인스턴스화 및 Forward Pass 테스트"""
    config = {"input_dim": 16, "hidden_dim": 8, "output_dim": 2}
    model_cls = MODEL_REGISTRY.get("graphsage")
    model = model_cls(config)
    
    N = 10
    x = torch.randn(N, 16)
    adj = torch.eye(N)
    
    out = model((x, adj))
    assert out.shape == (N, 2)

def test_gat_forward():
    """GAT 모델 인스턴스화 및 Forward Pass 테스트"""
    config = {"input_dim": 16, "hidden_dim": 8, "output_dim": 2, "n_heads": 4}
    model_cls = MODEL_REGISTRY.get("gat")
    model = model_cls(config)
    
    N = 5
    x = torch.randn(N, 16)
    adj = torch.ones(N, N)
    
    out = model((x, adj))
    assert out.shape == (N, 2)

def test_graphormer_forward():
    """Graphormer 모델 인스턴스화 및 Forward Pass 테스트"""
    d_model = 32
    config = {"input_dim": 16, "d_model": d_model, "output_dim": 2, "n_heads": 4, "n_layers": 2}
    model_cls = MODEL_REGISTRY.get("graphormer")
    model = model_cls(config)
    
    B, N = 2, 5
    x = torch.randn(B, N, 16)
    in_degree = torch.zeros(B, N, dtype=torch.long)
    out_degree = torch.zeros(B, N, dtype=torch.long)
    spatial_bias = torch.zeros(B * 4, N, N) # n_heads=4
    padding_mask = torch.zeros(B, N, dtype=torch.bool)
    
    out = model((x, in_degree, out_degree, spatial_bias, padding_mask))
    assert out.shape == (B, N, 2)
