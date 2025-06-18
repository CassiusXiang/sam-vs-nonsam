import torch
import torch.nn as nn
from typing import Optional, Callable, Dict, Any


class SAM(torch.optim.Optimizer):
    """
    Sharpness-Aware Minimization (SAM) 优化器
    
    该优化器通过同时最小化损失值和损失锐度来提升模型的泛化能力。
    实现基于论文: "Sharpness-Aware Minimization for Efficiently Improving Generalization"
    
    参数:
        params: 需要优化的参数
        base_optimizer: 基础优化器类（如SGD、Adam等）
        rho: 感知域半径，控制扰动大小
        adaptive: 是否使用自适应扰动
        **kwargs: 传递给基础优化器的其他参数
    """
    
    def __init__(self, params, base_optimizer, rho: float = 0.05, adaptive: bool = False, **kwargs):
        # 确保rho为非负值
        assert rho >= 0.0, f"rho必须为非负值，当前值: {rho}"
        
        # 初始化优化器参数
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        
        # 初始化基础优化器
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        """
        第一步：计算并应用扰动
        
        参数:
            zero_grad: 是否在应用扰动后清零梯度
        """
        # 计算梯度范数
        grad_norm = self._grad_norm()
        
        # 对每个参数组应用扰动
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)  # 添加小量防止除零
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                    
                # 保存原始参数
                self.state[p]["old_p"] = p.data.clone()
                
                # 计算扰动
                if group["adaptive"]:
                    # 自适应扰动：考虑参数大小
                    e_w = torch.pow(p, 2) * p.grad * scale.to(p)
                else:
                    # 非自适应扰动：仅考虑梯度
                    e_w = p.grad * scale.to(p)
                
                # 应用扰动
                p.add_(e_w)  # w + e(w)
        
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        """
        第二步：恢复参数并执行实际更新
        
        参数:
            zero_grad: 是否在更新后清零梯度
        """
        # 恢复原始参数
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]  # 恢复到原始参数 w
        
        # 执行基础优化器的更新步骤
        self.base_optimizer.step()
        
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """
        执行一步优化
        
        参数:
            closure: 计算损失的闭包函数
            
        返回:
            如果提供了closure，返回损失值；否则返回None
        """
        assert closure is not None, "SAM优化器需要closure函数来计算损失"
        
        # 启用梯度计算
        closure = torch.enable_grad()(closure)
        
        # 执行两步优化
        self.first_step(zero_grad=True)
        loss = closure()
        self.second_step()
        
        return loss

    def _grad_norm(self) -> torch.Tensor:
        """
        计算所有参数梯度的L2范数
        
        返回:
            梯度范数
        """
        # 获取共享设备（用于模型并行）
        shared_device = self.param_groups[0]["params"][0].device
        
        # 计算每个参数的梯度范数
        grad_norms = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                    
                # 根据是否使用自适应扰动计算范数
                if group["adaptive"]:
                    grad_norm = (torch.abs(p) * p.grad).norm(p=2)
                else:
                    grad_norm = p.grad.norm(p=2)
                    
                grad_norms.append(grad_norm.to(shared_device))
        
        # 计算总范数
        return torch.norm(torch.stack(grad_norms), p=2)

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        加载优化器状态
        
        参数:
            state_dict: 优化器状态字典
        """
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
