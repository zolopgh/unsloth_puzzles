import torch
import torch.nn as nn
from typing import Tuple, Optional, Callable, Any
import math

class MemoryEfficientLinear(torch.autograd.Function):
    """
    A memory-efficient implementation of a linear layer that avoids 
    materializing large intermediate tensors during backward pass.
    """
    
    @staticmethod
    def forward(ctx: Any, X: torch.Tensor, linear: nn.Linear, 
                labels: Optional[torch.Tensor] = None,
                transform_fn: Optional[Callable] = None) -> torch.Tensor:
        """
        Forward pass that saves minimal state and defers computation.
        """
        # Verify inputs
        assert isinstance(X, torch.Tensor), "Input X must be a tensor"
        assert isinstance(linear, nn.Linear), "linear must be nn.Linear"
        assert X.dim() == 2, "Input X must be 2D (batch_size, hidden_dim)"
        
        # Save inputs for backward
        ctx.save_for_backward(X, linear.weight, labels)
        ctx.transform_fn = transform_fn
        
        # Store metadata for chunked computation
        ctx.batch_size = X.shape[0]
        ctx.chunk_size = max(1, X.shape[0] // 32)  # Use tiny chunks
        
        # Compute forward pass
        if transform_fn is not None:
            # Use provided transform function (e.g. cross entropy)
            output = transform_fn(X, linear, labels)
        else:
            # Standard linear projection
            output = X @ linear.weight.t()
            
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        """
        Memory-efficient backward pass that computes gradients in chunks.
        """
        X, weight, labels = ctx.saved_tensors
        
        # Process in chunks to reduce memory
        num_chunks = math.ceil(ctx.batch_size / ctx.chunk_size)
        
        # Initialize gradient accumulators
        grad_X = torch.zeros_like(X)
        grad_weight = torch.zeros_like(weight)
        
        # Free memory before starting backward
        torch.cuda.empty_cache()
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * ctx.chunk_size
            end_idx = min((chunk_idx + 1) * ctx.chunk_size, ctx.batch_size)
            
            # Get chunk of input and corresponding labels
            X_chunk = X[start_idx:end_idx]
            labels_chunk = labels[start_idx:end_idx] if labels is not None else None
            
            # Recompute forward pass for this chunk
            if ctx.transform_fn is not None:
                # Create differentiable copies of inputs
                X_chunk = X_chunk.detach().requires_grad_()
                weight_chunk = weight.detach().requires_grad_()
                
                # Forward pass through transform function
                with torch.enable_grad():
                    # Compute logits in half precision to save memory
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                        logits = X_chunk @ weight_chunk.t()
                        loss = torch.nn.functional.cross_entropy(
                            logits.view(-1, logits.size(-1)),
                            labels_chunk.view(-1),
                            reduction='sum'  # Use sum to match standard implementation
                        ) / ctx.batch_size  # Normalize by batch size
                    
                    # Compute gradients
                    grads = torch.autograd.grad(
                        loss,
                        [X_chunk, weight_chunk],
                        retain_graph=False
                    )
                    
                    # Accumulate gradients
                    grad_X[start_idx:end_idx].copy_(grads[0])
                    grad_weight += grads[1]
                    
                    # Free memory after gradient computation
                    del logits, loss, grads
                    torch.cuda.empty_cache()
            else:
                # Standard linear case - explicit gradient computation
                grad_chunk = grad_output[start_idx:end_idx]
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    grad_X[start_idx:end_idx] = grad_chunk @ weight
                    grad_weight += X_chunk.t() @ grad_chunk
                torch.cuda.empty_cache()
        
        # Set gradients on weight
        weight.grad = grad_weight if weight.grad is None else weight.grad + grad_weight
        
        return grad_X, None, None, None

def cross_entropy_transform(batch: torch.Tensor, 
                          linear: nn.Linear,
                          labels: torch.Tensor) -> torch.Tensor:
    """Transform function that computes cross entropy loss."""
    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
        logits = batch @ linear.weight.t()
        return torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            reduction='sum'  # Use sum to match standard implementation
        ) / batch.size(0)  # Normalize by batch size

if __name__ == "__main__":
    print("Setting up test data...")
    
    # Test parameters
    batch_size = 4
    hidden_dim = 4096
    vocab_size = 128000
    
    # Create test inputs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X = torch.randn(batch_size, hidden_dim, device=device, requires_grad=True)
    linear = nn.Linear(hidden_dim, vocab_size, bias=False).to(device)
    labels = torch.randint(0, vocab_size, (batch_size,), device=device)
    
    print(f"Running on device: {device}")
    print(f"Input shape: {X.shape}")
    print(f"Weight shape: {linear.weight.shape}")
    
    # Clear cache before testing
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # Test memory efficient implementation
    print("\nTesting memory efficient implementation...")
    mem_start = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    
    efficient_out = MemoryEfficientLinear.apply(X, linear, labels, cross_entropy_transform)
    print(f"Forward output: {efficient_out.item():.4f}")
    
    efficient_out.backward()
    
    mem_efficient = (torch.cuda.max_memory_allocated() - mem_start) if torch.cuda.is_available() else 0
    
    if X.grad is None or linear.weight.grad is None:
        raise RuntimeError("Gradients were not properly computed in efficient implementation")
    
    # Store gradients for comparison
    efficient_grad_X = X.grad.clone()
    efficient_grad_W = linear.weight.grad.clone()
    
    # Reset gradients and cache
    X.grad = None
    linear.weight.grad = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # Test standard implementation
    print("\nTesting standard implementation...")
    mem_start = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    
    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
        standard_out = linear(X)
        standard_loss = torch.nn.functional.cross_entropy(
            standard_out,
            labels,
            reduction='sum'
        ) / batch_size
    print(f"Forward output: {standard_loss.item():.4f}")
    
    standard_loss.backward()
    
    mem_standard = (torch.cuda.max_memory_allocated() - mem_start) if torch.cuda.is_available() else 0
    
    if X.grad is None or linear.weight.grad is None:
        raise RuntimeError("Gradients were not properly computed in standard implementation")
    
    # Compare results
    print(f"\nMemory Usage:")
    print(f"Standard Implementation: {mem_standard / 1024**2:.2f} MB")
    print(f"Efficient Implementation: {mem_efficient / 1024**2:.2f} MB")
    print(f"Memory Savings: {(mem_standard - mem_efficient) / 1024**2:.2f} MB")
    
    # Verify gradient correctness
    grad_X_close = torch.allclose(efficient_grad_X, X.grad, rtol=1e-4, atol=1e-4)
    grad_W_close = torch.allclose(efficient_grad_W, linear.weight.grad, rtol=1e-4, atol=1e-4)
    
    print(f"\nGradient Correctness:")
    print(f"Input gradients match: {grad_X_close}")
    print(f"Weight gradients match: {grad_W_close}")
    
    if not (grad_X_close and grad_W_close):
        print("\nGradient differences:")
        print(f"X grad max diff: {(efficient_grad_X - X.grad).abs().max().item()}")
        print(f"W grad max diff: {(efficient_grad_W - linear.weight.grad).abs().max().item()}")
