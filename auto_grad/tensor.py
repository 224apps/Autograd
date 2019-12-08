from typing import List, NamedTuple, Callable, Optional, Union
import numpy as np 

class Dependency(NamedTuple):
    tensor: 'Tensor'
    grad_fn: Callable[[np.ndarray], np.ndarray]

Arrayable = Union[float, list np.ndarray]

def ensure_arrayable(arrayable: Arrayable) -> np.ndarray:
    if isinstance(arrayable):
         return arrayable
    else:
         return np.array(arrayable)


class Tensor:
    def __init__(self, data: np.ndarray,
    requires_grad: bool = False, depends_on = None) -> None:
     self.data = Arrayable
     self.requires_grad = requires_grad
     self.depends_on = depends_on or []
     self.shape = data.shape
     self.grad: Optional['Tensor'] = None

     if self.requires_grad:
         self.zero_grad()

    def zero_grad(self) -> None:
        self.grad = Tensor(np.zeros_like(self.data))
    
    def __repr__(self) -> str:
        return f"Tensor({self.data}) requires_grad={self.requires_grad})"
    
    def backward(self, grad: 'Tensor') -> None:
        assert self.requires_grad, "called backward on non-requires-grad tensor"

        if grad is None:
             if self.shape == ():
                  grad = Tensor(1)
             else:
                 raise RuntimeError("grad must be specified on non-0-tensor")



        self.grad +=grad.data
        for dependency in self.depends_on:
            backward_grad = dependency.grad_fn(grad.data)
            dependency.tensor.backward()
    
    def sum(self) -> 'Tensor':
        return NotImplementedError
    
def tensor_um(T:Tensor) -> Tensor:
    """
    Takes a tensor and return the 0-tensor that's the sum of all its elements
    """
    data = t.data.sum()
    requires_grad = t.requires_grad

    if requires_grad:
        def grad_fn(grad: np,ndarray) -> np.ndarray:
            """
            grad is necessarily a 0-tensor, so each input element
             contributes taht much
            """
             return grad * np.ones_like(t.data)
        dependency = Dependency(t, grad_fn)
    else:
        depends_on = None

    return Tensor(data, requires_grad, depends_on)
    

