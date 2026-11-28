# Reproduction of Dao-AILab/flash-attention#3179 (script from the issue, plus the module path, error numbers and a contiguous control).
import sys, torch, torch.nn.functional as F
import fa_noext  # registers flash_attn from FA_SRC without the CUDA extension
import flash_attn.ops.triton.cross_entropy as ce
from flash_attn.ops.triton.cross_entropy import cross_entropy_loss
print("module:", ce.__file__)
torch.set_default_device("cuda"); torch.manual_seed(0)
logits = torch.tensor([[3.,1.,0.],[0.,3.,1.],[1.,0.,3.]], requires_grad=True)
labels = torch.tensor([0,2,1,0,2,1], dtype=torch.int64)[::2]
actual,_ = cross_entropy_loss(logits, labels)
expected = F.cross_entropy(logits, labels, reduction="none")
ag, = torch.autograd.grad(actual.sum(), logits); eg, = torch.autograd.grad(expected.sum(), logits)
control,_ = cross_entropy_loss(logits, labels.contiguous())
cg, = torch.autograd.grad(control.sum(), logits)
print("labels:", labels.tolist(), "stride:", labels.stride(), "storage_offset:", labels.storage_offset())
print("observed loss:", [round(v,6) for v in actual.tolist()])
print("expected loss:", [round(v,6) for v in expected.tolist()])
print("max abs loss error (strided):", (actual-expected).abs().max().item())
print("max abs grad error (strided):", (ag-eg).abs().max().item())
print("max abs loss error (contiguous control):", (control-expected).abs().max().item())
print("max abs grad error (contiguous control):", (cg-eg).abs().max().item())
ok = torch.allclose(actual,expected,atol=1e-6) and torch.allclose(ag,eg,atol=1e-6)
print("RESULT", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
