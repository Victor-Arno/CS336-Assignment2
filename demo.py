import torch
from cs336_basics.model import BasicsTransformerLM

model = BasicsTransformerLM(
    vocab_size=1000, context_length=128, d_model=256,
    num_layers=2, num_heads=4, d_ff=512, rope_theta=10000.0
).cuda()

x = torch.randint(0, 1000, (1, 128)).cuda()

out1 = model(x)
print('Without autocast: output dtype =', out1.dtype)

with torch.amp.autocast('cuda', dtype=torch.float16):
    out2 = model(x)
    print('With autocast: output dtype =', out2.dtype)