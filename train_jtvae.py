import os
import sys
import math
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from model.VAE import JunctionTreeVAE
from lib.data_utils import JunctionTreeFolder
import lib.plot

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', required=True)
parser.add_argument('--hidden_size', type=int, default=450)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--latent_size', type=int, default=56)
parser.add_argument('--depthT', type=int, default=40)
parser.add_argument('--depthG', type=int, default=10)

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--clip_norm', type=float, default=50.0)
parser.add_argument('--beta', type=float, default=0.0)
parser.add_argument('--step_beta', type=float, default=0.002)
parser.add_argument('--max_beta', type=float, default=1.0)
parser.add_argument('--warmup', type=int, default=40000)

parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--anneal_rate', type=float, default=0.9)
parser.add_argument('--anneal_iter', type=int, default=40000)
parser.add_argument('--kl_anneal_iter', type=int, default=2000)
parser.add_argument('--print_iter', type=int, default=50)
parser.add_argument('--save_iter', type=int, default=5000)

args = parser.parse_args()
print(args)

lib.plot.set_output_dir('./')
lib.plot.suppress_stdout()

model = JunctionTreeVAE(args.hidden_size, args.latent_size, args.depthT, args.depthG).to(device)
print(model)
# for param in model.parameters():
#     print(param)
# exit()
for param in model.parameters():
    if param.dim() == 1:
        nn.init.constant_(param, 0)
    else:
        nn.init.xavier_normal_(param)


print("Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,))

optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.ExponentialLR(optimizer, args.anneal_rate)

param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))

total_step = 0
beta = args.beta
meters = np.zeros(4)

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

for epoch in range(args.epoch):
    loader = JunctionTreeFolder('data/rna_jt', args.batch_size, num_workers=8)
    for batch in loader:
        total_step += 1
        model.zero_grad()
        loss, kl_div, all_acc = model(batch, beta)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
        optimizer.step()

        meters = meters + np.array([kl_div, all_acc[0] * 100, all_acc[1] * 100, all_acc[2] * 100])

        if total_step % args.print_iter == 0:
            meters /= args.print_iter
            print("[%d] Beta: %.3f, KL: %.2f, Node: %.2f, Nucleotide: %.2f, Topo: %.2f, PNorm: %.2f, GNorm: %.2f" % (
            total_step, beta, meters[0], meters[1], meters[2], meters[3], param_norm(model), grad_norm(model)))
            lib.plot.plot('node acc', meters[1])
            lib.plot.plot('nucleotide acc', meters[2])
            lib.plot.plot('topo acc', meters[3])
            lib.plot.flush()
            sys.stdout.flush()
            meters *= 0

        lib.plot.tick()


        if total_step % args.save_iter == 0:
            torch.save(model.state_dict(), os.path.join(args.save_dir, "model.iter-" + str(total_step)))

        if total_step % args.anneal_iter == 0:
            scheduler.step()
            print("learning rate: %.6f" % scheduler.get_lr()[0])

        if total_step % args.kl_anneal_iter == 0 and total_step >= args.warmup:
            beta = min(args.max_beta, beta + args.step_beta)