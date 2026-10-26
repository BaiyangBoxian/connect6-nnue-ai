#!/usr/bin/env python3
# connect6-nnue-ai
# Neural network + alpha-beta search board game AI, trained with self-play
#
# BetaSix: 面向十九路六子棋的多任务 NNUE 高效评估与 α-β 搜索
# Copyright (C) 2026  白杨博贤
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# Copyright (c) 2026 白杨博贤
# train_multitask.py -- KataGo-style multi-task NNUE for Connect6 (PyTorch CPU).
# Shared trunk + heads: value / WDL / TTE / ownership / policy.  Auxiliary tasks
# (esp. policy + ownership) stop the net from "shortcutting" a single scalar and
# force rich board understanding (KataGo's key idea).
import numpy as np, os, struct, sys, torch, torch.nn as nn, torch.nn.functional as F

DATA = sys.argv[1] if len(sys.argv) > 1 else 'engine_data/mt_train.bin'
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 40
CAP = int(sys.argv[3]) if len(sys.argv) > 3 else 60000   # max training samples (merged sets are large)
H1, H2, H3 = 512, 256, 128   # Phase 2: wider 722->512->256->128
VSCALE = 200000.0
torch.manual_seed(7); torch.set_num_threads(2)

def fake_quant(w, bits=8):
    # QAT: simulate int8 quantization (Stockfish-style) so inference can use int8
    qmin, qmax = -2**(bits-1), 2**(bits-1)-1
    scale = w.abs().max() / qmax if w.abs().max() > 0 else 1.0
    q = torch.clamp((w / scale).round(), qmin, qmax) * scale
    return q

class MultiTaskNNUE(nn.Module):
    def __init__(s):
        super().__init__()
        s.trunk = nn.Sequential(nn.Linear(736,H1), nn.ReLU(), nn.Dropout(0.2), nn.Linear(H1,H2), nn.ReLU(), nn.Dropout(0.2), nn.Linear(H2,H3), nn.ReLU(), nn.Dropout(0.2))   # P0.4 dropout
        s.value = nn.Linear(H3,1)
        s.wdl   = nn.Linear(H3,3)
        s.tte   = nn.Linear(H3,1)
        s.owner = nn.Linear(H3,361*3)   # 3 classes per cell
        s.policy= nn.Linear(H3,361)
    def forward(s, x):
        # QAT: fake-quantize the linear weights (int8) during training
        # P0.1: QAT OFF
        h = s.trunk(x)
        return (torch.tanh(s.value(h)), s.wdl(h), s.tte(h).squeeze(1),
                s.owner(h).view(-1,361,3), s.policy(h))

def load(path, cap=60000):
    # OOM-safe loader (fix 2026-10-20): never materialize the whole merged file
    # (870MB+ np.fromfile got rc=137 OOM-killed at 23:31). memmap + random subset.
    rec = 361+4+1+4+1+1+1+56+361   # +mover +14 feats(56)
    with open(path,'rb') as f:
        n = struct.unpack('<i', f.read(4))[0]
    avail = (os.path.getsize(path) - 4) // rec
    if n <= 0 or avail <= 0:
        raw = np.empty((0, rec), dtype=np.uint8); n = 0
    else:
        n = min(n, avail)
        if n > cap:
            idx = np.sort(np.random.default_rng(1).choice(n, cap, replace=False)); n = cap
        else:
            idx = np.arange(n)
        mm = np.memmap(path, dtype=np.uint8, mode='r', offset=4)
        raw = np.ascontiguousarray(mm[:avail*rec].reshape(avail, rec)[idx])
        del mm
    board = raw[:, :361].astype(np.int8)
    off = 361
    value = raw[:, off:off+4].copy().view(np.float32)[:,0]; off += 4
    wdl   = raw[:, off].astype(np.int64); off += 1
    tte   = raw[:, off:off+4].copy().view(np.float32)[:,0]; off += 4
    polr  = raw[:, off].astype(np.int64); off += 1
    polc  = raw[:, off].astype(np.int64); off += 1
    mover = raw[:, off].astype(np.int64); off += 1   # side to move (0=black 1=white)
    feat = raw[:, off:off+56].copy().view('<f4').reshape(-1,14); off += 56   # 14 features
    owner = raw[:, off:off+361].astype(np.int64)
    return board, value, wdl, tte, polr, polc, mover, feat, owner

def make_xy(board, value, wdl, tte, polr, polc, mover, feat, owner):
    # self = MOVER (side to move).  WDL is already the mover's W/D/L (Step 1.1).
    self_is_black = (mover == 0)
    selfp = np.where(self_is_black[:,None], (board==1), (board==2))
    oppp  = np.where(self_is_black[:,None], (board==2), (board==1))
    X  = np.concatenate([selfp, oppp, feat.astype(np.float32)],1).astype(np.float32)   # 722+12=734
    # value: black's view -> convert to mover's view (flip for white mover)
    v  = np.tanh(value / VSCALE).astype(np.float32)
    v  = np.where(self_is_black, v, -v).astype(np.float32)
    v  = v - v.mean()   # P0.5: center the value labels (remove the first-player bias)
    W  = wdl.astype(np.int64)                 # mover's W/D/L (2/1/0)
    T  = tte.astype(np.float32)
    P  = (polr*19 + polc).astype(np.int64)    # policy: move cell
    # ownership from the MOVER's view (swap 1<->2 for white mover)
    o0 = owner.astype(np.int64)
    O  = np.where(self_is_black[:,None], o0, np.where(o0==1,2,np.where(o0==2,1,0))).astype(np.int64)
    return X, v, W, T, P, O

def main():
    board, value, wdl, tte, polr, polc, mover, feat, owner = load(DATA, CAP)
    if len(value) == 0:
        # live .bin files being written by generators have count=0 in header;
        # merge yields 0 usable samples. Skip cleanly instead of crashing.
        print('skip: 0 usable samples (generator files still open, will train when they close)')
        return
    # anti-shortcut check: degenerate labels (e.g. WDL all-draw) collapse the head
    # to a constant predictor with loss 0. Surface label distributions every run.
    import numpy as _np
    print('label check: wdl[0/1/2]=%s value[min/max/mean]=%.0f/%.0f/%.0f tte[min/max]=%.0f/%.0f' % (
        _np.bincount(wdl.astype(int), minlength=3).tolist(),
        value.min(), value.max(), value.mean(), tte.min(), tte.max()))
    X, v, W, T, P, O = make_xy(board, value, wdl, tte, polr, polc, mover, feat, owner)
    N = len(X); print(f"{N} samples")
    Xt=torch.tensor(X); vt=torch.tensor(v); Wt=torch.tensor(W); Tt=torch.tensor(T/70.0)
    Pt=torch.tensor(P); Ot=torch.tensor(O)
    # P0.2: 80/20 train/val split
    n=len(X); ntr=int(n*0.8); idx=np.random.permutation(n); tri, vai = idx[:ntr], idx[ntr:]
    ds=torch.utils.data.TensorDataset(Xt[tri],vt[tri],Wt[tri],Tt[tri],Pt[tri],Ot[tri])
    dl=torch.utils.data.DataLoader(ds,batch_size=2048,shuffle=True)
    Xva,vva,Wva,Tva,Pva,Ova = Xt[vai],vt[vai],Wt[vai],Tt[vai],Pt[vai],Ot[vai]
    vhist=[]
    model=MultiTaskNNUE(); opt=torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)   # P0.4
    for ep in range(EPOCHS):
        lv=lw=lt=lp=lo=0.0
        for xb,vb,Wb,Tb,Pb,Ob in dl:
            opt.zero_grad()
            pv,pw,pt,po,pp = model(xb)
            loss_v  = F.mse_loss(pv.squeeze(1), vb)
            # Step 1.2: WDL loss = KL divergence (target one-hot || predicted softmax)
            loss_w  = F.kl_div(F.log_softmax(pw, dim=1), F.one_hot(Wb, 3).float(), reduction='batchmean')
            loss_t  = F.mse_loss(pt, Tb)
            # ownership: empty-cell weight 0.1, occupied 1.0 (fix the stuck 0.826 imbalance)
            _om = (Ob.view(-1) != 0).float() * 0.9 + 0.1   # 1.0 for occupied, 0.1 for empty
            loss_o  = (F.cross_entropy(po.view(-1,3), Ob.view(-1), reduction='none') * _om).mean()
            loss_p  = F.cross_entropy(pp, Pb)
            loss = loss_v + 0.5*loss_w + 0.3*loss_t + 0.3*loss_o + 0.7*loss_p
            loss.backward(); opt.step()
            lv+=loss_v.item()*len(xb); lw+=loss_w.item()*len(xb); lt+=loss_t.item()*len(xb)
            lo+=loss_o.item()*len(xb); lp+=loss_p.item()*len(xb)
        # P0.2: validation loss (5-task sum)
        model.eval()
        with torch.no_grad():
            pv,pw,pt,po,pp = model(Xva)
            val = (F.mse_loss(pv.squeeze(1),vva)
                   + 0.5*F.kl_div(F.log_softmax(pw,1),F.one_hot(Wva,3).float(),reduction='batchmean')
                   + 0.3*F.mse_loss(pt,Tva) + 0.3*F.cross_entropy(po.view(-1,3),Ova.view(-1))
                   + 0.7*F.cross_entropy(pp,Pva)).item()
        model.train()
        vhist.append(val)
        # P0.3: early stop if val rose 3 consecutive rounds
        if len(vhist)>=4 and all(vhist[-i]>vhist[-i-1] for i in range(1,4)):
            print(f"early stop at ep{ep+1} (val rose 3 rounds: {vhist[-4:]})")
            break
        if (ep+1)%5==0 or ep==0:
            # LOG (user format): 时间戳,轮次,value_loss,policy_loss,wdl_loss,ownership_loss,tte_loss,样本数
            import datetime as _dt
            with open('train_log.txt','a') as _lf:
                _lf.write(f"{_dt.datetime.now().strftime('%F %T')},{ep+1},{lv/N:.4f},{lp/N:.4f},{lw/N:.4f},{lo/N:.4f},{lt/N:.4f},{val:.4f},{len(X)}\n")
            print(f"ep{ep+1:3d} value={lv/N:.4f} wdl={lw/N:.4f} tte={lt/N:.4f} own={lo/N:.4f} pol={lp/N:.4f} | VAL={val:.4f}")
    # QAT: also save int8-quantized weights for the int8 inference path
    def q8(w):
        qmin,qmax=-128,127; m=float(np.abs(w).max()); sc=(m/qmax if m>0 else 1.0)
        return (np.clip((w/sc).round(),qmin,qmax)).astype(np.int8), sc
    qW1,s1=q8(model.trunk[0].weight.detach().numpy().T); qW2,s2=q8(model.trunk[3].weight.detach().numpy().T); qW3,s3=q8(model.value.weight.detach().numpy().T)
    # WIDE-experiment outputs go to *_wide.* -- NEVER touch the live narrow
    # engine_data/mt_weights.npz / nnue_A.h (engine NN_H1=256/NN_H2=64 is narrow;
    # installing wide weights would silently misread them).
    np.savez('engine_data/mt_weights_int8_wide.npz', qW1=qW1,s1=s1,qW2=qW2,s2=s2,qW3=qW3,s3=s3)
    np.savez('engine_data/mt_weights_wide.npz',
             W1=model.trunk[0].weight.detach().numpy().T, b1=model.trunk[0].bias.detach().numpy(),
             W2=model.trunk[3].weight.detach().numpy().T, b2=model.trunk[3].bias.detach().numpy(),
             W3=model.trunk[6].weight.detach().numpy().T, b3=model.trunk[6].bias.detach().numpy(),
             W4=model.value.weight.detach().numpy().T, b4=model.value.bias.detach().numpy(),
             H1=np.int32(H1),H2=np.int32(H2),H3=np.int32(H3),VSCALE=np.float32(VSCALE))
    print("saved engine_data/mt_weights_wide.npz (value head for engine)")

if __name__ == '__main__':
    main()
