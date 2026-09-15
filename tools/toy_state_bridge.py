"""StateBridge toy verification (2-level, tiny dims).

Q1: can we inject custom initial_states into Hydra's scan (both directions)?
Q2: does the kernel return final_states (return_final_states=True)?
Q3: do gradients flow back through initial_states and final_states?
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "hydra"))

from hydra.modules.hydra import Hydra


def main():
    torch.manual_seed(0)
    D, L, B = 32, 24, 2
    hyd = Hydra(d_model=D, d_state=8, d_conv=3, expand=2, headdim=16,
                use_mem_eff_path=False).cuda()
    nheads = hyd.nheads
    print("nheads:", nheads, "dstate:", hyd.d_state, "headdim:", hyd.headdim)

    u = torch.randn(B, L, D, device="cuda", requires_grad=True)

    # --- Q2/Q3: extract final states via the raw kernel path ---
    # Replicate hydra.forward internals on a small case to test the kernel.
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
    from einops import rearrange, repeat

    def run_kernel(x_in, init_states=None, want_final=False):
        zxbcdt = hyd.in_proj(x_in)
        A = -torch.exp(hyd.A_log.float())
        z, xBC, dt = torch.split(
            zxbcdt,
            [hyd.d_inner,
             hyd.d_inner + 2 * (2 * hyd.ngroups * hyd.d_state),
             2 * hyd.nheads],
            dim=-1,
        )
        dt = torch.cat((dt[:, :, :hyd.nheads],
                        torch.flip(dt[:, :, hyd.nheads:], (1,))), dim=0)
        dt = torch.nn.functional.softplus(dt + hyd.dt_bias)
        xBC = hyd.act(hyd.conv1d(xBC.transpose(1, 2)).transpose(1, 2))
        x, BC = torch.split(
            xBC, [hyd.d_inner, 2 * (2 * hyd.ngroups * hyd.d_state)], dim=-1)
        x = torch.cat((x, torch.flip(x, (1,))), dim=0)
        BC = torch.cat((BC[:, :, :2 * hyd.ngroups * hyd.d_state],
                        torch.flip(BC[:, :, 2 * hyd.ngroups * hyd.d_state:],
                                   (1,))), dim=0)
        Bm, Cm = torch.split(
            BC, [hyd.ngroups * hyd.d_state, hyd.ngroups * hyd.d_state],
            dim=-1)
        out = mamba_chunk_scan_combined(
            rearrange(x, "b l (h p) -> b l h p", p=hyd.headdim),
            dt, A,
            rearrange(Bm, "b l (g n) -> b l g n", g=hyd.ngroups),
            rearrange(Cm, "b l (g n) -> b l g n", g=hyd.ngroups),
            chunk_size=hyd.chunk_size, D=None, z=None,
            dt_bias=hyd.dt_bias, dt_softplus=False,
            return_final_states=want_final,
            initial_states=init_states,
        )
        return out

    # final states extraction
    y, final_states = run_kernel(u, want_final=True)
    print("Q2 final_states shape:", tuple(final_states.shape),
          "finite:", bool(torch.isfinite(final_states).all()))

    # --- Q1/Q3: inject custom initial states, gradient check ---
    init = torch.zeros(2 * B, nheads, hyd.headdim, hyd.d_state,
                       device="cuda", requires_grad=True)
    init = init + 0.05 * torch.randn_like(init)
    y2, fs2 = run_kernel(u, init_states=init, want_final=True)
    loss = y2.sum() + fs2.sum()
    loss.backward()
    print("Q1 injected-run ok; finite:", bool(torch.isfinite(y2).all()))
    print("Q3 grad wrt initial_states:",
          "None" if init.grad is None else float(init.grad.abs().sum()),
          "| grad wrt input:", None if u.grad is None
          else float(u.grad.abs().sum()))
    diff = (y2.sum() - run_kernel(u)[0].sum().detach()).abs()
    print("injection changes output (should be >0):", float(diff))


if __name__ == "__main__":
    main()
