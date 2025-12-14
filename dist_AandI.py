import numpy as np
import matplotlib.pyplot as plt


# ========= 乱数生成：切断正規（A） =========
def sample_truncnorm(mu: float, sigma: float, low: float, high: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """N(mu, sigma^2) を [low, high] に切断して棄却サンプリングで n 個作る"""
    out = np.empty(n, dtype=float)
    filled = 0
    while filled < n:
        # 残り必要数より多めに引く
        m = int((n - filled) * 1.5) + 1000
        x = rng.normal(loc=mu, scale=sigma, size=m)
        x = x[(x >= low) & (x <= high)]
        take = min(x.size, n - filled)
        if take > 0:
            out[filled:filled + take] = x[:take]
            filled += take
    return out


# ========= 乱数生成：切断対数正規（I） =========
def lognormal_params_from_percentiles(q1: float, p1: float, q2: float, p2: float) -> tuple[float, float]:
    """
    X ~ LogNormal(mu, sigma)（= exp(N(mu,sigma^2))）について、
      P(X<=q1)=p1,  P(X<=q2)=p2
    を満たす mu,sigma を返す（mu,sigma は ln(X) の正規分布パラメータ）。
    """
    # 今回は p1=0.01, p2=0.99 の固定なので z 値を定数として使う（標準正規の 1% 点）
    # z0.01 = -2.3263478740408408, z0.99 = +2.3263478740408408
    if not (np.isclose(p1, 0.01) and np.isclose(p2, 0.99)):
        raise ValueError("この関数は現状 p1=0.01, p2=0.99 の前提（必要なら一般化する）")

    z1 = -2.3263478740408408
    z2 = +2.3263478740408408

    lnq1 = np.log(q1)
    lnq2 = np.log(q2)

    sigma = (lnq2 - lnq1) / (z2 - z1)
    mu = lnq1 - sigma * z1
    return mu, sigma


def sample_trunclognorm(mu_ln: float, sigma_ln: float, low: float, high: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """LogNormal(mu_ln, sigma_ln) を [low, high] に切断して棄却サンプリングで n 個作る"""
    out = np.empty(n, dtype=float)
    filled = 0
    while filled < n:
        m = int((n - filled) * 1.5) + 1000
        x = rng.lognormal(mean=mu_ln, sigma=sigma_ln, size=m)
        x = x[(x >= low) & (x <= high)]
        take = min(x.size, n - filled)
        if take > 0:
            out[filled:filled + take] = x[:take]
            filled += take
    return out


# ========= Δf 計算（1 seed） =========
def compute_df(muA: float, seed: int, n: int = 1000_000, sigma_A: float = 0.172) -> dict:
    rng = np.random.default_rng(seed)

    # A: 切断正規（σは固定、μだけシナリオで変える）
    A = sample_truncnorm(mu=muA, sigma=sigma_A, low=0.1, high=0.9, n=n, rng=rng)

    # I: 「非切断の対数正規で 1%点=0.1, 99%点=0.9」を満たす mu,sigma を作ってから、[0.1,0.9]で切断
    mu_lnI, sigma_lnI = lognormal_params_from_percentiles(q1=0.1, p1=0.01, q2=0.9, p2=0.99)
    I = sample_trunclognorm(mu_ln=mu_lnI, sigma_ln=sigma_lnI, low=0.1, high=0.9, n=n, rng=rng)

    # f
    f_return = (1 - A) * (1 - I)
    f_stay = (1 - A) * I

    # 90%幅（5%〜95%）
    q05_r, q95_r = np.quantile(f_return, [0.05, 0.95])
    q05_s, q95_s = np.quantile(f_stay, [0.05, 0.95])

    df = max(q95_r - q05_r, q95_s - q05_s)

    return {
        "muA": muA, "seed": seed, "n": n,
        "A": A, "I": I,
        "f_return": f_return, "f_stay": f_stay,
        "df": df,
        "width_return": (q95_r - q05_r),
        "width_stay": (q95_s - q05_s),
        "mu_lnI": mu_lnI, "sigma_lnI": sigma_lnI
    }


# ========= 分布を描画（1 seed分だけでOK） =========
def plot_distributions(res: dict, bins: int = 60) -> None:
    muA = res["muA"]

    plt.figure()
    plt.hist(res["A"], bins=bins)
    plt.title(f"A (trunc normal)  muA={muA}")
    plt.xlabel("A"); plt.ylabel("count")
    plt.show()

    plt.figure()
    plt.hist(res["I"], bins=bins)
    plt.title(f"I (trunc lognormal)  muA={muA}")
    plt.xlabel("I"); plt.ylabel("count")
    plt.show()

    plt.figure()
    plt.hist(res["f_return"], bins=bins)
    plt.title(f"f_return=(1-A)(1-I)  muA={muA}")
    plt.xlabel("f_return"); plt.ylabel("count")
    plt.show()

    plt.figure()
    plt.hist(res["f_stay"], bins=bins)
    plt.title(f"f_stay=(1-A)I  muA={muA}")
    plt.xlabel("f_stay"); plt.ylabel("count")
    plt.show()


# ========= seedを複数回回して Δf の平均±SD =========
def df_stats(muA: float, seeds: list[int], n: int = 1000_000, sigma_A: float = 0.172) -> dict:
    dfs = []
    widths_r = []
    widths_s = []
    for sd in seeds:
        r = compute_df(muA=muA, seed=sd, n=n, sigma_A=sigma_A)
        dfs.append(r["df"])
        widths_r.append(r["width_return"])
        widths_s.append(r["width_stay"])

    dfs = np.array(dfs)
    widths_r = np.array(widths_r)
    widths_s = np.array(widths_s)

    return {
        "muA": muA,
        "df_mean": float(dfs.mean()),
        "df_sd": float(dfs.std(ddof=1)),
        "df_min": float(dfs.min()),
        "df_max": float(dfs.max()),
        "width_return_mean": float(widths_r.mean()),
        "width_stay_mean": float(widths_s.mean()),
    }


# ========= 出力 =========
if __name__ == "__main__":
    n = 1000_000

    # まずは各 muA について seed=0 の分布を描画
    for muA in [0.3, 0.5, 0.7]:
        res0 = compute_df(muA=muA, seed=0, n=n)
        plot_distributions(res0)

        print(f"[muA={muA}] seed=0  Δf={res0['df']:.6f} | "
              f"width_return={res0['width_return']:.6f} | width_stay={res0['width_stay']:.6f}")

    # 次に seed を複数回回して Δf のばらつき（平均±SD）
    seeds = list(range(30))  # 0..29
    print("\n=== Δf stats over multiple seeds ===")
    for muA in [0.3, 0.5, 0.7]:
        st = df_stats(muA=muA, seeds=seeds, n=n)
        print(f"[muA={muA}] Δf = {st['df_mean']:.6f} ± {st['df_sd']:.6f}  "
              f"(min={st['df_min']:.6f}, max={st['df_max']:.6f})  "
              f"| mean width_return={st['width_return_mean']:.6f}, mean width_stay={st['width_stay_mean']:.6f}")
        
     # やっぱり2つのfの平均値にしよう！
    seeds = list(range(30))  # 0..29
    print("\n===average===")
    for muA in [0.3, 0.5, 0.7]:
        st = df_stats(muA=muA, seeds=seeds, n=n)
        f_return=st['width_return_mean']
        f_stay=st['width_stay_mean']
        ave_f=(f_return + f_stay ) /2
        
        print(f"[muA={muA}] ave_f={ave_f}")
     


