from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Tuple, Optional, List, Any

import numpy as np
import matplotlib.pyplot as plt


# ==========================================
# 1. 列挙型（状態・属性）と設定値（Config）
# ==========================================
class HousingState(Enum):
    """居住状態 H ∈ {PI, PO, TI, TO}"""
    PI = 0  # Permanent Inside: 自治体内で定住
    PO = 1  # Permanent Outside: 自治体外で定住（＝流出先に定着）
    TI = 2  # Temporary Inside: 自治体内で仮住まい
    TO = 3  # Temporary Outside: 自治体外で仮住まい


class HouseholdType(Enum):
    """世帯タイプ L ∈ {YC, YN, EW, ER}"""
    YC = 0  # 若年・子供あり
    YN = 1  # 若年・子供なし
    EW = 2  # 高齢・現役
    ER = 3  # 高齢・引退


class Config:
    """
    シミュレーション設定。

    時間の定義：
      - t は「月」を表す。
      - t=0 が初期状態であり、step() は t→t+1 の1か月更新を表す。

    評価スナップショット（T1/T2/TAU）の定義：
      - 「改定直前」を評価したいので、スナップショットは step() の最後（t を +1 した後）に保存する。
        例：T1 の“改定直前状態”は step(t=T1-1) 終了後の状態であり、その時点のログ時刻は self.t == T1 となる。
    """

    # --- 時間設定 ---
    T_MAX = 120          # シミュレーション期間（月）
    T1 = 36              # 計画改定1（この月の“直前”状態を評価対象にする）
    T2 = 72              # 計画改定2（この月の“直前”状態を評価対象にする）
    TAU = T_MAX          # 評価時点 τ（デフォルトは期末）

    # --- 世帯数と初期分布 ---
    N_HOUSEHOLDS = 5000
    INIT_COUNTS_H = {
        HousingState.PI: 1400,
        HousingState.TI: 2880,
        HousingState.TO: 720,
        HousingState.PO: 0,
    }
    INIT_COUNTS_L = {
        HouseholdType.YC: 1500,
        HouseholdType.YN: 1300,
        HouseholdType.EW: 450,
        HouseholdType.ER: 1750,
    }

    # --- 都市環境（3次元ベクトル：例として E/A/S を想定） ---
    C0 = np.array([0.25, 0.25, 0.25], dtype=float)
    V_POL = 1.1

    # --- 計画改定（方向ベクトル混合） ---
    ALPHA1 = 0.8
    ALPHA2 = 0.8
    LAMBDA = 0.3
    DELAY_WINDOW = 6

    # --- 居住地選択（閾値モデル） ---
    THETA_1 = 0.7
    THETA_2 = 0.2
    THETA_3 = 1.0
    THETA_4 = 0.45
    THETA_5 = 0.9
    BETA = 1.5
    GAMMA = 0.0

    # --- A/I の生成（母集団プールの生成で使用） ---
    LOVE_MU = 0.5
    LOVE_SIGMA = 0.172
    LOVE_LOW = 0.1
    LOVE_HIGH = 0.9

    INC_LOGMU = -0.8
    INC_LOGSIGMA = 0.5
    INC_LOW = 0.1
    INC_HIGH = 0.9


# ==========================================
# 2. 補助関数（数値処理・乱数生成）
# ==========================================
def normalize(v: np.ndarray) -> np.ndarray:
    """
    ベクトルの正規化（ユークリッドノルム）。

    - v がゼロベクトルの場合、0除算を避けるため v をそのまま返す。
    """
    n = float(np.linalg.norm(v))
    return v if n == 0.0 else (v / n)


def gap_nonneg(V: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    計画ギャップ P = V - C を計算し、負の成分は0に丸める（仕様B）。

    解釈：
      復興で「悪化方向（負のギャップ）」の計画は採用しない、という仮定の下で、
      P_j = max(0, V_j - C_j) とする。
    """
    return np.maximum(0.0, V - C)


def make_truncated_normal_pool(n: int, mu: float, sigma: float, lower: float, upper: float) -> List[float]:
    """
    N(mu, sigma^2) から [lower, upper] の範囲に入る値のみを採用して n 個集め、
    「母集団（固定プール）」として返す。

    目的：
      - Monte Carlo で複数runを比較するときに「母集団（値の集合）そのもの」を完全に同一化するため。
    """
    pool: List[float] = []
    while len(pool) < n:
        x = random.gauss(mu, sigma)
        if lower <= x <= upper:
            pool.append(x)
    return pool


def make_truncated_lognormal_pool(n: int, mu: float, sigma: float, lower: float, upper: float) -> List[float]:
    """
    logN(mu, sigma^2) から [lower, upper] の範囲に入る値のみを採用して n 個集め、
    「母集団（固定プール）」として返す。
    """
    pool: List[float] = []
    while len(pool) < n:
        x = random.lognormvariate(mu, sigma)
        if lower <= x <= upper:
            pool.append(x)
    return pool


def get_weights(h_state: HousingState, l_type: HouseholdType) -> np.ndarray:
    """
    選好ウェイト w_i（E/A/Sの重み）を、居住状態×世帯タイプから与える。

    - ここでは表（表-12相当）で与えられる離散値をハードコードする。
    - 最終的に sum(w_i)=1 となるよう正規化して返す。
    """
    w = np.array([0.333, 0.333, 0.333], dtype=float)

    if h_state == HousingState.PI:
        if l_type == HouseholdType.YC: w = np.array([0.4, 0.4, 0.2])
        elif l_type == HouseholdType.YN: w = np.array([0.45, 0.325, 0.225])
        elif l_type == HouseholdType.EW: w = np.array([0.367, 0.367, 0.267])
        elif l_type == HouseholdType.ER: w = np.array([0.3, 0.4, 0.3])
    elif h_state == HousingState.TI:
        if l_type == HouseholdType.YC: w = np.array([0.325, 0.325, 0.35])
        elif l_type == HouseholdType.YN: w = np.array([0.375, 0.25, 0.375])
        elif l_type == HouseholdType.EW: w = np.array([0.292, 0.292, 0.417])
        elif l_type == HouseholdType.ER: w = np.array([0.225, 0.325, 0.45])
    elif h_state == HousingState.TO:
        if l_type == HouseholdType.YC: w = np.array([0.3, 0.4, 0.3])
        elif l_type == HouseholdType.YN: w = np.array([0.35, 0.325, 0.325])
        elif l_type == HouseholdType.EW: w = np.array([0.267, 0.367, 0.367])
        elif l_type == HouseholdType.ER: w = np.array([0.2, 0.4, 0.4])

    return w / float(np.sum(w))


def summarize_numeric(x: List[float]) -> Dict[str, float]:
    """
    連続変数（A_i, I_i など）の要約統計を返す。

    返す指標：
      n, mean, std, min, q25, median, q75, max
    """
    if len(x) == 0:
        return {"n": 0, "mean": np.nan, "std": np.nan, "min": np.nan,
                "q25": np.nan, "median": np.nan, "q75": np.nan, "max": np.nan}

    arr = np.array(x, dtype=float)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "q25": float(np.quantile(arr, 0.25)),
        "median": float(np.quantile(arr, 0.50)),
        "q75": float(np.quantile(arr, 0.75)),
        "max": float(arr.max()),
    }


# ==========================================
# 3. エージェント定義
# ==========================================
@dataclass
class Agent:
    """
    住民（世帯）エージェント。

    主要変数：
      - state: 現在の居住状態 H_i(t)
      - initial_state: t=0 時点の居住状態（同調圧力の集計キー）
      - l_type: 世帯タイプ L_i
      - love: 地元愛 A_i
      - income: 所得 I_i
      - temp_months: 仮住まい期間（月数）
      - first_decision: 「t=0 の直後に初めて下した決定」の分類（同調圧力用）
    """
    id: int
    state: HousingState
    initial_state: HousingState
    l_type: HouseholdType
    love: float
    income: float
    temp_months: int = 0
    first_decision: Optional[str] = None

    @property
    def weights(self) -> np.ndarray:
        """現在の居住状態と世帯タイプに対応する w_i を返す。"""
        return get_weights(self.state, self.l_type)


# ==========================================
# 4. シミュレーション本体
# ==========================================
class Simulation:
    def __init__(
        self,
        base_A_pool: List[float],
        base_I_pool: List[float],
        shuffle_assignment_each_run: bool = True,
    ):
        """
        Parameters
        ----------
        base_A_pool, base_I_pool:
            A_i, I_i の「母集団（値の集合）」。
            複数runで比較したい場合は、外側で1回だけ生成し、全runへ同じものを渡す。
        shuffle_assignment_each_run:
            True の場合、各runの initialize() でプールの順序だけシャッフルして割当を変える。
            これにより「母集団は同一だが、誰がどの値を持つか」は run ごとにランダムになる。
        """
        if len(base_A_pool) != Config.N_HOUSEHOLDS or len(base_I_pool) != Config.N_HOUSEHOLDS:
            raise ValueError("base_A_pool / base_I_pool must have length Config.N_HOUSEHOLDS")

        self.base_A_pool = base_A_pool
        self.base_I_pool = base_I_pool
        self.shuffle_assignment_each_run = shuffle_assignment_each_run

        self.t = 0
        self.agents: List[Agent] = []

        # 都市環境ベクトル（実現値）: C(t) ∈ R^3
        self.C = Config.C0.copy()

        # 計画の進捗速度 K と有効計画方向ベクトル P_eff
        self.K: float = 0.0
        self.P_eff: np.ndarray = np.zeros(3, dtype=float)

        # 将来像 C_hat（区間内で固定）
        self.C_hat_interval: np.ndarray = Config.C0.copy()

        # 同調圧力用：初期属性（L_i, H_i(0)）ごとの人数と first_decision カウント
        self.group_totals: Dict[Tuple[HouseholdType, HousingState], int] = {}
        self.group_counts: Dict[Tuple[HouseholdType, HousingState], Dict[str, int]] = {}

        # 時系列ログ
        self.history_t: List[int] = []
        self.history_nPI: List[int] = []
        self.history_nPO: List[int] = []
        self.history_nTemp: List[int] = []  # TI+TO

        # 評価スナップショット（t=T1, T2, TAU）
        self.snapshots: Dict[int, Dict[str, Any]] = {}

    def initialize(self):
        """
        初期化処理。

        1) 初期分布に従い、居住状態 H と世帯タイプ L をランダムに割当てる
        2) A_i, I_i は「固定母集団プール」から重複なしで割当てる（runごとに対応だけシャッフル可能）
        3) t=0 時点の計画（P_eff, K, C_hat）を確定する
        4) 初期状態（t=0）を1回だけログに記録する（重複記録はしない）
        """
        # --- runごとの割当：母集団は固定、対応だけシャッフル ---
        A_pool = list(self.base_A_pool)
        I_pool = list(self.base_I_pool)
        if self.shuffle_assignment_each_run:
            random.shuffle(A_pool)
            random.shuffle(I_pool)

        # --- 初期状態プールの作成（「くじ引き」方式） ---
        h_pool: List[HousingState] = []
        for s, c in Config.INIT_COUNTS_H.items():
            h_pool.extend([s] * c)
        random.shuffle(h_pool)

        l_pool: List[HouseholdType] = []
        for lt, c in Config.INIT_COUNTS_L.items():
            l_pool.extend([lt] * c)
        random.shuffle(l_pool)

        # --- エージェント生成（A/Iはプールから重複なし割当） ---
        for i in range(Config.N_HOUSEHOLDS):
            love = A_pool[i]
            inc = I_pool[i]
            hs = h_pool[i] if i < len(h_pool) else HousingState.TI
            lt = l_pool[i] if i < len(l_pool) else HouseholdType.ER
            self.agents.append(Agent(i, hs, hs, lt, love, inc))

        # --- 同調圧力用のグループ総数（固定） ---
        for ag in self.agents:
            key = (ag.l_type, ag.initial_state)
            self.group_totals[key] = self.group_totals.get(key, 0) + 1
        for key in self.group_totals:
            self.group_counts[key] = {"return": 0, "exit": 0, "stay": 0}

        # --- t=0 の計画採択（初回は混合なし） ---
        self._adopt_new_plan(alpha=None, t_adopt=0)

        # --- 初期ログ（t=0） ---
        self._record()

    def _get_w_bar(self) -> np.ndarray:
        """
        住民意向の代表値 w̄ を計算する。

        仕様：
          - PO は計画形成の対象外とみなし、PI/TI/TO の w_i を平均する。
        """
        s = np.zeros(3, dtype=float)
        n = 0
        for ag in self.agents:
            if ag.state != HousingState.PO:
                s += ag.weights
                n += 1
        return s / n if n > 0 else np.array([1/3, 1/3, 1/3], dtype=float)

    def _determine_V(self, w_bar: np.ndarray) -> np.ndarray:
        """
        構想ベクトル V を決定する（最低基準つきの配分ルール）。

        Step 1: 3*V_POL を w̄ の比率で配分
        Step 2: V_j < 1 の成分があれば 1 に切上げ（最低基準）
        Step 3: 余剰を残り成分へ w̄ 比で再配分
        """
        V_pol = float(Config.V_POL)
        w_sum = float(np.sum(w_bar))
        V = 3.0 * V_pol * w_bar / w_sum

        fixed: List[int] = []
        for j in range(3):
            if V[j] < 1.0:
                V[j] = 1.0
                fixed.append(j)

        remain = [j for j in range(3) if j not in fixed]
        if remain:
            surplus = 3.0 * V_pol - float(np.sum(V[fixed])) if fixed else 3.0 * V_pol
            w_rem = float(np.sum(w_bar[remain]))
            if w_rem > 0:
                for j in remain:
                    V[j] = surplus * (w_bar[j] / w_rem)
        return V

    def _adopt_new_plan(self, alpha: Optional[float], t_adopt: int):
        """
        計画方向 P_eff と将来像 C_hat_interval を更新する。

        - ギャップは P = max(0, V - C) とする（負成分は0丸め）
        - 初回（alpha=None）:
            P_eff = P
            K = ||P_eff|| / T_MAX
        - 改定時（alphaが数値）:
            P_eff = alpha * dir(P_new) + (1-alpha) * dir(P_prev)
            （dir(·)は正規化ベクトル）
        - C_hat は改定時点の C を起点に、残り期間 rem=T_MAX-t_adopt だけ
          C_hat = C(t_adopt) + rem * K * dir(P_eff) として固定する。
        """
        w_bar = self._get_w_bar()
        V = self._determine_V(w_bar)
        P = gap_nonneg(V, self.C)

        if alpha is None:
            self.P_eff = P.copy()
            nP = float(np.linalg.norm(self.P_eff))
            self.K = (nP / Config.T_MAX) if nP > 0 else 0.0
        else:
            prev_dir = normalize(self.P_eff)
            new_dir = normalize(P)
            self.P_eff = alpha * new_dir + (1.0 - alpha) * prev_dir

        rem = Config.T_MAX - t_adopt
        self.C_hat_interval = (
            self.C + rem * self.K * normalize(self.P_eff)
            if float(np.linalg.norm(self.P_eff)) > 0
            else self.C.copy()
        )

    def _is_delayed(self, t: int) -> bool:
        """
        進捗低減（λ）の適用判定。

        仕様（ユーザ指定）：
          - 改定作業が存在する（ALPHA>0）場合のみ、
            改定直前 DELAY_WINDOW か月の進捗を低減する。
          - 改定作業がない場合（ALPHA=0）は低減しない。
        """
        w = Config.DELAY_WINDOW
        if Config.ALPHA1 > 0 and (Config.T1 - w <= t < Config.T1):
            return True
        if Config.ALPHA2 > 0 and (Config.T2 - w <= t < Config.T2):
            return True
        return False

    def _get_B(self, ag: Agent, mode: str) -> float:
        """
        同調圧力項 B_i を返す。

        仕様（ユーザ指定）：
          - B_i の集計は「最初の決定（first_decision）」のみを用いる。
          - TI/TO グループについて exit と比較する相手は stay（仮住まい継続）のみ。
            （not_exit = return + stay ではない）

        定義：
          - mode=="return": (stay - return) / n
          - mode in {"exit","stay"}: (exit - stay) / n
        """
        key = (ag.l_type, ag.initial_state)
        total = self.group_totals.get(key, 0)
        if total == 0:
            return 0.0
        cnt = self.group_counts[key]
        if mode == "return":
            return (cnt["stay"] - cnt["return"]) / total
        if mode in ("exit", "stay"):
            return (cnt["exit"] - cnt["stay"]) / total
        return 0.0

    def _log_first_decision(self, ag: Agent, started_state: HousingState):
        """
        各エージェントの「最初の決定」を一度だけ記録する。

        分類（ユーザ指定）：
          - started_state ∈ {TI, TO} の場合：
              PI へ移行 → "return"
              PO へ移行 → "exit"
              それ以外（仮住まい継続） → "stay"
          - started_state == PI の場合：
              その時点で PO へ移行しない限り "stay"
        """
        if ag.first_decision is not None:
            return

        if ag.state == HousingState.PO:
            ag.first_decision = "exit"
            return

        if started_state in (HousingState.TI, HousingState.TO):
            ag.first_decision = "return" if ag.state == HousingState.PI else "stay"
            return

        if started_state == HousingState.PI:
            ag.first_decision = "stay"
            return

        ag.first_decision = "stay"

    def _aggregate_first_decisions(self):
        """first_decision をグループ別に集計する（毎月の状態からは再集計しない）。"""
        for key in self.group_counts:
            self.group_counts[key] = {"return": 0, "exit": 0, "stay": 0}

        for ag in self.agents:
            if ag.first_decision is None:
                continue
            key = (ag.l_type, ag.initial_state)
            if key in self.group_counts:
                self.group_counts[key][ag.first_decision] += 1

    # -------- 評価スナップショット --------
    def _take_snapshot(self) -> Dict[str, Any]:
        """
        現時点（self.t）のスナップショットを作成する。

        取得内容：
          - 都市環境 C（3成分）
          - H={PI,PO,TI,TO} ごとの:
              - 人数 n
              - L（YC/YN/EW/ER）のカウント
              - love(A_i), income(I_i) の要約統計
              - 箱ひげ図用の生データ（A_values, I_values）
        """
        snap: Dict[str, Any] = {"t": int(self.t), "C": self.C.copy()}
        by_state: Dict[HousingState, Dict[str, Any]] = {}

        for hs in HousingState:
            members = [a for a in self.agents if a.state == hs]

            L_counts = {lt: 0 for lt in HouseholdType}
            for a in members:
                L_counts[a.l_type] += 1

            A_list = [a.love for a in members]
            I_list = [a.income for a in members]

            by_state[hs] = {
                "n": len(members),
                "L_counts": L_counts,
                "A_stats": summarize_numeric(A_list),
                "I_stats": summarize_numeric(I_list),
                "A_values": A_list,
                "I_values": I_list,
            }

        snap["by_state"] = by_state
        return snap

    def _maybe_store_snapshot(self):
        """
        評価対象時刻（T1, T2, TAU）ならスナップショットを保存する。

        「改定直前」の扱い：
          - step(t=T1) の冒頭で改定を採択するため、
            評価は step(t=T1-1) 終了後（ログ時刻 self.t==T1）で取得する。
          - 同様に、t=T2 も step(t=T2-1) 終了後の状態が「改定直前」に対応する。
          - τ=TAU は、最終更新後（self.t==T_MAX）に対応する。
        """
        target_times = {Config.T1, Config.T2, Config.TAU}
        if self.t in target_times and self.t not in self.snapshots:
            self.snapshots[self.t] = self._take_snapshot()

    def _record(self):
        """
        時系列ログを1時点分記録する。

        - history_* を更新する。
        - 必要に応じてスナップショットも保存する（T1/T2/TAU）。
        """
        c_PI = sum(1 for a in self.agents if a.state == HousingState.PI)
        c_PO = sum(1 for a in self.agents if a.state == HousingState.PO)
        c_T = sum(1 for a in self.agents if a.state in (HousingState.TI, HousingState.TO))

        self.history_t.append(self.t)
        self.history_nPI.append(c_PI)
        self.history_nPO.append(c_PO)
        self.history_nTemp.append(c_T)

        self._maybe_store_snapshot()

    def step(self):
        """
        1か月分の更新（t → t+1）。

        処理順：
          1) 計画改定タイミングなら P_eff と C_hat_interval を更新
          2) 都市環境 C を更新（進捗低減期間なら K を縮小）
          3) エージェントの居住地選択（閾値モデル）
          4) 仮住まい期間 temp_months は意思決定の“後”に +1（仕様B）
          5) first_decision を記録し、同調圧力の集計を更新
          6) t を進め、ログを記録
        """
        # --- 計画改定（t==T1, t==T2） ---
        if self.t == Config.T1:
            self._adopt_new_plan(alpha=float(Config.ALPHA1), t_adopt=self.t)
        if self.t == Config.T2:
            self._adopt_new_plan(alpha=float(Config.ALPHA2), t_adopt=self.t)

        # --- 都市環境の更新 ---
        if float(np.linalg.norm(self.P_eff)) > 0.0:
            stepK = self.K * (1.0 - Config.LAMBDA) if self._is_delayed(self.t) else self.K
            self.C = self.C + stepK * normalize(self.P_eff)

        # 将来像は区間内固定のため、現時点では C_mean を用いて評価する
        C_mean = (self.C + self.C_hat_interval) / 2.0

        # --- 居住地選択 ---
        for ag in self.agents:
            if ag.state == HousingState.PO:
                continue

            started_state = ag.state
            W_t = ag.temp_months / Config.T_MAX
            U_i = float(np.dot(ag.weights, C_mean))

            term_ret = (1.0 - ag.love) * (1.0 - ag.income)
            term_stay = (1.0 - ag.love) * ag.income

            # TI/TO → {PI, PO, 継続}
            if started_state in (HousingState.TI, HousingState.TO):
                B_ret = self._get_B(ag, "return")
                T_ret = (Config.THETA_1
                         - Config.THETA_2 * W_t
                         + Config.THETA_3 * term_ret
                         + Config.GAMMA * B_ret)

                if U_i > T_ret:
                    ag.state = HousingState.PI
                else:
                    # 流出判定は D_i < T_exit（仕様）
                    D_i = Config.THETA_5 - Config.THETA_2 * W_t
                    if started_state == HousingState.TO:
                        D_i = Config.THETA_5 - Config.BETA * Config.THETA_2 * W_t

                    B_exit = self._get_B(ag, "exit")
                    T_exit = (Config.THETA_4
                              + Config.THETA_3 * term_stay
                              + Config.GAMMA * B_exit)

                    if D_i < T_exit:
                        ag.state = HousingState.PO

            # PI → {PO, 継続}
            elif started_state == HousingState.PI:
                B_stay = self._get_B(ag, "stay")
                T_stay = (Config.THETA_4
                          + Config.THETA_3 * term_stay
                          + Config.GAMMA * B_stay)

                if U_i < T_stay:
                    ag.state = HousingState.PO

            # 仕様B：仮住まい期間は意思決定の“後”に 1 増える
            if started_state in (HousingState.TI, HousingState.TO):
                ag.temp_months += 1

            # 「最初の決定」を1回だけ記録
            self._log_first_decision(ag, started_state)

        # first_decision の集計（同調圧力計算にのみ使用）
        self._aggregate_first_decisions()

        # --- 次月へ ---
        self.t += 1
        self._record()

    def run(self):
        """初期化後、t=T_MAX まで step() を逐次実行する。"""
        self.initialize()
        while self.t < Config.T_MAX:
            self.step()


# ==========================================
# 5. 評価出力（表・グラフ）
# ==========================================
def print_snapshot_tables(snap: Dict[str, Any]):
    """
    スナップショットを標準出力に整形して表示する（人間が読むための表）。

    出力内容：
      - 時点 t と都市環境 C(t)
      - 各居住状態 H ごとの:
          - 人数 n
          - 世帯タイプ L のカウント
          - love(A_i), income(I_i) の要約統計
    """
    t = snap["t"]
    C = snap["C"]
    print("\n" + "=" * 80)
    print(f"Snapshot at t = {t}")
    print(f"City environment C(t) = [{C[0]:.4f}, {C[1]:.4f}, {C[2]:.4f}]")
    print("-" * 80)

    by_state = snap["by_state"]
    for hs in HousingState:
        s = by_state[hs]
        n = s["n"]
        L_counts = s["L_counts"]
        A_stats = s["A_stats"]
        I_stats = s["I_stats"]

        print(f"[{hs.name}] n={n}")
        print("  L_counts:", {lt.name: L_counts[lt] for lt in HouseholdType})
        print(f"  A(love):   mean={A_stats['mean']:.4f}, std={A_stats['std']:.4f}, "
              f"q25={A_stats['q25']:.4f}, med={A_stats['median']:.4f}, q75={A_stats['q75']:.4f}")
        print(f"  I(income): mean={I_stats['mean']:.4f}, std={I_stats['std']:.4f}, "
              f"q25={I_stats['q25']:.4f}, med={I_stats['median']:.4f}, q75={I_stats['q75']:.4f}")
        print("-" * 80)


def plot_L_stackedbar_percent_over_time(
    sim: Simulation,
    times: List[int],
    fname: str = "snapshot_L_share_stacked_T1_T2_TAU.png",
):
    """
    t1/t2/τ のスナップショットをまとめて、L 構成比（各バー100%）の積み上げ棒グラフを描画する。

    並び順：
      - H ごとに t1→t2→τ（例：PI@t1, PI@t2, PI@τ, PO@t1, ...）
    """
    snaps = {t: sim.snapshots[t] for t in times}
    states = list(HousingState)
    types = list(HouseholdType)

    labels: List[str] = []
    y_by_type: Dict[HouseholdType, List[float]] = {lt: [] for lt in types}

    for hs in states:
        for tt in times:
            snap = snaps[tt]
            by_state = snap["by_state"]
            n = by_state[hs]["n"]

            labels.append(f"{hs.name}\n t={tt}")
            for lt in types:
                cnt = by_state[hs]["L_counts"][lt]
                pct = 100.0 * cnt / n if n > 0 else 0.0
                y_by_type[lt].append(pct)

    x = np.arange(len(labels))
    bottom = np.zeros(len(labels), dtype=float)

    plt.figure(figsize=(14, 6))
    for lt in types:
        y = np.array(y_by_type[lt], dtype=float)
        plt.bar(x, y, bottom=bottom, label=lt.name)
        bottom += y

    plt.xticks(x, labels, rotation=0)
    plt.ylim(0, 100)
    plt.xlabel("Housing state H and time")
    plt.ylabel("Share of household type L (%)")
    plt.title("Household-type composition (100% stacked) across snapshots (t1, t2, τ)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.show()


def plot_box_by_state_over_time(
    sim: Simulation,
    times: List[int],
    attr: str,
    fname: str,
):
    """
    連続属性（love または income）について、t1/t2/τ をまとめて箱ひげ図で描画する。

    並び順：
      - H ごとに t1→t2→τ（例：PI@t1, PI@t2, PI@τ, PO@t1, ...）
    """
    key = "A_values" if attr == "love" else "I_values"
    states = list(HousingState)

    data: List[List[float]] = []
    labels: List[str] = []

    for hs in states:
        for tt in times:
            snap = sim.snapshots[tt]
            by_state = snap["by_state"]
            data.append(by_state[hs][key])
            labels.append(f"{hs.name}\n t={tt}")

    plt.figure(figsize=(14, 6))
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.xlabel("Housing state H and time")
    plt.ylabel(attr)
    plt.title(f"Distribution of {attr} across snapshots (t1, t2, τ)")
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.show()


def plot_C_components_over_time(
    sim: Simulation,
    times: List[int],
    fname: str = "snapshot_C_components_T1_T2_TAU.png",
):
    """
    都市環境 C の3成分を、成分ごとに t1→t2→τ を並べて棒グラフ（グループ化）で表示する。

    並び順：
      - 成分（E/A/S）ごとに t1/t2/τ を並べる（legend は時点）
    """
    C_mat = np.array([sim.snapshots[t]["C"] for t in times], dtype=float)  # (len(times), 3)

    components = ["C_E (Economy)", "C_A (Amenity)", "C_S (Safety)"]
    x = np.arange(len(components))
    width = 0.25

    plt.figure(figsize=(10, 5))
    for k, tt in enumerate(times):
        offset = (k - (len(times) - 1) / 2) * width
        plt.bar(x + offset, C_mat[k, :], width, label=f"t={tt}")

    plt.xticks(x, components)
    plt.xlabel("Component")
    plt.ylabel("C value")
    plt.title("Urban environment components C across snapshots (t1, t2, τ)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.show()


# ==========================================
# 6. 実行・可視化（時系列＋評価スナップショット）
# ==========================================
if __name__ == "__main__":
    # 乱数シードは固定しない（要求仕様）
    # 再現性が欲しい場合のみ、以下を有効化する：
    # random.seed(42)
    # np.random.seed(42)

    # --- A/I の「母集団（固定プール）」を1回だけ生成 ---
    base_A_pool = make_truncated_normal_pool(
        n=Config.N_HOUSEHOLDS,
        mu=Config.LOVE_MU,
        sigma=Config.LOVE_SIGMA,
        lower=Config.LOVE_LOW,
        upper=Config.LOVE_HIGH,
    )
    base_I_pool = make_truncated_lognormal_pool(
        n=Config.N_HOUSEHOLDS,
        mu=Config.INC_LOGMU,
        sigma=Config.INC_LOGSIGMA,
        lower=Config.INC_LOW,
        upper=Config.INC_HIGH,
    )

    # --- 1 run（割当は run ごとにシャッフルされる） ---
    sim = Simulation(base_A_pool, base_I_pool, shuffle_assignment_each_run=True)
    sim.run()

    # ---- 時系列（人数） ----
    # 非PO（自治体外に定着していない）人口：PI + TI + TO
    n_nonPO = [pi + tmp for pi, tmp in zip(sim.history_nPI, sim.history_nTemp)]

    plt.figure(figsize=(10, 6))
    plt.plot(sim.history_t, sim.history_nPI, label="n_PI (Permanent Inside)", linewidth=2)
    plt.plot(sim.history_t, sim.history_nPO, label="n_PO (Permanent Outside)", linewidth=2)
    plt.plot(sim.history_t, sim.history_nTemp, label="n_TI + n_TO (Temporary)", linewidth=2)
    plt.plot(sim.history_t, n_nonPO, label="n_PI + n_TI + n_TO (Non-PO total)", linewidth=2, linestyle=":")

    plt.axvline(x=Config.T1, linestyle="--", alpha=0.7, label="Plan Update 1 (pre)")
    plt.axvline(x=Config.T2, linestyle="--", alpha=0.7, label="Plan Update 2 (pre)")

    plt.xlabel("Time t (months)")
    plt.ylabel("Population (Households)")
    plt.title("Housing State Simulation")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("result_graph.png", dpi=200)
    plt.show()

    # ---- 評価スナップショット（表＋図） ----
    times = [Config.T1, Config.T2, Config.TAU]

    # 表：各時点を順に表示
    for tt in times:
        if tt in sim.snapshots:
            print_snapshot_tables(sim.snapshots[tt])
        else:
            print(f"[WARN] snapshot at t={tt} is not stored.")

    # 図：3時点をまとめて表示
    plot_L_stackedbar_percent_over_time(sim, times, fname="snapshot_L_share_stacked_T1_T2_TAU.png")
    plot_box_by_state_over_time(sim, times, attr="love", fname="snapshot_love_box_T1_T2_TAU.png")
    plot_box_by_state_over_time(sim, times, attr="income", fname="snapshot_income_box_T1_T2_TAU.png")
    plot_C_components_over_time(sim, times, fname="snapshot_C_components_T1_T2_TAU.png")
