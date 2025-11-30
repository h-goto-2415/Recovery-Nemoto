from __future__ import annotations

import enum
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict


# ==== Enums for household state/type ====

class HousingState(enum.Enum):
    PI = "PI"  # Permanent, Inside municipality
    PO = "PO"  # Permanent, Outside
    TI = "TI"  # Temporary, Inside
    TO = "TO"  # Temporary, Outside


class HouseholdType(enum.Enum):
    YC = "YC"  # Young, with children
    YN = "YN"  # Young, no children
    EW = "EW"  # Elderly, working
    ER = "ER"  # Elderly, retired


# ==== Data classes ====

@dataclass
class Resident:
    id: int
    state: HousingState
    household_type: HouseholdType
    love: float        # A_i
    income: float      # I_i（対数正規）
    temp_months: int = 0  # W_i
    w_E: float = 0.0
    w_A: float = 0.0
    w_S: float = 0.0

    def update_temporary_months(self) -> None:
        if self.state in (HousingState.TI, HousingState.TO):
            self.temp_months += 1
        else:
            self.temp_months = 0


@dataclass
class SimulationConfig:
    # 時間関連
    t1: int = 36
    t2: int = 72
    tau: int = 120

    # 自治体の復興目標
    V_pol: float = 1.1

    # 計画改定の大きさ α1, α2
    alpha1: float = 0
    alpha2: float = 0

    # 計画改定時の事業進捗の低減 λ
    lam: float = 0.3

    # 居住地選択パラメータ θ, β
    theta1: float = 4
    theta2: float = 0.02
    theta3: float =4
    theta4: float = 4
    beta: float = 1.5

    # 世帯数
    n: int = 10000

    # 初期の都市環境 C(0) = (Economy, Amenity, Safety)
    C0: Tuple[float, float, float] = (0.25, 0.5, 0.25)


# ==== Preference tables (Table 9) ====

# (HousingState, HouseholdType) -> (w_E, w_A, w_S)
PREF_TABLE: Dict[Tuple[HousingState, HouseholdType], Tuple[float, float, float]] = {}


def _init_pref_table() -> None:
    # YC
    PREF_TABLE[(HousingState.PI, HouseholdType.YC)] = (0.4,   0.4,   0.2)
    PREF_TABLE[(HousingState.TI, HouseholdType.YC)] = (0.325, 0.325, 0.35)
    PREF_TABLE[(HousingState.TO, HouseholdType.YC)] = (0.3,   0.4,   0.3)

    # YN
    PREF_TABLE[(HousingState.PI, HouseholdType.YN)] = (0.45,  0.325, 0.225)
    PREF_TABLE[(HousingState.TI, HouseholdType.YN)] = (0.375, 0.25,  0.375)
    PREF_TABLE[(HousingState.TO, HouseholdType.YN)] = (0.35,  0.325, 0.325)

    # EW
    PREF_TABLE[(HousingState.PI, HouseholdType.EW)] = (0.367, 0.367, 0.267)
    PREF_TABLE[(HousingState.TI, HouseholdType.EW)] = (0.292, 0.292, 0.417)
    PREF_TABLE[(HousingState.TO, HouseholdType.EW)] = (0.267, 0.367, 0.367)

    # ER
    PREF_TABLE[(HousingState.PI, HouseholdType.ER)] = (0.3,   0.4,   0.3)
    PREF_TABLE[(HousingState.TI, HouseholdType.ER)] = (0.225, 0.325, 0.45)
    PREF_TABLE[(HousingState.TO, HouseholdType.ER)] = (0.2,   0.4,   0.4)

    # PO については、参加しないので定義なし（最後に持っていた重みを維持する想定）


_init_pref_table()


def assign_preferences(resident: Resident) -> None:
    key = (resident.state, resident.household_type)
    if key in PREF_TABLE:
        wE, wA, wS = PREF_TABLE[key]
        s = wE + wA + wS
        resident.w_E = wE / s
        resident.w_A = wA / s
        resident.w_S = wS / s
    else:
        # PO など fallback
        resident.w_E = resident.w_A = resident.w_S = 1.0 / 3.0


# ==== Utility helpers for vectors ====

def vec_add(a: Tuple[float, float, float],
            b: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def vec_sub(a: Tuple[float, float, float],
            b: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vec_mul_scalar(a: Tuple[float, float, float],
                   k: float) -> Tuple[float, float, float]:
    return (a[0] * k, a[1] * k, a[2] * k)


def vec_norm(a: Tuple[float, float, float]) -> float:
    return math.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2)


def vec_unit(a: Tuple[float, float, float]) -> Tuple[float, float, float]:
    n = vec_norm(a)
    if n == 0:
        return (0.0, 0.0, 0.0)
    return (a[0] / n, a[1] / n, a[2] / n)


# ==== Simulation core ====

class RecoverySimulation:
    """
    ミニマムだけど、論文仕様にかなり沿った ABM 実装。
    """

    def __init__(self, config: SimulationConfig):
        self.cfg = config
        self.t = 0

        # 都市環境
        self.C: Tuple[float, float, float] = config.C0

        # 住民
        self.residents: List[Resident] = []
        self._init_population()

        # 計画関連
        self.V0: Tuple[float, float, float] | None = None
        self.V1: Tuple[float, float, float] | None = None
        self.V2: Tuple[float, float, float] | None = None

        self.P0: Tuple[float, float, float] | None = None
        self.P1: Tuple[float, float, float] | None = None
        self.P2: Tuple[float, float, float] | None = None

        self.Peff0: Tuple[float, float, float] | None = None
        self.Peff1: Tuple[float, float, float] | None = None
        self.Peff2: Tuple[float, float, float] | None = None

        self.K: float | None = None

        self.Cbar0: Tuple[float, float, float] | None = None
        self.Cbar1: Tuple[float, float, float] | None = None
        self.Cbar2: Tuple[float, float, float] | None = None

        # ログ
        self.history_C: List[Tuple[float, float, float]] = [self.C]
        self.history_nPI: List[int] = [self.count_state(HousingState.PI)]

        # t=0 の計画を生成
        self._create_plan0()

    # ----- population -----

    def _init_population(self) -> None:
        cfg = self.cfg
        n = cfg.n

        # 暫定：20% PI, 80% TI, 0% TO
        n_PI = int(n * 0.25)
        n_TI = int(n * 0.5)
        n_TO = n - n_PI - n_TI

        # 世帯類型の比率（仮）
        type_probs = [
            (HouseholdType.YC, 0.25),
            (HouseholdType.YN, 0.3),
            (HouseholdType.EW, 0.1),
            (HouseholdType.ER, 0.35),
        ]

        def random_household_type() -> HouseholdType:
            r = random.random()
            cum = 0.0
            for ht, p in type_probs:
                cum += p
                if r <= cum:
                    return ht
            return type_probs[-1][0]

        def draw_love() -> float:
            return random.uniform(0.1, 0.9)

        def draw_income() -> float:
            # 対数正規分布 Income ~ logN(mu, sigma^2)
            mu = -0.8
            sigma = 0.5
            return math.exp(random.gauss(mu, sigma))

        rid = 0
        for _ in range(n_PI):
            rid += 1
            r = Resident(
                id=rid,
                state=HousingState.PI,
                household_type=random_household_type(),
                love=draw_love(),
                income=draw_income(),
            )
            assign_preferences(r)
            self.residents.append(r)

        for _ in range(n_TI):
            rid += 1
            r = Resident(
                id=rid,
                state=HousingState.TI,
                household_type=random_household_type(),
                love=draw_love(),
                income=draw_income(),
            )
            assign_preferences(r)
            self.residents.append(r)

        for _ in range(n_TO):
            rid += 1
            r = Resident(
                id=rid,
                state=HousingState.TO,
                household_type=random_household_type(),
                love=draw_love(),
                income=draw_income(),
            )
            assign_preferences(r)
            self.residents.append(r)

        random.shuffle(self.residents)

    # ----- planning: V_k, P_k, Peff_k, K, Cbar_k -----

    def _aggregate_preference(self) -> Tuple[float, float, float]:
        """
        w̅(t) = PI/TI/TO 住民の平均選好
        """
        total_E = total_A = total_S = 0.0
        count = 0
        for r in self.residents:
            if r.state in (HousingState.PI, HousingState.TI, HousingState.TO):
                assign_preferences(r)
                total_E += r.w_E
                total_A += r.w_A
                total_S += r.w_S
                count += 1
        if count == 0:
            return (1 / 3, 1 / 3, 1 / 3)
        return (total_E / count, total_A / count, total_S / count)

    def _compute_V_from_pref(self, wbar: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """
        5.1節のステップ1〜3に対応。
        """
        Vpol = self.cfg.V_pol
        wE, wA, wS = wbar

        # step 1
        factor = 3 * Vpol / (wE + wA + wS)
        V_E = factor * wE
        V_A = factor * wA
        V_S = factor * wS

        V = [V_E, V_A, V_S]

        # step 2: 下限 1 を適用
        fixed = []
        remaining_indices = []
        for j, val in enumerate(V):
            if val < 1.0:
                fixed.append((j, 1.0))
            else:
                remaining_indices.append(j)

        if not fixed:
            return (V_E, V_A, V_S)

        # step 3: 残りを再配分
        sum_fixed = sum(v for (_, v) in fixed)
        total = 3 * Vpol
        remaining_total = total - sum_fixed
        w_list = [wE, wA, wS]
        sum_w_remain = sum(w_list[j] for j in remaining_indices)

        if remaining_total <= 0 or sum_w_remain == 0:
            # 変なケースは一旦均等割り＋正規化
            V = [max(1.0, x) for x in V]
            s = sum(V)
            return (
                total * V[0] / s,
                total * V[1] / s,
                total * V[2] / s,
            )

        for j in remaining_indices:
            V[j] = remaining_total * w_list[j] / sum_w_remain

        for j, v in fixed:
            V[j] = v

        return (V[0], V[1], V[2])

    def _create_plan0(self) -> None:
        wbar = self._aggregate_preference()
        self.V0 = self._compute_V_from_pref(wbar)
        self.P0 = vec_sub(self.V0, self.C)
        self.Peff0 = self.P0

        # K = ||V0 - C(0)|| / tau
        self.K = vec_norm(self.P0) / self.cfg.tau

        # Cbar0 = C(0) + τ・K・P_eff0/||P_eff0||
        direction0 = vec_unit(self.Peff0)
        self.Cbar0 = vec_add(self.C, vec_mul_scalar(direction0, self.K * self.cfg.tau))

    def _create_plan1(self) -> None:
        # t = t1 で呼び出し
        wbar = self._aggregate_preference()
        self.V1 = self._compute_V_from_pref(wbar)
        self.P1 = vec_sub(self.V1, self.C)  # V1 - C(t1)
        assert self.Peff0 is not None
        self.Peff1 = self._mix_plans(self.cfg.alpha1, self.P1, self.Peff0)

        direction1 = vec_unit(self.Peff1)
        self.Cbar1 = vec_add(
            self.C, vec_mul_scalar(direction1, self.K * (self.cfg.tau - self.cfg.t1))
        )

    def _create_plan2(self) -> None:
        # t = t2 で呼び出し
        wbar = self._aggregate_preference()
        self.V2 = self._compute_V_from_pref(wbar)
        self.P2 = vec_sub(self.V2, self.C)  # V2 - C(t2)
        assert self.Peff1 is not None
        self.Peff2 = self._mix_plans(self.cfg.alpha2, self.P2, self.Peff1)

        direction2 = vec_unit(self.Peff2)
        self.Cbar2 = vec_add(
            self.C, vec_mul_scalar(direction2, self.K * (self.cfg.tau - self.cfg.t2))
        )

    @staticmethod
    def _mix_plans(alpha: float,
                   P_new: Tuple[float, float, float],
                   P_prev_eff: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """
        P_eff,k = α_k P_k/||P_k|| + (1-α_k) P_eff,k-1/||P_eff,k-1||
        """
        Pn = vec_unit(P_new)
        Pp = vec_unit(P_prev_eff)
        return (
            alpha * Pn[0] + (1 - alpha) * Pp[0],
            alpha * Pn[1] + (1 - alpha) * Pp[1],
            alpha * Pn[2] + (1 - alpha) * Pp[2],
        )

    # ----- helpers to get current Peff / Cbar -----

    def _current_Peff(self, t: int) -> Tuple[float, float, float]:
        if t < self.cfg.t1:
            assert self.Peff0 is not None
            return self.Peff0
        elif t < self.cfg.t2:
            assert self.Peff1 is not None
            return self.Peff1
        else:
            assert self.Peff2 is not None
            return self.Peff2

    def _current_Cbar(self, t: int) -> Tuple[float, float, float]:
        if t < self.cfg.t1:
            assert self.Cbar0 is not None
            return self.Cbar0
        elif t < self.cfg.t2:
            assert self.Cbar1 is not None
            return self.Cbar1
        else:
            assert self.Cbar2 is not None
            return self.Cbar2

    # ----- choice-related helpers -----

    def _utility_inside(self, resident: Resident) -> float:
        # U_i(t) = Σ_j w_ij * (C_j(t) + C̅_k,j)/2
        Cbar = self._current_Cbar(self.t)
        C = self.C
        return (
            resident.w_E * (C[0] + Cbar[0]) / 2.0
            + resident.w_A * (C[1] + Cbar[1]) / 2.0
            + resident.w_S * (C[2] + Cbar[2]) / 2.0
        )

    def _T_return(self, resident: Resident) -> float:
        cfg = self.cfg
        return cfg.theta1 * (1 - resident.love) * (1 - resident.income) - cfg.theta2 * resident.temp_months

    def _T_exit(self, resident: Resident) -> float:
        cfg = self.cfg
        return cfg.theta3 * (1 - resident.love) * resident.income

    def _T_stay(self, resident: Resident) -> float:
        cfg = self.cfg
        return cfg.theta3 * (1 - resident.love) * resident.income

    def _D_I(self, resident: Resident) -> float:
        cfg = self.cfg
        return cfg.theta4 - cfg.theta2 * resident.temp_months

    def _D_O(self, resident: Resident) -> float:
        cfg = self.cfg
        return cfg.theta4 - cfg.beta * cfg.theta2 * resident.temp_months

    @staticmethod
    def _logit_prob(x: float, y: float) -> float:
        """
        （いまは未使用のロジット関数。残しておく）
        """
        ex = math.exp(x)
        ey = math.exp(y)
        return ex / (ex + ey)

    # ----- one time step -----

    def step(self) -> None:
        t = self.t
        cfg = self.cfg

        # 1. 計画改定タイミングなら新しい計画を作る（flow 1）
        if t == cfg.t1:
            self._create_plan1()
        if t == cfg.t2:
            self._create_plan2()

        # 2. 居住地選択（flow 3）: ロジット → 閾値ルール
        new_states: Dict[int, HousingState] = {}
        for r in self.residents:
            # PO からは戻らない
            if r.state == HousingState.PO:
                new_states[r.id] = HousingState.PO
                continue

            assign_preferences(r)
            Ui = self._utility_inside(r)

            if r.state in (HousingState.TI, HousingState.TO):
                # まず「自治体内で定住するか？」
                T_ret = self._T_return(r)

                if Ui >= T_ret:
                    # 自治体内で定住
                    new_states[r.id] = HousingState.PI
                else:
                    # 次に「仮住まいを続けるか、自治体外に出るか？」
                    Dtemp = self._D_I(r) if r.state == HousingState.TI else self._D_O(r)
                    Texit = self._T_exit(r)

                    if Dtemp >= Texit:
                        # 仮住まい継続（TI / TO のまま）
                        new_states[r.id] = r.state
                    else:
                        # 自治体外で定住
                        new_states[r.id] = HousingState.PO

            elif r.state == HousingState.PI:
                # 「自治体内にとどまるか、自治体外に出るか？」
                Tstay = self._T_stay(r)

                if Ui >= Tstay:
                    new_states[r.id] = HousingState.PI
                else:
                    new_states[r.id] = HousingState.PO
            else:
                new_states[r.id] = r.state

        # 新しい状態を適用し、仮住まい期間を更新
        for r in self.residents:
            r.state = new_states[r.id]
            r.update_temporary_months()

        # 3. 都市環境 C(t+1) の更新（flow 2, 式(21)）
        assert self.K is not None
        Peff = self._current_Peff(t)
        direction = vec_unit(Peff)
        effective_K = self.K

        # t1, t2 の各 6 ヶ月前〜直前は λ だけ進捗が目減り
        # ただし α=0（実質改定なし）のときは停滞させない
        if self.cfg.alpha1 > 0 and cfg.t1 - 6 <= t <= cfg.t1 - 1:
            effective_K = (1 - cfg.lam) * self.K
        if self.cfg.alpha2 > 0 and cfg.t2 - 6 <= t <= cfg.t2 - 1:
            effective_K = (1 - cfg.lam) * self.K

        deltaC = vec_mul_scalar(direction, effective_K)
        self.C = vec_add(self.C, deltaC)

        # 4. ログ & 時刻更新
        self.t += 1
        self.history_C.append(self.C)
        self.history_nPI.append(self.count_state(HousingState.PI))

    # ----- helpers & run -----

    def count_state(self, state: HousingState) -> int:
        return sum(1 for r in self.residents if r.state == state)

    def _state_counts(self) -> Tuple[int, int, int]:
        return (
            self.count_state(HousingState.PI),
            self.count_state(HousingState.TI),
            self.count_state(HousingState.TO),
        )

    def _print_status(self, t: int) -> None:
        n_PI, n_TI, n_TO = self._state_counts()
        C_E, C_A, C_S = self.C
        print(
            f"t={t}: "
            f"PI={n_PI}, TI={n_TI}, TO={n_TO}, "
            f"C=({C_E:.6f}, {C_A:.6f}, {C_S:.6f})"
        )

    def _pi_type_composition(self) -> Dict[HouseholdType, float]:
        """
        t = tau 時点の PI 住民について、世帯類型ごとの比率を返す。
        """
        pi_residents = [r for r in self.residents if r.state == HousingState.PI]
        total_pi = len(pi_residents)
        comp: Dict[HouseholdType, float] = {
            HouseholdType.YC: 0.0,
            HouseholdType.YN: 0.0,
            HouseholdType.EW: 0.0,
            HouseholdType.ER: 0.0,
        }
        if total_pi == 0:
            return comp

        for r in pi_residents:
            comp[r.household_type] += 1.0

        for ht in comp:
            comp[ht] /= total_pi

        return comp

    def run(self) -> None:
        """
        t = 0,12,24,...,tau のときに PI/TI/TO と C(t) を出力する。
        """
        # t=0 の状態
        self._print_status(t=0)

        while self.t < self.cfg.tau:
            self.step()

            if self.t % 12 == 0 or self.t == self.cfg.tau:
                self._print_status(t=self.t)

        # t = tau 時点の PI の構成
        comp = self._pi_type_composition()
        print("Composition in PI at t=tau:")
        for ht in HouseholdType:
            ratio = comp.get(ht, 0.0)
            print(f"  {ht.value}: {ratio:.3f}")

    def summary(self) -> str:
        n0 = self.history_nPI[0]
        n_end = self.history_nPI[-1]
        return (
            f"t = {self.t}, "
            f"PI(0) = {n0}, PI(tau) = {n_end}, "
            f"C(0) = {self.history_C[0]}, "
            f"C(tau) = {self.history_C[-1]}"
        )


def main():
    random.seed(0)  # 再現性のため
    cfg = SimulationConfig()
    sim = RecoverySimulation(cfg)
    sim.run()
    print(sim.summary())


if __name__ == "__main__":
    main()
