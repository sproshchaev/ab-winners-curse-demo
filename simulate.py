#!/usr/bin/env python3
"""Проклятие победителя в A/B-тестах: все таблицы статьи из одного фиксированного seed."""
import math

import numpy as np

from retrodesign import (
    ALPHA_Z,
    exaggeration,
    minimal_significant_lift,
    se_relative,
    shrink_factor,
    users_for_power,
)

SEED = 20261031
BASE = 0.05       # конверсия контроля
TRUE_LIFT = 0.02  # настоящий относительный рост варианта
RUNS = 200_000    # тестов на строку таблицы
DAYS = 14


def z_scores(conv_a, conv_b, users):
    """Z-статистика двух долей с общей дисперсией, как в типовом инструменте тестов."""
    pooled = (conv_a + conv_b) / (2 * users)
    return (conv_b - conv_a) / users / np.sqrt(pooled * (1 - pooled) * 2 / users)


def relative_lift(conv_a, conv_b):
    return (conv_b - conv_a) / conv_a


def table_exaggeration(rng):
    print(f"ТАБЛИЦА 1. Один взгляд в конце. Настоящий рост +{100 * TRUE_LIFT:.1f}% "
          f"при конверсии {100 * BASE:.0f}%, {RUNS} тестов на строку")
    print("мощность | на_группу | мощность_факт | мин_значимый_рост_% | рост_победителя_% "
          "| завышение | по_формуле | не_тот_знак_%")
    gaps = []
    for target in (0.10, 0.20, 0.30, 0.50, 0.80, 0.95):
        users = users_for_power(target, BASE, TRUE_LIFT)
        conv_a = rng.binomial(users, BASE, RUNS)
        conv_b = rng.binomial(users, BASE * (1 + TRUE_LIFT), RUNS)
        z = z_scores(conv_a, conv_b, users)
        lift = relative_lift(conv_a, conv_b)
        winners = z > ALPHA_Z
        losers = z < -ALPHA_Z
        winner_lift = lift[winners].mean()
        ratio = winner_lift / TRUE_LIFT
        formula = exaggeration(users, BASE, TRUE_LIFT)
        wrong_sign = 100 * losers.sum() / (winners.sum() + losers.sum())
        gaps.append(abs(ratio - formula))
        print(f"{target:.2f} | {users} | {winners.mean():.3f} "
              f"| {100 * minimal_significant_lift(users, BASE):.2f} "
              f"| {100 * winner_lift:.2f} | {ratio:.2f} | {formula:.2f} "
              f"| {wrong_sign:.2f}")
    return max(gaps)


def table_required_users():
    print(f"\nТАБЛИЦА 2. Пользователей на группу для мощности 80% при конверсии {100 * BASE:.0f}%")
    print("настоящий_рост_% | на_группу")
    for lift in (0.005, 0.01, 0.02, 0.05, 0.10):
        print(f"{100 * lift:.1f} | {users_for_power(0.80, BASE, lift)}")


def table_peeking(rng):
    users_per_day = math.ceil(users_for_power(0.80, BASE, TRUE_LIFT) / DAYS)
    print(f"\nТАБЛИЦА 3. Взгляд каждый день и остановка на первой значимости. "
          f"{users_per_day} на группу в день, {DAYS} дней, мощность к концу 80%")
    print("сценарий | победа_% | значимо_в_любую_сторону_% | рост_победителя_% | завышение "
          "| медианный_день_остановки")
    users = users_per_day * np.arange(1, DAYS + 1)
    rows = np.arange(RUNS)
    for name, lift, every_day in (
        ("эффекта_нет_взгляд_в_конце", 0.0, False),
        ("эффекта_нет_взгляд_каждый_день", 0.0, True),
        ("эффект_есть_взгляд_в_конце", TRUE_LIFT, False),
        ("эффект_есть_взгляд_каждый_день", TRUE_LIFT, True),
    ):
        conv_a = rng.binomial(users_per_day, BASE, (RUNS, DAYS)).cumsum(axis=1)
        conv_b = rng.binomial(users_per_day, BASE * (1 + lift), (RUNS, DAYS)).cumsum(axis=1)
        z = z_scores(conv_a, conv_b, users)
        # Тест останавливается на первой значимости в любую сторону, победа только в плюс
        first_look = 0 if every_day else DAYS - 1
        significant = np.abs(z[:, first_look:]) > ALPHA_Z
        any_side = significant.any(axis=1)
        stop_day = first_look + significant.argmax(axis=1)
        won = any_side & (z[rows, stop_day] > 0)
        lifts = relative_lift(conv_a[rows, stop_day], conv_b[rows, stop_day])
        mean_lift = lifts[won].mean()
        ratio = f"{mean_lift / lift:.2f}" if lift else "нет"
        print(f"{name} | {100 * won.mean():.2f} | {100 * any_side.mean():.2f} "
              f"| {100 * mean_lift:.2f} | {ratio} | {np.median(stop_day[won]) + 1:g}")


def program(rng, tau, users, heavy_tails=False):
    """Поток тестов: у каждого свой настоящий рост вокруг нуля со стандартным разбросом tau."""
    if heavy_tails:
        true = tau / math.sqrt(3) * rng.standard_t(3, RUNS)  # у t(3) дисперсия равна 3
    else:
        true = rng.normal(0, tau, RUNS)
    se = se_relative(users, BASE)
    estimate = true + rng.normal(0, se, RUNS)
    return true, estimate, se, estimate / se > ALPHA_Z


def tau_from_history(estimate, se):
    """Разброс настоящих эффектов по прошлым оценкам без отбора."""
    return math.sqrt(max(estimate.var() - se ** 2, 0))


def table_program(rng):
    print(f"\nТАБЛИЦА 4. Поток тестов: настоящие эффекты разные, в среднем ноль. "
          f"{RUNS} тестов на строку")
    print("на_группу | разброс_% | победителей_% | заявлено_% | на_самом_деле_% | завышение "
          "| победитель_вредит_%")
    for users in (20_000, 100_000, 500_000):
        for tau in (0.005, 0.01, 0.02):
            true, estimate, _, won = program(rng, tau, users)
            claimed, real = estimate[won].mean(), true[won].mean()
            print(f"{users} | {100 * tau:.1f} | {100 * won.mean():.2f} "
                  f"| {100 * claimed:.2f} | {100 * real:.2f} "
                  f"| {claimed / real:.2f} "
                  f"| {100 * (true[won] < 0).mean():.1f}")


def table_fixes(rng):
    users, tau = 100_000, 0.01
    print(f"\nТАБЛИЦА 5. Что возвращает оценку к настоящему эффекту. "
          f"{users} на группу, разброс {100 * tau:.1f}%")
    print("способ | оценка_% | на_самом_деле_% | завышение | осталось_тестов_%")

    def line(name, value, real, kept=1.0):
        print(f"{name} | {100 * value:.2f} | {100 * real:.2f} | {value / real:.2f} "
              f"| {100 * kept:.1f}")

    true, estimate, se, won = program(rng, tau, users)
    true_won, estimate_won = true[won], estimate[won]
    real = true_won.mean()
    line("как_есть", estimate_won.mean(), real)

    rerun = true_won + rng.normal(0, se, len(true_won))
    line("повтор_оценка_всех_повторов", rerun.mean(), real)
    won_again = rerun / se > ALPHA_Z
    line("повтор_только_победившие_снова", rerun[won_again].mean(), true_won[won_again].mean(),
         won_again.mean())

    line("снижение_разброс_известен", shrink_factor(tau, se) * estimate_won.mean(), real)
    # На практике разброс неизвестен и оценивается по прошлым тестам, здесь их заменяет эта же выборка
    line("снижение_разброс_по_истории",
         shrink_factor(tau_from_history(estimate, se), se) * estimate_won.mean(), real)

    # Тот же рецепт, когда нормальное распределение эффектов неверно: хвосты тяжёлые
    true, estimate, se, won = program(rng, tau, users, heavy_tails=True)
    real = true[won].mean()
    line("тяжёлые_хвосты_как_есть", estimate[won].mean(), real)
    line("тяжёлые_хвосты_снижение",
         shrink_factor(tau_from_history(estimate, se), se) * estimate[won].mean(), real)


def controls(rng, max_gap):
    users = users_for_power(0.80, BASE, TRUE_LIFT)
    conv_a = rng.binomial(users, BASE, RUNS)
    conv_b = rng.binomial(users, BASE, RUNS)
    significant = np.abs(z_scores(conv_a, conv_b, users)) > ALPHA_Z
    print("\nКОНТРОЛИ")
    print(f"тесты_без_эффекта_значимо_% {100 * significant.mean():.2f}")
    print(f"наибольшее_расхождение_симуляции_и_формулы {max_gap:.3f}")


def main():
    rng = np.random.default_rng(SEED)
    max_gap = table_exaggeration(rng)
    table_required_users()
    table_peeking(rng)
    table_program(rng)
    table_fixes(rng)
    controls(rng, max_gap)


if __name__ == "__main__":
    main()
