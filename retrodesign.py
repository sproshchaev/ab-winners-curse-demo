#!/usr/bin/env python3
"""Сколько завышает победитель A/B-теста: формулы и проверка своего теста.

Пример: python3 retrodesign.py --users 120000 --base 0.05 --lift 0.02
Со снижением оценки: добавить --observed 0.06 --tau 0.01

Нужен только Python 3. Все эффекты относительные: 0.02 значит рост на 2%.
"""
import argparse
import math
from statistics import NormalDist

NORMAL = NormalDist()
ALPHA_Z = 1.959964  # двусторонний уровень 5%


def variance_of_difference(base, lift):
    """Дисперсия разницы конверсий при одном пользователе на группу."""
    variant = base * (1 + lift)
    return base * (1 - base) + variant * (1 - variant)


def se_absolute(base, lift, users):
    """Стандартная ошибка разницы конверсий при users пользователей на группу."""
    return math.sqrt(variance_of_difference(base, lift) / users)


def users_for_power(power, base, lift):
    """Пользователей на группу для заданной мощности по положительному результату."""
    z_total = ALPHA_Z + NORMAL.inv_cdf(power)
    return math.ceil(variance_of_difference(base, lift) * (z_total / (base * lift)) ** 2)


def power(users, base, lift):
    """Вероятность получить значимый рост."""
    effect = base * lift
    return 1 - NORMAL.cdf(ALPHA_Z - effect / se_absolute(base, lift, users))


def exaggeration(users, base, lift):
    """Во сколько раз средний значимый рост больше настоящего."""
    z_effect = base * lift / se_absolute(base, lift, users)
    return 1 + NORMAL.pdf(ALPHA_Z - z_effect) / (power(users, base, lift) * z_effect)


def minimal_significant_lift(users, base):
    """Расчётный порог значимого роста при этой выборке."""
    return ALPHA_Z * se_relative(users, base)


def se_relative(users, base):
    """Стандартная ошибка относительного роста, когда сам рост мал."""
    return math.sqrt(2 * (1 - base) / (base * users))


def shrink_factor(tau, se):
    """Доля наблюдаемого роста, которая остаётся при разбросе настоящих эффектов tau."""
    return tau ** 2 / (tau ** 2 + se ** 2)


def main():
    parser = argparse.ArgumentParser(
        description="Сколько завышает победитель A/B-теста: формулы и проверка своего теста.")
    parser.add_argument("--users", type=int, required=True, help="пользователей на группу")
    parser.add_argument("--base", type=float, required=True, help="конверсия контроля, 0.05")
    parser.add_argument("--lift", type=float, required=True,
                        help="настоящий рост, в который вы верите до теста, 0.02")
    parser.add_argument("--observed", type=float, help="рост, который показал тест, 0.06")
    parser.add_argument("--tau", type=float,
                        help="разброс настоящих эффектов в ваших прошлых тестах, 0.01")
    args = parser.parse_args()

    if args.users <= 0:
        parser.error("--users: нужно положительное число пользователей на группу")
    if not 0 < args.base < 1:
        parser.error("--base: конверсия задаётся долей между 0 и 1, например 0.05")
    if not 0 < args.base * (1 + args.lift) < 1:
        parser.error("--lift: конверсия варианта должна остаться между 0 и 1, рост задаётся долей")
    if (args.observed is None) != (args.tau is None):
        parser.error("--observed и --tau задаются только вместе")

    print(f"мощность {power(args.users, args.base, args.lift):.3f}")
    print(f"минимальный значимый рост {100 * minimal_significant_lift(args.users, args.base):.2f}%")
    if args.lift > 0:
        print(f"завышение у победителя {exaggeration(args.users, args.base, args.lift):.2f} раза")
    else:
        print("завышение у победителя не определено: настоящий рост не положительный")
    if args.observed is not None:
        kept = shrink_factor(args.tau, se_relative(args.users, args.base))
        print(f"оценка после снижения {100 * kept * args.observed:.2f}% (остаётся доля {kept:.3f})")


if __name__ == "__main__":
    main()
