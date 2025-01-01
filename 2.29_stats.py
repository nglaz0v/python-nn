"""
Уравнение статистического теста на согласие (например, тест хи-квадрат).
"""

from scipy.stats import chi2_contingency

# Пример данных для теста на согласие
observed = [[10, 15, 20],
            [5, 10, 15]]

# Выполнение теста хи-квадрат
chi2, p_value, dof, expected = chi2_contingency(observed)

print("Статистика хи-квадрат:", chi2)
print("р-значение:", p_value)
print("Степени свободы:", dof)
print("Ожидаемые частоты:", expected)
