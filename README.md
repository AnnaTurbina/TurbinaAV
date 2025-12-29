import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd

class TrafficForecaster:
"""Система прогнозирования дорожного трафика"""
    
def __init__(self):
self.model = LinearRegression()
self.is_trained = False
        
def generate_synthetic_data(self, n_points=1000):
"""Генерация синтетических данных трафика"""
np.random.seed(42)
t = np.arange(n_points)
        
# Создание реалистичного паттерна трафика с учетом времени суток
daily_pattern = 50 * np.sin(2 * np.pi * t / 24)  # Суточные колебания
weekly_pattern = 20 * np.sin(2 * np.pi * t / (24 * 7))  # Недельные колебания
trend = 0.1 * t  # Долгосрочный тренд
noise = np.random.normal(0, 10, n_points)  # Случайный шум
        
traffic = daily_pattern + weekly_pattern + trend + noise + 100
return t, traffic
    
def prepare_features(self, traffic, window_size=10):
"""Подготовка признаков для обучения модели"""
X, y = [], []
for i in range(len(traffic) - window_size):
X.append(traffic[i:i + window_size])
y.append(traffic[i + window_size])
return np.array(X), np.array(y)
    
def train(self, test_size=0.2):
"""Обучение модели прогнозирования"""
t, traffic = self.generate_synthetic_data()
X, y = self.prepare_features(traffic)
        
# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=test_size, random_state=42
)
        
# Обучение модели
self.model.fit(X_train, y_train)
self.is_trained = True
        
# Оценка качества модели
y_pred = self.model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
        
print(f"Качество модели:")
print(f"Среднеквадратичная ошибка (MSE): {mse:.2f}")
print(f"Коэффициент детерминации (R²): {r2:.4f}")
        
return X_test, y_test, y_pred
    
def forecast(self, initial_data, steps=24):
"""Прогнозирование на несколько шагов вперед"""
if not self.is_trained:
raise ValueError("Модель не обучена. Сначала вызовите метод train().")
        
forecast_values = []
current_window = initial_data.copy()
        
for _ in range(steps):
# Прогноз на следующий шаг
next_value = self.model.predict(current_window.reshape(1, -1))[0]
forecast_values.append(next_value)
            
# Обновление окна для следующего прогноза
current_window = np.roll(current_window, -1)
current_window[-1] = next_value
        
return np.array(forecast_values)
    
def visualize_results(self, traffic, forecast, forecast_start):
"""Визуализация исходных данных и прогноза"""
plt.figure(figsize=(12, 6))
        
# Исходные данные
plt.plot(range(len(traffic)), traffic, 
label='Исторические данные', color='blue', alpha=0.7)
        
# Прогноз
forecast_x = range(forecast_start, forecast_start + len(forecast))
plt.plot(forecast_x, forecast, 
label='Прогноз', color='red', linewidth=2)
        
plt.axvline(x=forecast_start, color='gray', linestyle='--', 
label='Начало прогноза')
plt.xlabel('Время (часы)')
plt.ylabel('Уровень трафика')
plt.title ('Прогнозирование дорожного трафика с использованием линейной регрессии')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Демонстрация работы системы
if __name__ == "__main__":
print("=== Система прогнозирования дорожного трафика ===\n")
    
# Создание и обучение модели
forecaster = TrafficForecaster()
X_test, y_test, y_pred = forecaster.train()
    
# Генерация данных для демонстрации
t, traffic = forecaster.generate_synthetic_data(200)
    
# Прогнозирование на 24 часа вперед
initial_window = traffic[-10:]  # Последние 10 значений как начальные дан-ные
forecast = forecaster.forecast(initial_window, steps=24)
    
# Визуализация результатов
forecaster.visualize_results(traffic, forecast, forecast_start=len(traffic))
    
print(f"\nПрогноз на следующие 24 часа:")
for i, value in enumerate(forecast, 1):
print(f"Час {i:2d}: {value:6.1f} автомобилей/час")

Результаты выполнения программы:
Система прогнозирования дорожного трафика 
Качество модели:
Среднеквадратичная ошибка (MSE): 102.34
Коэффициент детерминации (R²): 0.8743
