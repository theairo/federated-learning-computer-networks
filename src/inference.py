# inference.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import numpy as np

# Імпорт архітектури моделі
from model import mnistNet
from utils.fl_utils import test_global

from config import OUTPUT_DIR

# --- КОНФІГУРАЦІЯ ---
MODEL_PATH = OUTPUT_DIR / 'server_model.pth' # Шлях до збереженої моделі
N_CLIENTS_USED = 2              # Потрібно для правильного завантаження даних
# ---------------------

def load_model_and_data(n_clients):
    """Завантажує модель та тестовий набір даних."""
    
    # 1. Завантаження моделі
    model = mnistNet()
    try:
        # Завантаження ваг
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()
        print(f"✅ Модель успішно завантажена з {MODEL_PATH}")
    except FileNotFoundError:
        print(f"❌ Помилка: Файл моделі {MODEL_PATH} не знайдено.")
        print("Переконайтеся, що ви запустили server.py і він успішно завершив роботу.")
        return None, None
    except Exception as e:
        print(f"❌ Помилка завантаження стану моделі: {e}")
        return None, None

    # 2. Завантаження тестового набору даних (використовуючи той самий механізм)
    # Зверніться до функції get_partitions у вашому data_utils.py. 
    # Останній елемент у списку partitions — це тестовий набір.
    try:
        from utils.data_utils import get_partitions
        partitions = get_partitions(n_clients)
        # Останній елемент у списку partitions — це тестовий набір
        test_data = partitions[-1]
        print(f"✅ Завантажено тестовий набір даних: {len(test_data)} зразків.")
        
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)
        return model, test_loader
        
    except ImportError:
        print("❌ Помилка: не знайдено data_utils.py або get_partitions.")
        return None, None


def visualize_inference(model, test_loader, num_examples=8):
    """Візуалізує приклади класифікації."""
    
    data_iterator = iter(test_loader)
    images, labels = next(data_iterator)
    
    # Зробити прогноз
    with torch.no_grad():
        output = model(images)
    
    # Отримати передбачені класи
    predictions = output.argmax(dim=1)
    
    fig = plt.figure(figsize=(10, 4))
    fig.suptitle('Приклади класифікації навченою моделлю', fontsize=16)

    for i in range(num_examples):
        ax = fig.add_subplot(2, 4, i + 1, xticks=[], yticks=[])
        
        # Відображення зображення (MNIST має 1 канал, тому [0] індекс)
        ax.imshow(images[i].squeeze(), cmap='gray')
        
        # Визначення кольору тексту: зелений, якщо вірно, червоний, якщо ні
        is_correct = predictions[i] == labels[i]
        color = 'green' if is_correct else 'red'
        
        # Виведення підпису
        title = f"Очікується: {labels[i].item()}\nПрогноз: {predictions[i].item()}"
        ax.set_title(title, color=color, fontsize=10)
        
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Налаштування простору для заголовка
    plt.show()

def main():
    # Завантаження моделі та тестових даних
    model, test_loader = load_model_and_data(N_CLIENTS_USED)

    if model is None:
        return

    print("\n--- Оцінка моделі на тестовому наборі ---")
    
    # 1. Обчислення метрик
    try:
        # Використовуємо функцію test_global з fl_utils
        avg_loss, accuracy = test_global(model, test_loader.dataset)
        
        print("-" * 40)
        print(f"| Фінальна точність (Accuracy): {accuracy:.2f}%")
        print(f"| Середня втрата (Loss): {avg_loss:.4f}")
        print("-" * 40)
        
    except Exception as e:
        print(f"❌ Помилка при обчисленні метрик (перевірте fl_utils.py): {e}")
        return

    # 2. Візуалізація
    print("\n--- Візуалізація прикладів інференсу ---")
    # Додамо тег для візуалізації результатів 
    visualize_inference(model, test_loader, num_examples=8)

if __name__ == '__main__':
    # Встановлюємо matplotlib для візуалізації
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Помилка: Бібліотека Matplotlib не встановлена.")
        print("Встановіть її командою: pip install matplotlib")
        exit()
        
    main()