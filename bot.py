import telebot
from telebot import types
import os
import json
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
from datetime import datetime, timedelta
import time
import threading
import instructor
from openai import OpenAI
from pydantic import BaseModel
from instructor import Mode
from typing import Optional, List, Dict
import cv2
from pathlib import Path
import tensorflow as tf
import sqlite3
from io import BytesIO
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Функция для чтения токена из файла
def get_bot_token():
    """Читает токен бота из файла bot.txt"""
    try:
        possible_paths = [
            'bot.txt',
            'config/bot.txt',
            '../bot.txt',
            os.path.join(os.path.dirname(__file__), 'bot.txt')
        ]

        for path in possible_paths:
            if os.path.exists(path):
                with open(path, 'r') as file:
                    token = file.readline().strip()
                    if token:
                        print(f"✅ Токен бота загружен из файла: {path}")
                        return token

        raise FileNotFoundError("Файл bot.txt не найден ни в одном из ожидаемых мест")

    except Exception as e:
        print(f"❌ Ошибка при чтении токена: {e}")
        print("Создайте файл bot.txt и поместите в него токен вашего бота")
        return None


# Получаем токен
API_TOKEN = get_bot_token()
if not API_TOKEN:
    print("❌ Не удалось получить токен бота. Бот не может быть запущен.")
    exit(1)

# Настройки прокси (если нужно)
PROXY = None
if PROXY:
    telebot.apihelper.proxy = {'https': PROXY}

# Инициализация бота
bot = telebot.TeleBot(API_TOKEN, threaded=True)

# Конфигурация
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
DB_PATH = "skin_analysis.db"
IMG_SIZE = (224, 224)
DNO_CLASSES = {0: 'нормальная кожа', 1: 'жирная кожа', 2: 'сухая кожа'}


# ========== БАЗА ДАННЫХ ==========
def init_db():
    if os.path.exists(DB_PATH):
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT recommendation FROM analysis_history LIMIT 1")
            conn.close()
        except:
            conn.close()
            os.remove(DB_PATH)
            print("Старая БД удалена")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            username TEXT,
            first_name TEXT,
            name TEXT,
            age INTEGER,
            preferences TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            role TEXT,
            content TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analysis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            mode TEXT,
            dno_prediction TEXT DEFAULT '',
            dno_confidence REAL DEFAULT 0,
            acne_detected TEXT DEFAULT '',
            acne_confidence REAL DEFAULT 0,
            recommendation TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detailed_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_id INTEGER,
            zone_name TEXT,
            dno_prediction TEXT DEFAULT '',
            dno_confidence REAL DEFAULT 0,
            acne_detected TEXT DEFAULT '',
            acne_confidence REAL DEFAULT 0,
            FOREIGN KEY (analysis_id) REFERENCES analysis_history (id)
        )
    ''')

    conn.commit()
    conn.close()
    print("✅ База данных готова")


def add_user(user_id, username=None, first_name=None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('INSERT OR IGNORE INTO users (user_id, username, first_name) VALUES (?, ?, ?)',
                   (str(user_id), username, first_name))
    conn.commit()
    conn.close()


def add_message(user_id, role, content):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('INSERT INTO messages (user_id, role, content) VALUES (?, ?, ?)',
                   (str(user_id), role, content))
    cursor.execute('''
        DELETE FROM messages 
        WHERE user_id = ? AND id NOT IN (
            SELECT id FROM messages WHERE user_id = ? ORDER BY created_at DESC LIMIT 50
        )
    ''', (str(user_id), str(user_id)))
    conn.commit()
    conn.close()


def get_history(user_id, limit=30):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT role, content FROM messages WHERE user_id = ? ORDER BY created_at DESC LIMIT ?',
                   (str(user_id), limit))
    messages = cursor.fetchall()
    conn.close()
    return [{'role': m[0], 'content': m[1]} for m in reversed(messages)]


def clear_history(user_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM messages WHERE user_id = ?', (str(user_id),))
    conn.commit()
    conn.close()


def save_analysis(user_id, mode, results, recommendation=""):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    dno_pred = results.get('dno', {}).get('prediction', '') if results.get('dno') else ''
    dno_conf = results.get('dno', {}).get('confidence', 0) if results.get('dno') else 0

    acne_pred = results.get('acne', {}).get('predicted_class', '') if results.get('acne') else ''
    acne_conf = results.get('acne', {}).get('confidence', 0) if results.get('acne') else 0

    cursor.execute('''
        INSERT INTO analysis_history (user_id, mode, dno_prediction, dno_confidence, 
                                    acne_detected, acne_confidence, recommendation)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (str(user_id), mode, dno_pred, dno_conf, acne_pred, acne_conf, recommendation))

    analysis_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return analysis_id


def save_detailed_analysis(analysis_id, zone_name, results):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    dno_pred = results.get('dno', {}).get('prediction', '') if results.get('dno') else ''
    dno_conf = results.get('dno', {}).get('confidence', 0) if results.get('dno') else 0
    acne_pred = results.get('acne', {}).get('predicted_class', '') if results.get('acne') else ''
    acne_conf = results.get('acne', {}).get('confidence', 0) if results.get('acne') else 0

    cursor.execute('''
        INSERT INTO detailed_analysis (analysis_id, zone_name, dno_prediction, dno_confidence, 
                                      acne_detected, acne_confidence)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (analysis_id, zone_name, dno_pred, dno_conf, acne_pred, acne_conf))

    conn.commit()
    conn.close()


def get_user_analyses(user_id, limit=10):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, mode, dno_prediction, dno_confidence, acne_detected, acne_confidence, 
               recommendation, created_at
        FROM analysis_history WHERE user_id = ? ORDER BY created_at DESC LIMIT ?
    ''', (str(user_id), limit))
    analyses = cursor.fetchall()
    conn.close()
    return analyses


# ========== МОДЕЛИ ==========
class DNOClassifier:
    """Классификатор для определения типа кожи (жирная/сухая/нормальная)"""

    def __init__(self):
        self.model = None
        self._load_model()

    def _load_model(self):
        dno_path = os.path.join(MODELS_DIR, "best_model_dno.keras")
        if os.path.exists(dno_path):
            try:
                self.model = tf.keras.models.load_model(dno_path)
                logger.info("✅ Модель DNO (жирность/сухость) загружена")
            except Exception as e:
                logger.error(f"❌ Ошибка загрузки DNO: {e}")
        else:
            logger.warning(f"⚠️ Модель {dno_path} не найдена")

    def predict(self, image_path):
        if self.model is None:
            return None
        try:
            img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMG_SIZE)
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_batch = np.expand_dims(img_array, axis=0)
            img_batch = tf.keras.applications.mobilenet_v2.preprocess_input(img_batch)
            pred = self.model.predict(img_batch, verbose=0)[0]
            class_idx = np.argmax(pred)
            return {
                'prediction': DNO_CLASSES.get(class_idx, 'неизвестно'),
                'confidence': float(pred[class_idx])
            }
        except Exception as e:
            logger.error(f"Ошибка предсказания DNO: {e}")
            return None


class AcneClassifier:
    """PyTorch классификатор для определения наличия прыщей"""

    def __init__(self, model_path=None, classes_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if model_path is None:
            possible_paths = [
                'models/skin_classifier.pth',
                'model/skin_classifier.pth',
                'skin_classifier.pth',
                '../models/skin_classifier.pth',
                '../model/skin_classifier.pth'
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    break

        if model_path is None or not os.path.exists(model_path):
            logger.warning(f"⚠️ PyTorch модель не найдена.")
            self.model = None
            return

        if classes_path is None:
            possible_class_paths = [
                'models/classes.json',
                'model/classes.json',
                'classes.json'
            ]
            for path in possible_class_paths:
                if os.path.exists(path):
                    classes_path = path
                    break

        if classes_path and os.path.exists(classes_path):
            with open(classes_path, 'r') as f:
                self.classes = json.load(f)
        else:
            self.classes = ["Acne", "normal"]

        logger.info(f"📂 Загрузка PyTorch модели из: {model_path}")

        self.model = models.resnet50(weights=None)
        num_features = self.model.fc.in_features

        self.model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, len(self.classes))
        )

        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            logger.info(f"✅ PyTorch модель успешно загружена")
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки PyTorch модели: {e}")
            self.model = None
            return

        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.recommendations = {
            "Acne": [
                "🔹 Используйте средства с салициловой кислотой или бензоил пероксидом",
                "🔹 Не выдавливайте прыщи - это может привести к рубцам",
                "🔹 Регулярно меняйте наволочку (каждые 3-4 дня)",
                "🔹 Сократите потребление сахара и молочных продуктов",
                "🔹 Используйте некомедогенную косметику",
                "🔹 Умывайтесь дважды в день мягким очищающим средством",
                "🔹 Не трогайте лицо руками в течение дня"
            ],
            "normal": [
                "✅ Продолжайте ухаживать за кожей в том же режиме",
                "✅ Используйте увлажняющие средства с SPF защитой",
                "✅ Защищайте кожу от солнца",
                "✅ Пейте достаточно воды (1.5-2 литра в день)",
                "✅ Регулярно очищайте кожу перед сном",
                "✅ Используйте легкие увлажняющие средства"
            ]
        }

    def predict(self, image_path, confidence_threshold=0.7):
        if self.model is None:
            return None

        try:
            img = cv2.imread(image_path)
            if img is None:
                raise FileNotFoundError(f"Не удалось загрузить изображение {image_path}")

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img)
            img_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(img_tensor)
                probs = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probs, 1)

            class_idx = predicted.item()
            predicted_class = self.classes[class_idx]
            confidence_value = confidence.item()

            return {
                'predicted_class': predicted_class,
                'confidence': confidence_value,
                'is_confident': confidence_value >= confidence_threshold,
                'recommendations': self.recommendations.get(predicted_class, self.recommendations["normal"])
            }
        except Exception as e:
            logger.error(f"Ошибка PyTorch предсказания: {e}")
            return None


# Инициализация классификаторов
dno_classifier = DNOClassifier()
acne_classifier = AcneClassifier()


# OpenAI клиент
class UserData(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = None
    preferences: Optional[str] = None


def get_api_key():
    """Читает API ключ OpenAI из файла keys.txt"""
    try:
        possible_paths = [
            'keys.txt',
            'config/keys.txt',
            '../keys.txt',
            os.path.join(os.path.dirname(__file__), 'keys.txt')
        ]

        for path in possible_paths:
            if os.path.exists(path):
                with open(path, 'r') as file:
                    api_key = file.readline().strip()
                    if api_key:
                        print(f"✅ API ключ загружен из файла: {path}")
                        return api_key

        print("❌ Файл keys.txt не найден!")
        return None

    except Exception as e:
        print(f"❌ Ошибка при чтении файла keys.txt: {e}")
        return None


# Инициализация OpenAI клиента
client = None
openai_client = None
API_KEY = get_api_key()
if API_KEY:
    try:
        client = instructor.from_openai(
            OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=API_KEY,
            ),
            mode=Mode.JSON
        )
        openai_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=API_KEY
        )
        print("✅ OpenAI клиент инициализирован")
    except Exception as e:
        print(f"⚠️ Ошибка инициализации OpenAI: {e}")
else:
    print("⚠️ OpenAI клиент не инициализирован.")

# Системный промпт для дерматолога
SYSTEM_PROMPT = """Ты - профессиональный дерматолог с многолетним опытом. Твоя задача - консультировать пациентов по вопросам кожи.

ПРАВИЛА:
1. Отвечай на русском языке, кратко и по делу.
2. Задавай уточняющие вопросы, если информации недостаточно.
3. Учитывай историю диалога. Помни, что говорил пациент раньше.
4. Если пациент описывает проблему - дай конкретные рекомендации.
5. Если пациент спрашивает про средство/компонент - объясни как оно работает.
6. Не ставь диагнозов, если недостаточно данных. Рекомендуй очную консультацию в сложных случаях.
7. Будь вежлив и профессионален.

Твои знания включают:
- Акне и постакне
- Розацеа и покраснения
- Гиперпигментация
- Сухость и обезвоженность
- Жирная кожа и себорея
- Антивозрастной уход
- Косметические ингредиенты и их действие
- Диета и образ жизни при проблемах кожи
- Домашний уход и профессиональные процедуры"""

# Файлы для хранения данных
USERS_FILE = 'users.json'
REMINDERS_FILE = 'reminders.json'

# Создаем директории
os.makedirs('photos', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('config', exist_ok=True)


# Загрузка данных
def load_json(filename):
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {} if filename == USERS_FILE else []
    return {} if filename == USERS_FILE else []


def save_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# Состояния пользователей
user_states = {}
user_temp_data = {}


# Функция проверки заполненности профиля
def is_profile_complete(user_id):
    users = load_json(USERS_FILE)
    if user_id not in users:
        return False
    user = users[user_id]
    return user.get('name') and user.get('age') and user.get('preferences')


# Рекомендации по типу кожи
def get_skin_type_recommendations(skin_type):
    recommendations = {
        'жирная кожа': [
            "🔹 Используйте легкие гелевые текстуры для увлажнения",
            "🔹 Применяйте средства с салициловой кислотой для контроля жирности",
            "🔹 Используйте матирующие салфетки в течение дня",
            "🔹 Не пересушивайте кожу - это приводит к еще большей выработке кожного сала",
            "🔹 Выбирайте средства с пометкой 'некомедогенно'",
            "🔹 Регулярно используйте глиняные маски (1-2 раза в неделю)"
        ],
        'сухая кожа': [
            "🔹 Используйте плотные увлажняющие кремы с керамидами",
            "🔹 Добавьте в уход масла (жожоба, шиповника, аргановое)",
            "🔹 Избегайте агрессивных очищающих средств",
            "🔹 Используйте увлажнитель воздуха в помещении",
            "🔹 Наносите увлажняющие средства на влажную кожу",
            "🔹 Включите в рацион больше омега-3 жирных кислот"
        ],
        'нормальная кожа': [
            "✅ Поддерживайте текущий уход за кожей",
            "✅ Используйте легкие увлажняющие средства",
            "✅ Не забывайте про SPF защиту",
            "✅ Регулярно очищайте кожу утром и вечером",
            "✅ Пейте достаточно воды",
            "✅ Используйте антивозрастные средства с 25 лет для профилактики"
        ]
    }
    return recommendations.get(skin_type, recommendations['нормальная кожа'])


# Клавиатуры
def get_main_keyboard():
    keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=2)
    btn1 = types.KeyboardButton("📸 Анализ лица")
    btn2 = types.KeyboardButton("📸 Анализ по зонам")
    btn3 = types.KeyboardButton("💬 Консультация дерматолога")
    btn4 = types.KeyboardButton("📋 Мои напоминания")
    btn5 = types.KeyboardButton("📊 История анализов")
    btn6 = types.KeyboardButton("👤 Мой профиль")
    keyboard.add(btn1, btn2, btn3, btn4, btn5, btn6)
    return keyboard


def get_consultation_keyboard():
    """Клавиатура для режима консультации"""
    keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=2)
    btn1 = types.KeyboardButton("🆕 Новый диалог")
    btn2 = types.KeyboardButton("❌ Завершить консультацию")
    keyboard.add(btn1, btn2)
    return keyboard


def get_confirmation_keyboard():
    keyboard = types.InlineKeyboardMarkup(row_width=2)
    btn1 = types.InlineKeyboardButton("✅ Подтвердить", callback_data="confirm")
    btn2 = types.InlineKeyboardButton("✏️ Изменить", callback_data="edit")
    btn3 = types.InlineKeyboardButton("❌ Отмена", callback_data="cancel")
    keyboard.add(btn1, btn2, btn3)
    return keyboard


def get_profile_setup_keyboard():
    keyboard = types.InlineKeyboardMarkup()
    btn = types.InlineKeyboardButton("📝 Заполнить профиль", callback_data="setup_profile")
    keyboard.add(btn)
    return keyboard


# Обработка команды /start
@bot.message_handler(commands=['start'])
def send_welcome(message):
    user_id = str(message.from_user.id)
    users = load_json(USERS_FILE)

    # Добавляем пользователя в БД
    add_user(user_id, message.from_user.username, message.from_user.first_name)

    if user_id not in users:
        users[user_id] = {
            'id': user_id,
            'username': message.from_user.username,
            'first_name': message.from_user.first_name,
            'name': None,
            'age': None,
            'preferences': None,
            'created_at': datetime.now().isoformat()
        }
        save_json(users, USERS_FILE)

    if not is_profile_complete(user_id):
        welcome_text = (
            f"👋 Здравствуйте! Я бот для анализа состояния кожи.\n\n"
            f"🎯 *Для начала работы необходимо заполнить профиль.*\n\n"
            f"Давайте настроим его! Как к вам обращаться?"
        )
        bot.send_message(message.chat.id, welcome_text, parse_mode='Markdown')
        user_states[user_id] = 'waiting_name'
    else:
        user = users[user_id]
        bot.send_message(message.chat.id,
                         f"👋 С возвращением, {user['name']}! Используйте кнопки меню для навигации.",
                         reply_markup=get_main_keyboard())


# Обработка текстовых сообщений
@bot.message_handler(func=lambda message: True)
def handle_text(message):
    user_id = str(message.from_user.id)
    users = load_json(USERS_FILE)

    # Проверяем состояние консультации
    if user_id in user_states and user_states[user_id].get('state') == 'consultation':
        if message.text == "❌ Завершить консультацию":
            end_consultation(message)
            return
        elif message.text == "🆕 Новый диалог":
            new_dialog_in_consultation(message)
            return
        elif message.text not in ["📸 Анализ лица", "📸 Анализ по зонам", "💬 Консультация дерматолога",
                                  "📋 Мои напоминания", "📊 История анализов", "👤 Мой профиль"]:
            handle_consultation_message(message)
            return

    # Обработка состояний (заполнение профиля)
    if user_id in user_states:
        state = user_states[user_id]

        if state == 'waiting_name':
            user_temp_data[user_id] = {'name': message.text}
            bot.reply_to(message, f"Приятно познакомиться, {message.text}! Сколько вам лет?")
            user_states[user_id] = 'waiting_age'

        elif state == 'waiting_age':
            try:
                age = int(message.text)
                if age < 1 or age > 120:
                    bot.reply_to(message, "Пожалуйста, введите реальный возраст (1-120 лет).")
                    return
                user_temp_data[user_id]['age'] = age
                bot.reply_to(message,
                             "Что бы вы хотели улучшить в уходе за кожей?\n"
                             "Например: диета, смена постельного белья, уходовые средства и т.д.")
                user_states[user_id] = 'waiting_preferences'
            except ValueError:
                bot.reply_to(message, "Пожалуйста, введите возраст числом.")

        elif state == 'waiting_preferences':
            user_temp_data[user_id]['preferences'] = message.text
            show_profile_summary(user_id, message.chat.id)

        elif state == 'editing':
            handle_editing(user_id, message)

        return

    # Обработка кнопок меню
    if message.text == "📸 Анализ лица":
        if not is_profile_complete(user_id):
            bot.send_message(message.chat.id,
                             "❌ *Профиль не заполнен!*\n\n"
                             "Для анализа лица необходимо заполнить профиль.",
                             parse_mode='Markdown',
                             reply_markup=get_profile_setup_keyboard())
            return

        user_states[user_id] = {'state': 'waiting_photo'}
        bot.reply_to(message, "📸 Отправьте мне фотографию лица для анализа.",
                     reply_markup=types.ReplyKeyboardMarkup(resize_keyboard=True).add(
                         types.KeyboardButton("❌ Отмена")))

    elif message.text == "📸 Анализ по зонам":
        if not is_profile_complete(user_id):
            bot.send_message(message.chat.id,
                             "❌ *Профиль не заполнен!*\n\n"
                             "Для анализа необходимо заполнить профиль.",
                             parse_mode='Markdown',
                             reply_markup=get_profile_setup_keyboard())
            return

        user_states[user_id] = {
            'state': 'detailed',
            'zones': ['Лоб', 'Нос', 'Подбородок', 'Левая щека', 'Правая щека'],
            'current': 0,
            'results': {}
        }
        bot.send_message(message.chat.id, "Сфотографируйте лоб крупным планом.",
                         reply_markup=types.ReplyKeyboardMarkup(resize_keyboard=True).add(
                             types.KeyboardButton("❌ Отмена")))

    elif message.text == "💬 Консультация дерматолога":
        start_consultation(message)

    elif message.text == "📋 Мои напоминания":
        if not is_profile_complete(user_id):
            bot.send_message(message.chat.id,
                             "❌ *Профиль не заполнен!*\n\n"
                             "Для просмотра напоминаний необходимо заполнить профиль.",
                             parse_mode='Markdown',
                             reply_markup=get_profile_setup_keyboard())
            return
        show_reminders(user_id, message.chat.id)

    elif message.text == "📊 История анализов":
        if not is_profile_complete(user_id):
            bot.send_message(message.chat.id,
                             "❌ *Профиль не заполнен!*\n\n"
                             "Для просмотра истории необходимо заполнить профиль.",
                             parse_mode='Markdown',
                             reply_markup=get_profile_setup_keyboard())
            return
        show_history(message)

    elif message.text == "👤 Мой профиль":
        show_profile_settings(user_id, message.chat.id)

    elif message.text == "❌ Отмена":
        if user_id in user_states:
            del user_states[user_id]
        bot.send_message(message.chat.id, "❌ Отменено.", reply_markup=get_main_keyboard())

    else:
        bot.reply_to(message, "Используйте кнопки меню для навигации.",
                     reply_markup=get_main_keyboard() if is_profile_complete(user_id) else None)


def show_profile_summary(user_id, chat_id):
    data = user_temp_data[user_id]
    summary = f"📋 *Проверьте ваши данные:*\n\n"
    summary += f"👤 Имя: {data['name']}\n"
    summary += f"🎂 Возраст: {data['age']}\n"
    summary += f"🎯 Предпочтения: {data['preferences']}\n\n"
    summary += "Всё верно?"

    bot.send_message(chat_id, summary, parse_mode='Markdown',
                     reply_markup=get_confirmation_keyboard())
    user_states[user_id] = 'confirming_profile'


@bot.callback_query_handler(func=lambda call: True)
def handle_callback(call):
    user_id = str(call.from_user.id)

    if call.data == "setup_profile":
        bot.edit_message_text(
            "Давайте заполним профиль! Как к вам обращаться?",
            call.message.chat.id,
            call.message.message_id
        )
        user_states[user_id] = 'waiting_name'

    elif call.data == "confirm":
        if user_states.get(user_id) == 'confirming_profile':
            save_user_profile(user_id, call.message.chat.id)
        elif user_states.get(user_id) == 'confirming_reminder':
            save_reminder(user_id, call.message.chat.id)

    elif call.data == "edit":
        if user_states.get(user_id) == 'confirming_profile':
            bot.edit_message_text(
                "Что хотите изменить? Напишите в формате:\n"
                "имя: [новое имя]\n"
                "возраст: [новый возраст]\n"
                "предпочтения: [новые предпочтения]",
                call.message.chat.id,
                call.message.message_id
            )
            user_states[user_id] = 'editing'

    elif call.data == "cancel":
        bot.edit_message_text("❌ Операция отменена.",
                              call.message.chat.id,
                              call.message.message_id)
        if user_id in user_temp_data:
            del user_temp_data[user_id]
        if user_id in user_states:
            del user_states[user_id]

    elif call.data.startswith("delete_reminder_"):
        reminder_id = call.data.replace("delete_reminder_", "")
        delete_reminder(user_id, reminder_id, call.message.chat.id)

    elif call.data.startswith('hist_'):
        show_analysis_detail(call)

    elif call.data == "close_hist":
        bot.delete_message(call.message.chat.id, call.message.message_id)
        bot.answer_callback_query(call.id)


def save_user_profile(user_id, chat_id):
    users = load_json(USERS_FILE)
    data = user_temp_data[user_id]

    users[user_id]['name'] = data['name']
    users[user_id]['age'] = data['age']
    users[user_id]['preferences'] = data['preferences']

    save_json(users, USERS_FILE)

    add_user(user_id, users[user_id].get('username'), data['name'])

    bot.send_message(chat_id,
                     f"✅ *Профиль сохранен!*\n\n"
                     f"{data['name']}, теперь вы можете использовать все функции бота.",
                     parse_mode='Markdown',
                     reply_markup=get_main_keyboard())

    del user_temp_data[user_id]
    if user_id in user_states:
        del user_states[user_id]


def handle_editing(user_id, message):
    text = message.text.lower()
    data = user_temp_data[user_id]

    if 'имя:' in text:
        data['name'] = text.split('имя:')[1].strip()
    elif 'возраст:' in text:
        try:
            age = int(text.split('возраст:')[1].strip())
            if age < 1 or age > 120:
                bot.reply_to(message, "Пожалуйста, введите реальный возраст (1-120 лет).")
                return
            data['age'] = age
        except:
            bot.reply_to(message, "Неверный формат возраста.")
            return
    elif 'предпочтения:' in text:
        data['preferences'] = text.split('предпочтения:')[1].strip()
    else:
        bot.reply_to(message, "Используйте формат: имя: [новое имя] или возраст: [новый возраст]")
        return

    show_profile_summary(user_id, message.chat.id)


# ========== ОБРАБОТКА ФОТО ==========
@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    user_id = str(message.from_user.id)

    if not is_profile_complete(user_id):
        bot.reply_to(message,
                     "❌ *Профиль не заполнен!*\n\n"
                     "Пожалуйста, сначала заполните профиль с помощью команды /start",
                     parse_mode='Markdown')
        return

    state = user_states.get(user_id, {}).get('state')

    if state == 'waiting_photo':
        process_single_photo(message)
    elif state == 'detailed':
        process_zone_photo(message)
    else:
        bot.send_message(message.chat.id, "Сначала выберите режим анализа.",
                         reply_markup=get_main_keyboard())


def process_single_photo(message):
    user_id = str(message.from_user.id)

    msg = bot.send_message(message.chat.id, "🔍 Анализирую фото...")

    file_info = bot.get_file(message.photo[-1].file_id)
    downloaded = bot.download_file(file_info.file_path)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    photo_path = f"photos/user_{user_id}_{timestamp}.jpg"

    with open(photo_path, 'wb') as f:
        f.write(downloaded)

    try:
        results = {}

        # Анализ типа кожи (жирность/сухость)
        dno = dno_classifier.predict(photo_path)
        results['dno'] = dno if dno else {'prediction': 'неизвестно', 'confidence': 0}

        # Анализ на наличие прыщей (PyTorch)
        acne = acne_classifier.predict(photo_path) if acne_classifier else None
        results['acne'] = acne if acne else {'predicted_class': 'неизвестно', 'confidence': 0, 'is_confident': False,
                                             'recommendations': []}

        # Формируем результат
        text = "📊 *Результаты анализа:*\n\n"

        # Тип кожи
        skin_emoji = {'жирная кожа': '🟡', 'сухая кожа': '🔵', 'нормальная кожа': '🟢'}.get(results['dno']['prediction'],
                                                                                         '⚪')
        text += f"{skin_emoji} *Тип кожи:* {results['dno']['prediction']} (уверенность {results['dno']['confidence']:.0%})\n\n"

        # Наличие прыщей
        if results['acne']['predicted_class'] == 'Acne':
            text += "🔴 *Обнаружены признаки акне*\n"
            text += f"Уверенность: {results['acne']['confidence']:.1%}\n\n"
        elif results['acne']['predicted_class'] == 'normal':
            text += "✅ *Кожа выглядит здоровой*\n"
            text += f"Уверенность: {results['acne']['confidence']:.1%}\n\n"
        else:
            text += "⚠️ *Анализ на акне недоступен*\n\n"

        # Рекомендации по типу кожи
        text += "💧 *Рекомендации по типу кожи:*\n"
        skin_recs = get_skin_type_recommendations(results['dno']['prediction'])
        for rec in skin_recs[:3]:
            text += f"{rec}\n"

        # Рекомендации по акне
        if results['acne']['predicted_class'] == 'Acne' and results['acne'].get('recommendations'):
            text += f"\n💊 *Рекомендации по акне:*\n"
            for rec in results['acne']['recommendations'][:4]:
                text += f"{rec}\n"

        rec_text = "; ".join(skin_recs[:2])
        if results['acne']['predicted_class'] == 'Acne':
            rec_text += "; Обнаружено акне"

        # Сохраняем анализ
        save_analysis(user_id, 'single', results, rec_text)

        # Сохраняем в историю консультации
        analysis_summary = f"[Анализ]: Тип кожи: {results['dno']['prediction']}, Акне: {'обнаружено' if results['acne']['predicted_class'] == 'Acne' else 'не обнаружено'}"
        add_message(user_id, 'system', analysis_summary)

        # Отправляем результат
        bot.send_message(user_id, text, parse_mode='Markdown')

        # Предлагаем напоминания
        create_reminders_suggestions(user_id, message.chat.id, results)

    except Exception as e:
        logger.error(f"Ошибка анализа: {e}")
        bot.send_message(user_id, "❌ Произошла ошибка при анализе фото.",
                         reply_markup=get_main_keyboard())

    finally:
        if os.path.exists(photo_path):
            os.remove(photo_path)
        try:
            bot.delete_message(message.chat.id, msg.message_id)
        except:
            pass
        if user_id in user_states:
            del user_states[user_id]
        bot.send_message(user_id, "Выберите действие:", reply_markup=get_main_keyboard())


def process_zone_photo(message):
    user_id = str(message.from_user.id)
    state = user_states[user_id]
    zone_idx = state['current']
    zone_name = state['zones'][zone_idx]

    msg = bot.send_message(message.chat.id, f"🔍 Анализирую {zone_name.lower()}...")

    file_info = bot.get_file(message.photo[-1].file_id)
    downloaded = bot.download_file(file_info.file_path)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    photo_path = f"photos/user_{user_id}_{zone_idx}_{timestamp}.jpg"

    with open(photo_path, 'wb') as f:
        f.write(downloaded)

    try:
        results = {}

        # Анализ типа кожи
        dno = dno_classifier.predict(photo_path)
        results['dno'] = dno if dno else {'prediction': 'неизвестно', 'confidence': 0}

        # Анализ прыщей
        acne = acne_classifier.predict(photo_path) if acne_classifier else None
        results['acne'] = acne if acne else {'predicted_class': 'неизвестно', 'confidence': 0}

        state['results'][zone_name] = results

        text = f"✅ *{zone_name}:*\n"
        text += f"🟢 Тип кожи: {results['dno']['prediction']}\n"
        text += f"🔴 Акне: {'Обнаружено' if results['acne']['predicted_class'] == 'Acne' else 'Не обнаружено'}\n"

        bot.send_message(message.chat.id, text, parse_mode='Markdown')

    finally:
        if os.path.exists(photo_path):
            os.remove(photo_path)
        try:
            bot.delete_message(message.chat.id, msg.message_id)
        except:
            pass

    zone_idx += 1
    state['current'] = zone_idx

    if zone_idx < len(state['zones']):
        next_zone = state['zones'][zone_idx]
        bot.send_message(message.chat.id, f"Теперь сфотографируйте {next_zone.lower()}.")
    else:
        analysis_id = save_analysis(user_id, 'detailed', {})

        full_text = "📊 *Анализ по зонам:*\n\n"
        for zn in state['zones']:
            zr = state['results'][zn]
            full_text += f"📍 *{zn}:*\n"
            full_text += f"  Тип кожи: {zr['dno']['prediction']}\n"
            full_text += f"  Акне: {'🔴 Обнаружено' if zr['acne']['predicted_class'] == 'Acne' else '✅ Не обнаружено'}\n\n"
            save_detailed_analysis(analysis_id, zn, zr)

        add_message(user_id, 'system', f"[Детальный анализ по зонам]")

        bot.send_message(message.chat.id, full_text, parse_mode='Markdown',
                         reply_markup=get_main_keyboard())
        del user_states[user_id]


def create_reminders_suggestions(user_id, chat_id, results):
    users = load_json(USERS_FILE)
    user = users[user_id]

    suggestions = []

    # Напоминания для жирной кожи
    if results['dno']['prediction'] == 'жирная кожа':
        suggestions.append({
            'type': 'oily_care',
            'text': 'Уход за жирной кожей (очищение и матирование)',
            'time': '09:00'
        })
        suggestions.append({
            'type': 'mask',
            'text': 'Сделать глиняную маску (1-2 раза в неделю)',
            'time': '20:00'
        })

    # Напоминания для сухой кожи
    if results['dno']['prediction'] == 'сухая кожа':
        suggestions.append({
            'type': 'dry_care',
            'text': 'Нанести увлажняющий крем с керамидами',
            'time': '09:00'
        })
        suggestions.append({
            'type': 'humidity',
            'text': 'Проверить влажность в помещении',
            'time': '12:00'
        })

    # Напоминания при акне
    if results.get('acne', {}).get('predicted_class') == 'Acne':
        suggestions.append({
            'type': 'acne_care',
            'text': 'Обработать проблемные участки',
            'time': '21:00'
        })
        suggestions.append({
            'type': 'bedding',
            'text': 'Проверить чистоту наволочки',
            'time': '20:00'
        })

    if user.get('preferences'):
        if 'диет' in user['preferences'].lower():
            suggestions.append({
                'type': 'water',
                'text': 'Выпить стакан воды',
                'time': '11:00'
            })

    if suggestions:
        response = "📅 *Предлагаемые напоминания:*\n\n"
        for i, sug in enumerate(suggestions[:5], 1):
            response += f"{i}. {sug['text']} - ежедневно в {sug['time']}\n"

        response += "\nХотите создать эти напоминания?"

        user_temp_data[user_id] = {'suggestions': suggestions[:5]}
        user_states[user_id] = 'confirming_reminder'

        bot.send_message(chat_id, response, parse_mode='Markdown',
                         reply_markup=get_confirmation_keyboard())


def save_reminder(user_id, chat_id):
    reminders = load_json(REMINDERS_FILE)
    suggestions = user_temp_data[user_id]['suggestions']

    for sug in suggestions:
        reminder = {
            'id': f"{user_id}_{datetime.now().timestamp()}_{sug['type']}",
            'user_id': user_id,
            'text': sug['text'],
            'time': sug['time'],
            'created_at': datetime.now().isoformat(),
            'active': True
        }
        reminders.append(reminder)

    save_json(reminders, REMINDERS_FILE)
    bot.send_message(chat_id, "✅ Напоминания созданы!")

    if user_id in user_temp_data:
        del user_temp_data[user_id]
    if user_id in user_states:
        del user_states[user_id]


def show_reminders(user_id, chat_id):
    reminders = load_json(REMINDERS_FILE)
    user_reminders = [r for r in reminders if r['user_id'] == user_id and r.get('active', True)]

    if not user_reminders:
        bot.send_message(chat_id, "У вас пока нет активных напоминаний.")
        return

    response = "📋 *Ваши напоминания:*\n\n"
    keyboard = types.InlineKeyboardMarkup(row_width=1)

    for reminder in user_reminders:
        response += f"🕐 {reminder['time']} - {reminder['text']}\n"
        btn = types.InlineKeyboardButton(
            f"❌ Удалить: {reminder['text'][:30]}...",
            callback_data=f"delete_reminder_{reminder['id']}"
        )
        keyboard.add(btn)

    bot.send_message(chat_id, response, parse_mode='Markdown', reply_markup=keyboard)


def delete_reminder(user_id, reminder_id, chat_id):
    reminders = load_json(REMINDERS_FILE)

    for reminder in reminders:
        if reminder['id'] == reminder_id and reminder['user_id'] == user_id:
            reminder['active'] = False
            break

    save_json(reminders, REMINDERS_FILE)
    bot.send_message(chat_id, "✅ Напоминание удалено.")
    show_reminders(user_id, chat_id)


def show_history(message):
    user_id = str(message.from_user.id)
    analyses = get_user_analyses(user_id)

    if not analyses:
        bot.send_message(message.chat.id, "История анализов пуста.", reply_markup=get_main_keyboard())
        return

    markup = types.InlineKeyboardMarkup(row_width=1)
    for a in analyses[:10]:
        a_id, mode, dno_pred, dno_conf, acne_pred, acne_conf, rec, created = a
        date = created[:10] if created else "?"
        btn_text = f"{date} | {mode} | {dno_pred}"
        markup.add(types.InlineKeyboardButton(btn_text, callback_data=f"hist_{a_id}"))

    markup.add(types.InlineKeyboardButton("❌ Закрыть", callback_data="close_hist"))
    bot.send_message(message.chat.id, "📊 История анализов:", reply_markup=markup)


def show_analysis_detail(call):
    a_id = int(call.data.split('_')[1])
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM analysis_history WHERE id = ?', (a_id,))
    a = cursor.fetchone()

    if a:
        text = f"📊 Анализ от {a[8]}\n\n" if len(a) > 8 and a[8] else "📊 Анализ:\n\n"
        if a[2]:
            text += f"🟢 Тип кожи: {a[2]} ({a[3]:.0%})\n"
        if a[4]:
            text += f"🔴 Акне: {a[4]} ({a[5]:.0%})\n"
        if a[6]:
            text += f"\n💡 Рекомендации:\n{a[6]}"

        if a[1] == 'detailed':
            cursor.execute(
                'SELECT zone_name, dno_prediction, dno_confidence, acne_detected, acne_confidence FROM detailed_analysis WHERE analysis_id = ?',
                (a_id,))
            zones = cursor.fetchall()
            if zones:
                text += "\n\n📍 По зонам:\n"
                for z in zones:
                    text += f"\n{z[0]}:\n"
                    text += f"  Тип кожи: {z[1]}\n"
                    text += f"  Акне: {z[3]}\n"

        bot.send_message(call.message.chat.id, text, parse_mode='Markdown')
    conn.close()
    bot.answer_callback_query(call.id)


def show_profile_settings(user_id, chat_id):
    users = load_json(USERS_FILE)
    user = users.get(user_id, {})

    if not is_profile_complete(user_id):
        response = "⚠️ *Профиль не заполнен!*\n\n"
        response += f"👤 Имя: {user.get('name', 'Не указано')}\n"
        response += f"🎂 Возраст: {user.get('age', 'Не указано')}\n"
        response += f"🎯 Предпочтения: {user.get('preferences', 'Не указано')}\n\n"
        response += "Для заполнения профиля используйте команду /start"
    else:
        response = "⚙️ *Ваш профиль:*\n\n"
        response += f"👤 Имя: {user.get('name')}\n"
        response += f"🎂 Возраст: {user.get('age')}\n"
        response += f"🎯 Предпочтения: {user.get('preferences')}\n\n"
        response += "Для изменения данных используйте команду /start"

    bot.send_message(chat_id, response, parse_mode='Markdown')


# ========== КОНСУЛЬТАЦИЯ ЧЕРЕЗ ИИ ==========
def start_consultation(message):
    user_id = str(message.from_user.id)

    if not is_profile_complete(user_id):
        bot.send_message(message.chat.id,
                         "❌ *Профиль не заполнен!*\n\n"
                         "Для консультации необходимо заполнить профиль.",
                         parse_mode='Markdown',
                         reply_markup=get_profile_setup_keyboard())
        return

    user_states[user_id] = {'state': 'consultation'}

    if not openai_client:
        bot.send_message(
            message.chat.id,
            "⚠️ Консультация с ИИ временно недоступна (нет API ключа).\n"
            "Но вы можете сделать анализ фото!",
            reply_markup=get_main_keyboard()
        )
        return

    history = get_history(user_id)

    if history:
        bot.send_message(
            message.chat.id,
            "👨‍⚕️ Продолжаем консультацию. Я помню наш диалог.\n\n"
            "Задайте вопрос или расскажите, что беспокоит.\n\n"
            "Для выхода из режима консультации нажмите кнопку ниже.",
            reply_markup=get_consultation_keyboard()
        )
    else:
        add_message(user_id, 'assistant', "Здравствуйте! Я дерматолог. Расскажите, что вас беспокоит?")
        bot.send_message(
            message.chat.id,
            "👨‍⚕️ Здравствуйте! Я дерматолог с многолетним опытом.\n\n"
            "Расскажите о вашей коже:\n"
            "- Какая проблема беспокоит?\n"
            "- Как давно она появилась?\n"
            "- Какой у вас тип кожи?\n"
            "- Чем сейчас пользуетесь?\n\n"
            "Я слушаю 🩺\n\n"
            "Для выхода из режима консультации нажмите кнопку ниже.",
            reply_markup=get_consultation_keyboard()
        )


def handle_consultation_message(message):
    user_id = str(message.from_user.id)
    text = message.text.strip()

    if not openai_client:
        bot.send_message(message.chat.id, "⚠️ Консультация недоступна без API ключа.",
                         reply_markup=get_main_keyboard())
        return

    # Сохраняем сообщение пользователя
    add_message(user_id, 'user', text)

    # Получаем историю
    history = get_history(user_id)

    # Отправляем в нейросеть
    bot.send_chat_action(message.chat.id, 'typing')

    try:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend([{"role": m['role'], "content": m['content']} for m in history])

        response = openai_client.chat.completions.create(
            model="openai/gpt-3.5-turbo",
            messages=messages,
            max_tokens=1000,
            temperature=0.7
        )

        reply = response.choices[0].message.content.strip()

        # Сохраняем ответ
        add_message(user_id, 'assistant', reply)

        if len(reply) > 4000:
            for i in range(0, len(reply), 4000):
                bot.send_message(message.chat.id, reply[i:i + 4000])
        else:
            bot.send_message(message.chat.id, reply, reply_markup=get_consultation_keyboard())

    except Exception as e:
        logger.error(f"Ошибка нейросети: {e}")
        bot.send_message(message.chat.id, "❌ Произошла ошибка при обращении к ИИ. Попробуйте позже.",
                         reply_markup=get_consultation_keyboard())


def end_consultation(message):
    """Завершение консультации и возврат в главное меню"""
    user_id = str(message.from_user.id)

    # Сохраняем сообщение о завершении консультации
    add_message(user_id, 'system', 'Консультация завершена пользователем')

    if user_id in user_states:
        del user_states[user_id]

    bot.send_message(
        message.chat.id,
        "✅ Консультация завершена. Вы вернулись в главное меню.\n\n"
        "История диалога сохранена. Чтобы начать новый диалог, "
        "нажмите кнопку 'Новый диалог' в режиме консультации.",
        reply_markup=get_main_keyboard()
    )


def new_dialog_in_consultation(message):
    """Начать новый диалог внутри режима консультации"""
    user_id = str(message.from_user.id)

    # Очищаем историю
    clear_history(user_id)

    # Добавляем приветственное сообщение
    add_message(user_id, 'assistant', "Здравствуйте! Я дерматолог. Расскажите, что вас беспокоит?")

    bot.send_message(
        message.chat.id,
        "🆕 Начинаем новый диалог.\n\n"
        "👨‍⚕️ Здравствуйте! Расскажите о вашей коже:\n"
        "- Какая проблема беспокоит?\n"
        "- Как давно она появилась?\n"
        "- Какой у вас тип кожи?\n"
        "- Чем сейчас пользуетесь?\n\n"
        "Я слушаю 🩺",
        reply_markup=get_consultation_keyboard()
    )


# Функция для проверки и отправки напоминаний
def check_reminders():
    while True:
        try:
            current_time = datetime.now().strftime("%H:%M")
            reminders = load_json(REMINDERS_FILE)

            for reminder in reminders:
                if reminder.get('active', True) and reminder['time'] == current_time:
                    try:
                        send_reminder(reminder)
                    except Exception as e:
                        print(f"Ошибка отправки напоминания: {e}")
        except Exception as e:
            print(f"Ошибка в планировщике: {e}")

        time.sleep(60)


def send_reminder(reminder):
    try:
        users = load_json(USERS_FILE)
        user = users.get(reminder['user_id'], {})
        name = user.get('name', 'Пользователь')

        message = f"🔔 *Напоминание для {name}!*\n\n"
        message += f"📝 {reminder['text']}\n"
        message += f"🕐 Время: {datetime.now().strftime('%H:%M')}"

        bot.send_message(reminder['user_id'], message, parse_mode='Markdown')
    except Exception as e:
        print(f"Ошибка отправки напоминания пользователю {reminder['user_id']}: {e}")


# Запуск планировщика напоминаний
reminder_thread = threading.Thread(target=check_reminders)
reminder_thread.daemon = True
reminder_thread.start()

# Запуск бота
if __name__ == "__main__":
    print("=" * 60)
    print("🤖 Бот для анализа кожи запускается...")
    print("=" * 60)

    # Инициализируем базу данных
    init_db()

    # Проверяем наличие необходимых файлов
    print("\n📁 Проверка файлов:")
    print(f"   - bot.txt: {'✅' if os.path.exists('bot.txt') else '❌'}")
    print(f"   - keys.txt: {'✅' if os.path.exists('keys.txt') else '❌'}")

    # Проверяем модели
    dno_model_path = os.path.join(MODELS_DIR, "best_model_dno.keras")
    print(f"   - best_model_dno.keras: {'✅' if os.path.exists(dno_model_path) else '❌'}")
    print(f"   - skin_classifier.pth: {'✅' if acne_classifier and acne_classifier.model else '❌'}")

    if dno_classifier.model:
        print("✅ Модель определения типа кожи (жирность/сухость) загружена")
    else:
        print("⚠️ Модель типа кожи не загружена")

    if acne_classifier and acne_classifier.model:
        print("✅ Модель определения акне загружена")
    else:
        print("⚠️ Модель акне не загружена")

    if openai_client:
        print("✅ Нейросеть для консультации подключена")
    else:
        print("⚠️ Нейросеть не подключена (нет keys.txt)")

    print("\n🚀 Запуск бота...")
    print("Для остановки нажмите Ctrl+C")
    print("=" * 60)

    while True:
        try:
            bot.infinity_polling(timeout=30, long_polling_timeout=30)
        except Exception as e:
            print(f"❌ Ошибка подключения: {e}")
            print("🔄 Переподключение через 5 секунд...")
            time.sleep(5)