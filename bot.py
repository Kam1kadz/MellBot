# === CONFIG ===
import os
import logging
import asyncio
import tempfile
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager
from datetime import datetime

from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, F
from aiogram.types import (
    InlineQuery,
    InlineQueryResultCachedVideo,
    InlineQueryResultArticle,
    InputTextMessageContent,
    Message,
    CallbackQuery,
    FSInputFile,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
)
from aiogram.filters import Command
from aiogram.enums import ParseMode
import aiosqlite
from faster_whisper import WhisperModel
from moviepy.editor import VideoFileClip, CompositeVideoClip, ImageClip
from PIL import Image, ImageDraw, ImageFont

# Загрузка переменных окружения
load_dotenv()

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Конфигурация
BOT_TOKEN = os.getenv("BOT_TOKEN")
ADMIN_IDS = [int(x.strip()) for x in os.getenv("ADMIN_IDS", "").split(",") if x.strip()]
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "tiny")
DB_PATH = os.getenv("DB_PATH", "bot.db")

if not BOT_TOKEN:
    raise ValueError("BOT_TOKEN не найден в .env файле!")

if not ADMIN_IDS:
    logger.warning("ADMIN_IDS не указаны! Админ-команды будут недоступны.")

# Инициализация бота и диспетчера
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()


# === DATABASE ===
class Database:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path

    @asynccontextmanager
    async def _get_db(self):
        """Контекстный менеджер для подключения к БД."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            yield db

    async def init_db(self):
        """Инициализация таблиц базы данных."""
        async with self._get_db() as db:
            # Таблица видео
            await db.execute("""
                CREATE TABLE IF NOT EXISTS videos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_id TEXT UNIQUE NOT NULL,
                    title TEXT NOT NULL,
                    aliases TEXT DEFAULT '',
                    transcript TEXT DEFAULT '',
                    is_active BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Таблица пользователей
            await db.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY,
                    username TEXT,
                    is_banned BOOLEAN DEFAULT 0,
                    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Таблица статистики запросов
            await db.execute("""
                CREATE TABLE IF NOT EXISTS query_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    query TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await db.commit()
        logger.info("База данных инициализирована")

    async def add_video(self, file_id: str, title: str, aliases: str = "", transcript: str = "") -> int:
        """Добавление нового видео в базу."""
        async with self._get_db() as db:
            cursor = await db.execute(
                "INSERT INTO videos (file_id, title, aliases, transcript) VALUES (?, ?, ?, ?)",
                (file_id, title, aliases, transcript),
            )
            await db.commit()
            return cursor.lastrowid

    async def get_video(self, video_id: int) -> Optional[dict]:
        """Получение видео по ID."""
        async with self._get_db() as db:
            cursor = await db.execute(
                "SELECT * FROM videos WHERE id = ? AND is_active = 1",
                (video_id,),
            )
            row = await cursor.fetchone()
            return dict(row) if row else None

    async def search_videos(self, query: str, limit: int = 10) -> List[dict]:
        """Поиск видео по aliases и transcript."""
        search_pattern = f"%{query}%"
        async with self._get_db() as db:
            cursor = await db.execute(
                """
                SELECT * FROM videos 
                WHERE is_active = 1 
                AND (aliases LIKE ? OR transcript LIKE ? OR title LIKE ?)
                LIMIT ?
                """,
                (search_pattern, search_pattern, search_pattern, limit),
            )
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def delete_video(self, video_id: int) -> bool:
        """Мягкое удаление видео (установка is_active = 0)."""
        async with self._get_db() as db:
            cursor = await db.execute(
                "UPDATE videos SET is_active = 0 WHERE id = ?",
                (video_id,),
            )
            await db.commit()
            return cursor.rowcount > 0

    async def list_videos(self) -> List[dict]:
        """Получение списка всех активных видео."""
        async with self._get_db() as db:
            cursor = await db.execute(
                "SELECT id, title, aliases, file_id FROM videos WHERE is_active = 1 ORDER BY id"
            )
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def update_video_aliases(self, video_id: int, aliases: str):
        """Обновление aliases видео."""
        async with self._get_db() as db:
            await db.execute(
                "UPDATE videos SET aliases = ? WHERE id = ?",
                (aliases, video_id),
            )
            await db.commit()

    async def update_video_transcript(self, video_id: int, transcript: str):
        """Обновление transcript видео."""
        async with self._get_db() as db:
            await db.execute(
                "UPDATE videos SET transcript = ? WHERE id = ?",
                (transcript, video_id),
            )
            await db.commit()

    async def get_or_create_user(self, user_id: int, username: str = None):
        """Получение или создание пользователя."""
        async with self._get_db() as db:
            cursor = await db.execute(
                "SELECT * FROM users WHERE user_id = ?",
                (user_id,),
            )
            row = await cursor.fetchone()
            if row:
                await db.execute(
                    "UPDATE users SET last_activity = ?, username = COALESCE(?, username) WHERE user_id = ?",
                    (datetime.now(), username, user_id),
                )
                await db.commit()
                return dict(row)
            else:
                await db.execute(
                    "INSERT INTO users (user_id, username) VALUES (?, ?)",
                    (user_id, username),
                )
                await db.commit()
                return {"user_id": user_id, "username": username, "is_banned": 0}

    async def is_user_banned(self, user_id: int) -> bool:
        """Проверка, забанен ли пользователь."""
        async with self._get_db() as db:
            cursor = await db.execute(
                "SELECT is_banned FROM users WHERE user_id = ?",
                (user_id,),
            )
            row = await cursor.fetchone()
            return row and row["is_banned"]

    async def ban_user(self, user_id: int) -> bool:
        """Бан пользователя."""
        async with self._get_db() as db:
            await db.execute(
                "INSERT OR REPLACE INTO users (user_id, is_banned) VALUES (?, 1)",
                (user_id,),
            )
            await db.commit()
            return True

    async def unban_user(self, user_id: int) -> bool:
        """Разбан пользователя."""
        async with self._get_db() as db:
            await db.execute(
                "UPDATE users SET is_banned = 0 WHERE user_id = ?",
                (user_id,),
            )
            await db.commit()
            return True

    async def log_query(self, user_id: int, query: str):
        """Логирование поискового запроса."""
        async with self._get_db() as db:
            await db.execute(
                "INSERT INTO query_stats (user_id, query) VALUES (?, ?)",
                (user_id, query),
            )
            await db.commit()

    async def get_stats(self) -> dict:
        """Получение статистики."""
        async with self._get_db() as db:
            cursor = await db.execute("SELECT COUNT(*) as count FROM videos WHERE is_active = 1")
            videos_count = (await cursor.fetchone())["count"]

            cursor = await db.execute("SELECT COUNT(*) as count FROM users")
            users_count = (await cursor.fetchone())["count"]

            cursor = await db.execute("SELECT COUNT(*) as count FROM query_stats")
            queries_count = (await cursor.fetchone())["count"]

            return {
                "videos": videos_count,
                "users": users_count,
                "queries": queries_count,
            }


db = Database()


# === TRANSCRIPTION ===
class Transcriber:
    def __init__(self, model_name: str = WHISPER_MODEL):
        self.model = None
        self.model_name = model_name

    def _load_model(self):
        """Ленивая загрузка модели Whisper."""
        if self.model is None:
            logger.info(f"Загрузка модели Whisper: {self.model_name}")
            self.model = WhisperModel(self.model_name, device="cpu", compute_type="int8")
        return self.model

    async def transcribe(self, audio_path: str) -> str:
        """Транскрибация аудио/видео файла."""
        loop = asyncio.get_event_loop()
        model = self._load_model()

        # Запускаем синхронную транскрибацию в отдельном потоке
        def _transcribe():
            segments, _ = model.transcribe(audio_path, beam_size=5, language="ru")
            return " ".join([segment.text for segment in segments])

        try:
            transcript = await loop.run_in_executor(None, _transcribe)
            return transcript.strip()
        except Exception as e:
            logger.error(f"Ошибка транскрибации: {e}")
            return ""


transcriber = Transcriber()


# === VIDEO OVERLAY ===
class VideoOverlay:
    # Цвета как в тёмной теме Telegram
    BUBBLE_COLOR = (43, 47, 58, 240)  # Тёмно-серый фон пузырька
    TEXT_COLOR = (255, 255, 255, 255)  # Белый текст
    NAME_COLOR = (135, 181, 226, 255)  # Синий цвет имени
    TIME_COLOR = (170, 170, 170, 255)  # Серый цвет времени

    @staticmethod
    def round_rectangle(draw, xy, radius, fill):
        """Рисование прямоугольника с закруглёнными углами."""
        x1, y1, x2, y2 = xy
        r = radius

        # Основной прямоугольник
        draw.rectangle([x1 + r, y1, x2 - r, y2], fill=fill)
        draw.rectangle([x1, y1 + r, x2, y2 - r], fill=fill)

        # Четыре угла
        draw.pieslice([x1, y1, x1 + r * 2, y1 + r * 2], 180, 270, fill=fill)
        draw.pieslice([x2 - r * 2, y1, x2, y1 + r * 2], 270, 360, fill=fill)
        draw.pieslice([x1, y2 - r * 2, x1 + r * 2, y2], 90, 180, fill=fill)
        draw.pieslice([x2 - r * 2, y2 - r * 2, x2, y2], 0, 90, fill=fill)

    @staticmethod
    def create_message_bubble(text: str, video_width: int, sender_name: str = None) -> Image.Image:
        """Создание изображения пузырька сообщения Telegram для наложения на видео."""
        # Шрифты
        name_font_size = 28
        text_font_size = 32
        time_font_size = 20

        try:
            name_font = ImageFont.truetype("arial.ttf", name_font_size)
        except:
            try:
                name_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", name_font_size)
            except:
                name_font = ImageFont.load_default()

        try:
            text_font = ImageFont.truetype("arial.ttf", text_font_size)
        except:
            try:
                text_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", text_font_size)
            except:
                text_font = ImageFont.load_default()

        try:
            time_font = ImageFont.truetype("arial.ttf", time_font_size)
        except:
            time_font = ImageFont.load_default()

        # Ширина пузырька - 70% от ширины видео
        bubble_width = int(video_width * 0.7)
        margin = 20
        bubble_margin = 40  # Отступ от краёв видео

        # Перенос текста
        max_text_width = bubble_width - margin * 2
        words = text.split()
        lines = []
        current_line = []

        for word in words:
            test_line = " ".join(current_line + [word])
            bbox = text_font.getbbox(test_line)
            text_width = bbox[2] - bbox[0] if bbox else 0

            if text_width <= max_text_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]

        if current_line:
            lines.append(" ".join(current_line))

        if not lines:
            lines = [text]

        # Вычисляем высоту пузырька
        line_height = text_font_size + 6
        name_height = name_font_size + 8 if sender_name else 0
        time_height = time_font_size + 10
        total_text_height = len(lines) * line_height + name_height + time_height + margin * 2

        # Минимальная ширина пузырька по содержимому
        max_line_width = 0
        for line in lines:
            bbox = text_font.getbbox(line)
            w = bbox[2] - bbox[0] if bbox else 0
            max_line_width = max(max_line_width, w)

        # Пузырёк не шире чем нужно
        actual_bubble_width = min(bubble_width, max_line_width + margin * 3)
        actual_bubble_width = max(actual_bubble_width, 200)  # Минимум 200px

        # Размеры всего изображения (с отступом сверху)
        img_width = video_width
        img_height = total_text_height + bubble_margin + 30

        # Создаём изображение с прозрачным фоном
        img = Image.new("RGBA", (img_width, img_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Координаты пузырька (по центру, с отступом сверху)
        bubble_x1 = (video_width - actual_bubble_width) // 2
        bubble_y1 = bubble_margin
        bubble_x2 = bubble_x1 + actual_bubble_width
        bubble_y2 = bubble_y1 + total_text_height

        # Рисуем тень
        shadow_offset = 4
        VideoOverlay.round_rectangle(
            draw,
            [bubble_x1 + shadow_offset, bubble_y1 + shadow_offset,
             bubble_x2 + shadow_offset, bubble_y2 + shadow_offset],
            radius=18,
            fill=(0, 0, 0, 100)
        )

        # Рисуем пузырёк
        VideoOverlay.round_rectangle(
            draw,
            [bubble_x1, bubble_y1, bubble_x2, bubble_y2],
            radius=18,
            fill=VideoOverlay.BUBBLE_COLOR
        )

        # Рисуем имя отправителя
        y_offset = bubble_y1 + margin
        if sender_name:
            draw.text((bubble_x1 + margin, y_offset), sender_name,
                     font=name_font, fill=VideoOverlay.NAME_COLOR)
            y_offset += name_height

        # Рисуем текст сообщения
        for line in lines:
            draw.text((bubble_x1 + margin, y_offset), line,
                     font=text_font, fill=VideoOverlay.TEXT_COLOR)
            y_offset += line_height

        # Рисуем время (в правом нижнем углу пузырька)
        time_text = "1:38"  # Можно заменить на реальное время
        bbox = time_font.getbbox(time_text)
        time_width = bbox[2] - bbox[0] if bbox else 30
        draw.text((bubble_x2 - margin - time_width, bubble_y2 - margin - time_font_size),
                 time_text, font=time_font, fill=VideoOverlay.TIME_COLOR)

        return img

    @staticmethod
    async def add_message_overlay(video_path: str, text: str, output_path: str, sender_name: str = None):
        """Добавление пузырька сообщения на верхнюю часть видео."""
        loop = asyncio.get_event_loop()

        def _process():
            # Загружаем видео
            video = VideoFileClip(video_path)

            # Создаём изображение пузырька
            bubble_img = VideoOverlay.create_message_bubble(
                text, video.w, sender_name
            )

            # Сохраняем во временный файл
            temp_img_path = tempfile.mktemp(suffix=".png")
            bubble_img.save(temp_img_path)

            # Создаём клип из изображения
            bubble_clip = (
                ImageClip(temp_img_path)
                .set_duration(video.duration)
                .set_position(("center", "top"))
            )

            # Комбинируем видео и пузырёк
            final = CompositeVideoClip([video, bubble_clip], size=video.size)

            # Сохраняем результат
            final.write_videofile(
                output_path,
                codec="libx264",
                audio_codec="aac",
                temp_audiofile=tempfile.mktemp(suffix=".m4a"),
                remove_temp=True,
                threads=2,
                preset="ultrafast",
                logger=None,
            )

            # Очистка
            video.close()
            final.close()
            bubble_clip.close()
            Path(temp_img_path).unlink(missing_ok=True)

        await loop.run_in_executor(None, _process)


video_overlay = VideoOverlay()


# === INLINE HANDLER ===
@dp.inline_query()
async def inline_handler(inline_query: InlineQuery):
    """Обработчик inline запросов."""
    user_id = inline_query.from_user.id
    query_text = inline_query.query.strip()

    # Проверяем бан
    if await db.is_user_banned(user_id):
        await inline_query.answer([], cache_time=1)
        return

    # Обновляем/создаем пользователя
    await db.get_or_create_user(user_id, inline_query.from_user.username)

    # Логируем запрос
    if query_text:
        await db.log_query(user_id, query_text)

    # Поиск видео
    if query_text:
        videos = await db.search_videos(query_text, limit=10)
    else:
        # Если запрос пустой, показываем последние видео
        async with db._get_db() as conn:
            cursor = await conn.execute(
                "SELECT * FROM videos WHERE is_active = 1 ORDER BY id DESC LIMIT 10"
            )
            rows = await cursor.fetchall()
            videos = [dict(row) for row in rows]

    # Формируем результаты
    results = []

    # При пустом запросе добавляем инструкцию первым пунктом
    if not query_text:
        help_result = InlineQueryResultArticle(
            id="help",
            title="📖 Как использовать бота?",
            description="Нажмите сюда для инструкции",
            input_message_content=InputTextMessageContent(
                message_text=(
                    "📖 <b>Как использовать @mellbot:</b>\n\n"
                    "1️⃣ Введите <code>@mellbot &lt;поиск&gt;</code> в любом чате\n"
                    "2️⃣ Выберите видео из списка\n"
                    "3️⃣ Для оверлея с текстом — используйте /overlay в личке\n\n"
                    "<i>Пример: @mellbot мотик</i>"
                ),
                parse_mode=ParseMode.HTML,
            ),
        )
        results.append(help_result)

    for video in videos:
        result = InlineQueryResultCachedVideo(
            id=str(video["id"]),
            video_file_id=video["file_id"],
            title=video["title"],
            description=video["aliases"][:100] if video["aliases"] else None,
        )
        results.append(result)

    # Сохраняем контекст запроса (для возможного оверлея)
    # В aiogram 3 inline_query.from_user доступен, но reply_to_message нет
    # Пользователь должен использовать специальный синтаксис или кнопку
    inline_context[inline_query.id] = {
        "user_id": user_id,
        "query": query_text,
        "timestamp": datetime.now(),
    }

    await inline_query.answer(results, cache_time=10, is_personal=True)


@dp.chosen_inline_result()
async def chosen_inline_handler(chosen_result):
    """Обработчик выбранного inline результата."""
    # Здесь можно добавить логику при выборе результата
    pass


# Словарь для отслеживания контекста inline запросов
# Ключ: query_id, Значение: {user_id, reply_message_text, reply_user_name, chat_type}
inline_context = {}


@dp.message(F.video, F.reply_to_message)
async def handle_video_with_overlay(message: Message):
    """
    Обработка видео, отправленного в чат.
    Если видео было отправлено как ответ на сообщение - накладываем текст.
    """
    # Проверяем, есть ли текст в сообщении-оригинале
    if not message.reply_to_message.text:
        return  # Не ответ на текстовое сообщение - игнорируем

    # Проверяем, что видео отправлено через бота (по наличию в нашей базе)
    video_file_id = message.video.file_id

    # Ищем видео в базе по file_id
    async with db._get_db() as conn:
        cursor = await conn.execute(
            "SELECT * FROM videos WHERE file_id = ? AND is_active = 1",
            (video_file_id,),
        )
        video = await cursor.fetchone()

    if not video:
        return  # Видео не из нашей базы - игнорируем

    # Получаем текст сообщения, на которое отвечают
    overlay_text = message.reply_to_message.text

    if not overlay_text:
        return

    # Получаем имя отправителя исходного сообщения
    reply_user = message.reply_to_message.from_user
    sender_name = reply_user.full_name if reply_user else None

    # Показываем статус "обработка"
    processing_msg = await message.answer("🎬 Накладываю сообщение на видео...")

    try:
        # Скачиваем оригинальное видео
        file = await bot.get_file(video_file_id)
        temp_dir = tempfile.mkdtemp()
        original_path = Path(temp_dir) / "original.mp4"
        output_path = Path(temp_dir) / "output.mp4"

        await bot.download_file(file.file_path, original_path)

        # Накладываем пузырёк сообщения
        await video_overlay.add_message_overlay(
            str(original_path),
            overlay_text,
            str(output_path),
            sender_name=sender_name,
        )

        # Отправляем новое видео
        video_file = FSInputFile(str(output_path))
        await message.reply_video(
            video=video_file,
            caption=f"📝 Текст: {overlay_text[:100]}{'...' if len(overlay_text) > 100 else ''}",
        )

        # Удаляем оригинальное сообщение с видео (опционально)
        # await message.delete()

    except Exception as e:
        logger.error(f"Ошибка при наложении текста: {e}")
        await message.answer(f"❌ Ошибка обработки видео: {e}")

    finally:
        # Очистка
        await processing_msg.delete()
        if 'original_path' in locals():
            original_path.unlink(missing_ok=True)
        if 'output_path' in locals():
            output_path.unlink(missing_ok=True)
        if 'temp_dir' in locals():
            Path(temp_dir).rmdir()


# === ADMIN HANDLERS ===
# Состояния для загрузки видео
upload_states = {}


@dp.message(Command("upload"))
async def cmd_upload(message: Message):
    """Начало процесса загрузки видео."""
    if message.from_user.id not in ADMIN_IDS:
        await message.answer("⛔ У вас нет прав для использования этой команды.")
        return

    user_id = message.from_user.id
    upload_states[user_id] = {"step": "waiting_video"}

    await message.answer(
        "📹 Отправьте мне видео для загрузки.\n"
        "Поддерживаются видео-файлы и кружочки (video notes)."
    )


@dp.message(Command("delete"))
async def cmd_delete(message: Message):
    """Удаление видео по ID."""
    if message.from_user.id not in ADMIN_IDS:
        await message.answer("⛔ У вас нет прав для использования этой команды.")
        return

    args = message.text.split()[1:]
    if not args:
        await message.answer("❌ Укажите ID видео: /delete <id>")
        return

    try:
        video_id = int(args[0])
    except ValueError:
        await message.answer("❌ ID должен быть числом.")
        return

    if await db.delete_video(video_id):
        await message.answer(f"✅ Видео #{video_id} удалено.")
    else:
        await message.answer(f"❌ Видео #{video_id} не найдено.")


@dp.message(Command("ban"))
async def cmd_ban(message: Message):
    """Бан пользователя."""
    if message.from_user.id not in ADMIN_IDS:
        await message.answer("⛔ У вас нет прав для использования этой команды.")
        return

    args = message.text.split()[1:]
    if not args:
        await message.answer("❌ Укажите user_id: /ban <user_id>")
        return

    try:
        user_id = int(args[0])
    except ValueError:
        await message.answer("❌ user_id должен быть числом.")
        return

    await db.ban_user(user_id)
    await message.answer(f"✅ Пользователь {user_id} забанен.")


@dp.message(Command("unban"))
async def cmd_unban(message: Message):
    """Разбан пользователя."""
    if message.from_user.id not in ADMIN_IDS:
        await message.answer("⛔ У вас нет прав для использования этой команды.")
        return

    args = message.text.split()[1:]
    if not args:
        await message.answer("❌ Укажите user_id: /unban <user_id>")
        return

    try:
        user_id = int(args[0])
    except ValueError:
        await message.answer("❌ user_id должен быть числом.")
        return

    await db.unban_user(user_id)
    await message.answer(f"✅ Пользователь {user_id} разбанен.")


@dp.message(Command("list"))
async def cmd_list(message: Message):
    """Список всех видео."""
    if message.from_user.id not in ADMIN_IDS:
        await message.answer("⛔ У вас нет прав для использования этой команды.")
        return

    videos = await db.list_videos()
    if not videos:
        await message.answer("📭 База видео пуста.")
        return

    text = "📹 <b>Список видео:</b>\n\n"
    for v in videos:
        aliases = v["aliases"][:50] + "..." if len(v["aliases"]) > 50 else v["aliases"]
        text += f"<b>#{v['id']}</b> - {v['title']}\n"
        text += f"📝 aliases: {aliases or 'нет'}\n\n"

    # Разбиваем на части если слишком длинное сообщение
    if len(text) > 4000:
        parts = []
        current = "📹 <b>Список видео:</b>\n\n"
        for v in videos:
            entry = f"<b>#{v['id']}</b> - {v['title']}\n📝 aliases: {v['aliases'][:50] or 'нет'}\n\n"
            if len(current) + len(entry) > 4000:
                parts.append(current)
                current = entry
            else:
                current += entry
        parts.append(current)

        for part in parts:
            await message.answer(part, parse_mode=ParseMode.HTML)
    else:
        await message.answer(text, parse_mode=ParseMode.HTML)


@dp.message(Command("stats"))
async def cmd_stats(message: Message):
    """Статистика бота."""
    if message.from_user.id not in ADMIN_IDS:
        await message.answer("⛔ У вас нет прав для использования этой команды.")
        return

    stats = await db.get_stats()
    text = (
        "📊 <b>Статистика бота:</b>\n\n"
        f"🎬 Видео в базе: <b>{stats['videos']}</b>\n"
        f"👥 Пользователей: <b>{stats['users']}</b>\n"
        f"🔍 Поисковых запросов: <b>{stats['queries']}</b>"
    )
    await message.answer(text, parse_mode=ParseMode.HTML)


@dp.message(Command("help"))
async def cmd_help(message: Message):
    """Помощь по использованию бота."""
    text = (
        "📖 <b>Как использовать бота:</b>\n\n"
        "<b>1. Inline режим (в любом чате):</b>\n"
        "   <code>@mellbot мотик</code> — ищем видео\n"
        "   Выберите видео из списка\n\n"
        "<b>2. Оверлей с текстом:</b>\n"
        "   • <b>В личном чате с ботом:</b>\n"
        "     Отправьте видео как <b>ответ</b> на сообщение с текстом\n"
        "     Бот автоматически добавит пузырёк сообщения\n\n"
        "   • <b>В группе (где бот не админ):</b>\n"
        "     Используйте команду:\n"
        "     <code>/overlay &lt;id&gt; \"текст\" \"имя\"</code>\n"
        "     Пример: <code>/overlay 5 \"Привет\" \"Максим\"</code>\n\n"
        "<b>3. Список видео:</b>\n"
        "   <code>/list</code> — посмотреть все видео с ID\n\n"
        "<b>4. Поиск:</b>\n"
        "   Бот ищет по названию, описанию и транскрипции видео"
    )
    await message.answer(text, parse_mode=ParseMode.HTML)


@dp.message(Command("start"))
async def cmd_start(message: Message):
    """Приветственное сообщение."""
    await cmd_help(message)


@dp.message(Command("overlay"))
async def cmd_overlay(message: Message):
    """
    Отправка видео с наложением текста (для групп, где бот не админ).
    Использование: /overlay <id видео> <текст для наложения> [имя отправителя]
    Пример: /overlay 5 "Привет всем!" "Максим"
    """
    # Получаем аргументы
    args = message.text.split(maxsplit=3)[1:]
    if len(args) < 2:
        await message.answer(
            "❌ Неверный формат.\n\n"
            "<b>Использование:</b>\n"
            "<code>/overlay &lt;id видео&gt; &lt;текст&gt; [имя]</code>\n\n"
            "<b>Пример:</b>\n"
            "<code>/overlay 5 \"Привет всем\"</code>"
        )
        return

    try:
        video_id = int(args[0])
    except ValueError:
        await message.answer("❌ ID видео должен быть числом.")
        return

    overlay_text = args[1]
    sender_name = args[2] if len(args) > 2 else None

    # Ищем видео
    video = await db.get_video(video_id)
    if not video:
        await message.answer(f"❌ Видео #{video_id} не найдено.")
        return

    # Показываем статус
    processing_msg = await message.answer("🎬 Создаю видео с оверлеем...")

    try:
        # Скачиваем видео
        file = await bot.get_file(video["file_id"])
        temp_dir = tempfile.mkdtemp()
        original_path = Path(temp_dir) / "original.mp4"
        output_path = Path(temp_dir) / "output.mp4"

        await bot.download_file(file.file_path, original_path)

        # Накладываем пузырёк сообщения
        await video_overlay.add_message_overlay(
            str(original_path),
            overlay_text,
            str(output_path),
            sender_name=sender_name,
        )

        # Отправляем видео
        video_file = FSInputFile(str(output_path))
        await message.reply_video(
            video=video_file,
            caption=f"📝 {overlay_text[:100]}{'...' if len(overlay_text) > 100 else ''}",
        )

    except Exception as e:
        logger.error(f"Ошибка при создании оверлея: {e}")
        await message.answer(f"❌ Ошибка: {e}")

    finally:
        await processing_msg.delete()
        if 'original_path' in locals():
            original_path.unlink(missing_ok=True)
        if 'output_path' in locals():
            output_path.unlink(missing_ok=True)
        if 'temp_dir' in locals():
            Path(temp_dir).rmdir()


# Обработка видео при загрузке
@dp.message(F.video | F.video_note)
async def handle_video_upload(message: Message):
    """Обработка загруженного видео."""
    user_id = message.from_user.id

    # Проверяем, находимся ли в режиме загрузки
    if user_id not in upload_states:
        return

    state = upload_states[user_id]

    if state["step"] == "waiting_video":
        # Сохраняем file_id видео
        file_id = message.video.file_id if message.video else message.video_note.file_id
        state["file_id"] = file_id
        state["step"] = "waiting_title"

        await message.answer(
            "✅ Видео получено!\n\n"
            "Теперь отправьте <b>название</b> для этого видео:"
        )

    elif state["step"] == "processing":
        await message.answer("⏳ Подождите, идёт обработка предыдущего видео...")


# Обработка текста при загрузке
@dp.message(F.text)
async def handle_text_upload(message: Message):
    """Обработка текста (название и aliases) при загрузке видео."""
    user_id = message.from_user.id

    if user_id not in upload_states:
        return

    state = upload_states[user_id]

    if state["step"] == "waiting_title":
        state["title"] = message.text
        state["step"] = "waiting_aliases"
        await message.answer(
            f"✅ Название: <b>{message.text}</b>\n\n"
            "Теперь отправьте <b>aliases</b> через запятую (или отправьте '-', чтобы пропустить):"
        )

    elif state["step"] == "waiting_aliases":
        aliases = "" if message.text == "-" else message.text
        state["aliases"] = aliases
        state["step"] = "processing"

        await message.answer("⏳ Обрабатываю видео и запускаю транскрибацию...")

        try:
            # Скачиваем видео для транскрибации
            file = await bot.get_file(state["file_id"])
            temp_dir = tempfile.mkdtemp()
            video_path = Path(temp_dir) / "temp_video.mp4"

            await bot.download_file(file.file_path, video_path)

            # Транскрибируем
            transcript = await transcriber.transcribe(str(video_path))

            # Добавляем слова из транскрипции к aliases
            all_aliases = state["aliases"]
            if transcript:
                transcript_words = " ".join(
                    list(set(transcript.lower().split()))[:20]  # Берём первые 20 уникальных слов
                )
                if all_aliases:
                    all_aliases += ", " + transcript_words
                else:
                    all_aliases = transcript_words

            # Сохраняем в БД
            video_id = await db.add_video(
                file_id=state["file_id"],
                title=state["title"],
                aliases=all_aliases,
                transcript=transcript,
            )

            # Очистка
            video_path.unlink(missing_ok=True)
            Path(temp_dir).rmdir()

            await message.answer(
                f"✅ Видео успешно добавлено!\n\n"
                f"<b>ID:</b> #{video_id}\n"
                f"<b>Название:</b> {state['title']}\n"
                f"<b>Aliases:</b> {all_aliases[:200]}{'...' if len(all_aliases) > 200 else ''}\n\n"
                f"📝 Транскрипция: {transcript[:150]}{'...' if len(transcript) > 150 else '' if transcript else 'нет'}"
            )

        except Exception as e:
            logger.error(f"Ошибка при загрузке видео: {e}")
            await message.answer(f"❌ Ошибка при обработке видео: {e}")

        finally:
            del upload_states[user_id]


# === MAIN ===
async def main():
    """Главная функция запуска бота."""
    # Инициализация БД
    await db.init_db()

    logger.info("Бот запущен!")

    # Запуск бота
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
