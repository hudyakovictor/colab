# -*- coding: utf-8 -*-
"""p3dmm_ui.py — панель статуса и шаги ноутбука Pixel3DMM «одна кнопка».

Идея: ноутбук показывает ровно три вещи — кнопку запуска подготовки, кнопку загрузки
фото/архива и живую панель, по которой видно «работает / повисло / готово / ошибка».
Вся логика шагов живёт здесь, чтобы ячейки читались с одного экрана.

Панель (класс Board) каждую секунду перерисовывает:
    этап + фаза + прогресс-бар + счётчики + «сколько секунд назад был вывод» +
    хвост активного лога трекера. Если вывода нет дольше IDLE_WARN секунд, панель
    честно пишет, что шаг тяжёлый, а не молчит.

Публичные функции:
    prepare(board, ...)      подготовка: Drive → файлы → окружение → веса → проверка → кеш
    process(board, ...)      загрузка кнопкой → план → реконструкция → индекс → сводка
    status(output_root)      что уже лежит на Drive (работает без GPU и без окружения)
    status_html(result)      таблица результатов для вывода в ячейке
    legend_html()            что именно извлекается в папку каждого фото
"""
from __future__ import annotations

import base64
import collections
import html
import json
import os
import pathlib
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import zipfile
from datetime import datetime

# --------------------------------------------------------------------------- константы
UI_VERSION = "3.1"        # 3.1: скрипты окружения/весов больше не качаются (был HTTP 404)
REPO_RAW = "https://raw.githubusercontent.com/hudyakovictor/colab/main"

CONTENT = pathlib.Path("/content")
PIPELINE = CONTENT / "p3dmm_pipeline.py"
ENV_SH = CONTENT / "p3dmm_env.sh"
WEIGHTS_PY = CONTENT / "p3dmm_weights.py"
CFG_FILE = CONTENT / "p3dmm_config.json"
STOP_FILE = CONTENT / "STOP"
UPLOAD_RAW = CONTENT / "upload_raw"
UPLOAD_DIR = CONTENT / "upload"
ENV_PYTHON = CONTENT / "p3dmm-env/bin/python"
DRIVE_ROOT = pathlib.Path("/content/drive/MyDrive")
LOG_DIR = CONTENT / "logs"
UI_LOG_DIR = LOG_DIR / "ui"
PIPELINE_LOG_DIR = LOG_DIR            # сюда pipeline пишет логи трекинга <батч>__<кадр>.log

OUTPUT_DEFAULT = "MyDrive/pixel3dmm_results"
FLAME_ZIPS = ("FLAME2020.zip", "FLAME2023.zip")
# что действительно качается/берётся с Drive; p3dmm_env.sh и p3dmm_weights.py лежат ВНУТРИ
# этого модуля (константы ENV_SCRIPT / WEIGHTS_SCRIPT) и создаются на месте — качать их неоткуда
SUPPORT_FILES = ("p3dmm_pipeline.py",)
IMG_EXT = {".jpg", ".jpeg", ".png"}
ARCHIVE_EXT = {".zip"}

IDLE_NOTE = 45          # секунд без вывода, после которых панель начинает объяснять паузу
IDLE_WARN = 180         # секунд без вывода — предупреждение «возможно, тяжёлый шаг / зависло»

# что лежит в папке каждого фото (имя файла -> человеческое описание)
PHOTO_LEGEND = (
    ("mesh.ply", "3D-меш лица этого фото в позе снимка (FLAME, вершины + текстурные координаты)"),
    ("canonical.ply", "меш в канонической позе — одна и та же голова без поворота и мимики"),
    ("params.json", "все параметры: shape/exp/jaw/neck/eyes/eyelids, R, t, камера, FLAME-конфиг, "
                    "ракурс (yaw/pitch/roll + метка), наблюдаемость, список файлов"),
    ("checkpoint.frame", "сырой чекпоинт трекера: тензоры FLAME-параметров без округления"),
    ("landmarks_68.npy", "68 ландмарков iBUG на кадре (результат трекинга)"),
    ("landmarks_7.npy", "7 ключевых точек (глаза, нос, рты) в координатах кадра"),
    ("landmarks_eyes.npy", "контуры глаз"),
    ("landmarks_pipnet.npy", "98 точек PIPNet (WFLW) — детекция до трекинга"),
    ("mica_identity.npy", "identity-код MICA (300 чисел) — «кто это», для сравнения лиц"),
    ("mica_mesh.ply", "меш MICA по identity-коду"),
    ("mica_kpt7.npy", "7 точек MICA"), ("mica_kpt68.npy", "68 точек MICA"),
    ("arcface.npy", "эмбеддинг ArcFace — вектор лица для распознавания"),
    ("crop.jpg", "кроп лица, который реально видела сеть"),
    ("segmentation.png", "сегментация лица/кожи/волос (FaceR)"),
    ("normals.png", "карта нормалей, извлечённая сетью Pixel3DMM"),
    ("uv_map.png", "UV-карта лица в развёртке FLAME"),
    ("mask_fg.png", "маска силуэта головы"), ("mask_uv.png", "маска, где работали UV-сопоставления"),
    ("mask_normal.png", "маска, где работали нормали"),
    ("visibility_points.npy", "какие точки меша видно на снимке (наблюдаемость)"),
    ("overlay.jpg", "кадр из result.mp4: меш поверх лица"),
    ("source.jpg", "исходный файл фото (копия)"),
    ("run.log", "полный лог трекинга этого фото: итерации, лоссы, ошибки"),
    ("done.json", "метка готовности + время расчёта"),
)


# --------------------------------------------------------------------------- мелочи
def _now_stamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M")


def human(seconds):
    try:
        seconds = int(max(0, float(seconds)))
    except (TypeError, ValueError):
        return "?"
    if seconds < 60:
        return f"{seconds} с"
    if seconds < 3600:
        return f"{seconds // 60} мин {seconds % 60:02d} с"
    return f"{seconds // 3600} ч {seconds % 3600 // 60:02d} мин"


TRANSLIT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e", "ж": "zh", "з": "z",
    "и": "i", "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r",
    "с": "s", "т": "t", "у": "u", "ф": "f", "х": "h", "ц": "c", "ч": "ch", "ш": "sh", "щ": "sch",
    "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
}


def latin(text, fallback="photos"):
    """Имя батча латиницей: pipeline подставляет его в CLI и в имя папки на Drive."""
    text = str(text or "")
    out = "".join(TRANSLIT.get(ch, ch) for ch in text.lower())
    out = re.sub(r"[^0-9a-z_-]+", "_", out).strip("_")
    return (out[:40] or fallback)


def drive_path(raw):
    """«MyDrive/x», «x» или абсолютный путь -> pathlib.Path."""
    raw = str(raw or "").strip().strip('"')
    if raw.startswith("/"):
        return pathlib.Path(raw)
    return DRIVE_ROOT / (raw[len("MyDrive/"):] if raw.startswith("MyDrive/") else raw)


def download(url, dst, timeout=120):
    """Скачать файл (urllib, без внешних пакетов)."""
    import urllib.request
    dst = pathlib.Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "pixel3dmm-colab"})
    with urllib.request.urlopen(request, timeout=timeout) as response, dst.open("wb") as stream:
        shutil.copyfileobj(response, stream)
    if dst.stat().st_size < 512:
        raise OSError(f"файл {url} скачался пустым ({dst.stat().st_size} байт)")
    return dst


# --------------------------------------------------------------------------- панель
class Board:
    """Живая панель: одна строка на этап, прогресс, «сколько секунд назад был вывод»."""

    def __init__(self, title, total=None):
        self.title = title
        self.phase = "Запуск"
        self.stage_label = ""
        self.stage_no, self.stage_total = 0, 0
        self.current = ""
        self.stats = ""
        self.completed, self.total = 0, total
        self.lines = collections.deque(maxlen=14)
        self.state = "running"
        self.warning = False
        self.started = time.monotonic()
        self.stage_started = self.started        # часы текущего этапа (для оценки остатка)
        self.last_output = self.started
        self.lock = threading.RLock()
        self._stop = threading.Event()
        self.stop_reason = ""
        self._handle = None
        self._worker = None
        self._stop_button = None

    # ----- жизненный цикл
    def __enter__(self):
        from IPython.display import HTML, display
        self._display = display
        self._handle = display(HTML(self.render()), display_id=True)
        self._worker = threading.Thread(target=self._refresh_loop, daemon=True)
        self._worker.start()
        self._make_stop_button()
        return self

    def __exit__(self, kind, error, traceback):
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=2)
        if kind:
            self.state = "stopped" if issubclass(kind, KeyboardInterrupt) else "error"
            self.phase = "Прервано" if issubclass(kind, KeyboardInterrupt) else "Ошибка"
            self.note(str(error) or "Выполнение прервано. Всё, что успело сохраниться, лежит на Drive.")
        else:
            self.state = "warning" if self.warning else "done"
            self.phase = "Завершено"
            if self.total:
                self.completed = self.total
        self._paint()
        return False

    def _make_stop_button(self):
        """Кнопка «Стоп»: ставит STOP-файл и гасит текущий дочерний процесс."""
        try:
            import ipywidgets as widgets
            button = widgets.Button(description="⏹ Остановить", button_style="danger",
                                    layout=widgets.Layout(width="170px"))
            note = widgets.HTML("<span style='font-size:12px;color:#666'>остановка после текущего фото</span>")
            button.on_click(self._on_stop_click)
            self._display(widgets.HBox([button, note]))
            self._stop_button = button
        except Exception:
            self._stop_button = None

    def _on_stop_click(self, _button=None):
        self.request_stop("нажата кнопка «Остановить»")

    def request_stop(self, reason):
        with self.lock:
            if self._stop.is_set():
                return
            self.stop_reason = reason
            self._stop.set()
            self.stats = "останавливаюсь после текущего шага…"
        try:
            STOP_FILE.write_text(reason, encoding="utf-8")
        except OSError:
            pass
        self._paint()

    @property
    def stop_requested(self):
        return self._stop.is_set()

    # ----- обновление содержимого
    def set(self, **values):
        with self.lock:
            for key, value in values.items():
                if hasattr(self, key):
                    setattr(self, key, value)

    def stage(self, label, index=None, total=None):
        with self.lock:
            self.stage_label = label
            if index is not None:
                self.stage_no = index
            if total is not None:
                self.stage_total = total
            self.phase = label
            self.current = ""
            self.stats = ""
            self.completed, self.total = 0, None
            self.stage_started = time.monotonic()
            self.last_output = self.stage_started
        self._paint()

    def note(self, text):
        """Строки вывода шага: обычный текст в хвост, __P3DMM_STATUS__ — в поля панели."""
        with self.lock:
            for raw in str(text).replace("\r\n", "\n").replace("\r", "\n").splitlines():
                line = raw.rstrip()
                if not line.strip():
                    continue
                if line.startswith("__P3DMM_STATUS__"):
                    try:
                        event = json.loads(line[len("__P3DMM_STATUS__"):])
                    except (ValueError, TypeError):
                        continue
                    for key in ("phase", "current", "stats", "completed", "total"):
                        if key in event and event[key] not in (None, ""):
                            setattr(self, key, event[key])
                    continue
                self.lines.append(line[:600])
                self.last_output = time.monotonic()

    def warn(self, text):
        with self.lock:
            self.warning = True
        self.note("ПРЕДУПРЕЖДЕНИЕ: " + text)

    def _paint(self):
        if self._handle is None:
            return
        try:
            from IPython.display import HTML
            self._handle.update(HTML(self.render()))
        except Exception:
            pass

    def _refresh_loop(self):
        while not self._stop.wait(1.0):
            self._paint()

    # ----- активный лог трекера (главное доказательство, что процесс жив)
    def _tracker_tail(self):
        try:
            files = [p for p in PIPELINE_LOG_DIR.glob("*.log") if p.is_file()]
        except OSError:
            return "", 0, 0
        if not files:
            return "", 0, 0
        newest = max(files, key=lambda p: p.stat().st_mtime)
        age = time.time() - newest.stat().st_mtime
        tail = ""
        try:
            with newest.open("rb") as stream:
                size = newest.stat().st_size
                stream.seek(max(0, size - 6000))
                block = stream.read().decode("utf-8", "replace")
            chunks = [c.strip() for c in block.replace("\r\n", "\n").split("\n") if c.strip()]
            tail = " ⟵ ".join(chunks[-2:])[-500:]
        except OSError:
            pass
        return f"{newest.name}: {tail}", age, newest.stat().st_size

    def render(self):
        with self.lock:
            esc = html.escape
            elapsed = time.monotonic() - self.started
            idle = time.monotonic() - self.last_output
            state_text = {"running": "⏳ РАБОТАЕТ", "done": "✅ ГОТОВО",
                          "warning": "⚠️ ГОТОВО С ЗАМЕЧАНИЯМИ", "error": "❌ ОШИБКА",
                          "stopped": "⏹ ОСТАНОВЛЕНО"}.get(self.state, self.state)
            color = {"done": "#1a7f4b", "warning": "#a26a00", "error": "#c0392b",
                     "stopped": "#a26a00"}.get(self.state, "#1c5d99")
            if idle < IDLE_NOTE:
                beat, beat_color = f"вывод {int(idle)} с назад", "#1a7f4b"
            elif idle < IDLE_WARN:
                beat, beat_color = f"вывод {int(idle)} с назад — шаг тяжёлый, это нормально", "#a26a00"
            else:
                beat, beat_color = (f"вывода нет {human(idle)} — смотри лог ниже; "
                                    f"если висит долго, жми «Остановить»"), "#c0392b"

            stage_elapsed = time.monotonic() - self.stage_started
            if self.total:
                value = min(self.total, max(0, self.completed))
                percent = 100.0 * value / max(1, self.total)
                bar = (f'<div style="background:#e8eaed;border-radius:6px;height:14px;overflow:hidden">'
                       f'<div style="width:{percent:.1f}%;height:100%;background:{color};'
                       f'transition:width .4s"></div></div>'
                       f'<div style="font-size:12px;color:#444;margin-top:2px">'
                       f'{value} из {self.total} ({percent:.0f}%)'
                       + (f' · осталось ~{human(stage_elapsed / value * (self.total - value))}'
                          if value else '') + '</div>')
            else:
                bar = (f'<div style="background:#e8eaed;border-radius:6px;height:14px;overflow:hidden">'
                       f'<div style="width:35%;height:100%;background:{color};opacity:.55"></div></div>')

            stages = (f"Этап {self.stage_no}/{self.stage_total}: " if self.stage_total else "")
            tracker, tracker_age, _size = self._tracker_tail()
            tracker_html = ""
            if tracker and self.state == "running":
                fresh = "color:#1a7f4b" if tracker_age < 60 else (
                    "color:#a26a00" if tracker_age < IDLE_WARN else "color:#c0392b")
                tracker_html = (f'<div style="font-size:12px;margin-top:8px;{fresh};'
                                f'font-family:ui-monospace,Menlo,monospace;overflow-wrap:anywhere">'
                                f'трекер ({human(tracker_age)} назад): {esc(tracker)}</div>')
            recent = esc("\n".join(self.lines) or "пока нет вывода")
            return (
                f'<section style="font:14px/1.55 -apple-system,system-ui,Segoe UI,Roboto,sans-serif;'
                f'border:1px solid {color};border-left:6px solid {color};border-radius:10px;'
                f'padding:14px 16px;background:#fff;color:#111;max-width:100%">'
                f'<div style="display:flex;justify-content:space-between;gap:14px;align-items:baseline">'
                f'<strong style="font-size:15px">{esc(self.title)}</strong>'
                f'<span style="color:{color};font-weight:600;white-space:nowrap">{state_text}</span></div>'
                f'<div style="margin-top:8px">{esc(stages)}{esc(self.stage_label)}</div>'
                f'<div style="font-weight:600">{esc(self.phase)}</div>'
                f'<div style="font-family:ui-monospace,Menlo,monospace;font-size:13px;'
                f'overflow-wrap:anywhere">{esc(self.current)}</div>'
                f'<div style="font-size:13px;color:#333">{esc(self.stats)}</div>'
                f'<div style="margin:10px 0 6px">{bar}</div>'
                f'<div style="font-size:12px;color:{beat_color}">⏱ всего {human(elapsed)} · {esc(beat)}</div>'
                f'{tracker_html}'
                f'<details style="margin-top:10px"><summary style="cursor:pointer;color:#1c5d99">'
                f'последние строки ({esc(str(UI_LOG_DIR))})</summary>'
                f'<pre style="white-space:pre-wrap;overflow-wrap:anywhere;max-height:240px;'
                f'overflow:auto;font-size:12px;background:#f6f7f9;padding:8px;border-radius:6px;'
                f'margin:6px 0 0">{recent}</pre></details></section>'
            )


# --------------------------------------------------------------------------- запуск шагов
def run_stream(board, command, label, cwd="/content", env=None, check=True, timeout_min=None):
    """Выполнить команду, показывая её вывод на панели. Возвращает код возврата."""
    command = [str(part) for part in command]
    board.set(phase=label, current="", stats="")
    UI_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = UI_LOG_DIR / (re.sub(r"[^0-9A-Za-z]+", "_", label)[:50] + ".log")
    environment = dict(os.environ)
    environment.update({"PYTHONUNBUFFERED": "1", "P3DMM_NOTEBOOK_STATUS": "1",
                        "PIP_DISABLE_PIP_VERSION_CHECK": "1", "PIP_PREFER_BINARY": "1",
                        "GIT_TERMINAL_PROMPT": "0"})
    environment.update(env or {})
    deadline = time.time() + timeout_min * 60 if timeout_min else None
    with log_path.open("a", encoding="utf-8", errors="replace") as log:
        log.write(f"\n$ {' '.join(command)}\n")
        log.flush()
        proc = subprocess.Popen(command, cwd=cwd, env=environment, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, errors="replace",
                                bufsize=1, start_new_session=True)

        def reader():
            for line in proc.stdout:
                log.write(line)
                board.note(line)
            log.flush()

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        try:
            while proc.poll() is None:
                if board.stop_requested or (deadline and time.time() > deadline):
                    if deadline and time.time() > deadline and not board.stop_requested:
                        board.note("ТАЙМАУТ шага — прерываю")
                    try:
                        os.killpg(proc.pid, signal.SIGINT)
                    except ProcessLookupError:
                        pass
                    time.sleep(5)
                    if proc.poll() is None:
                        try:
                            os.killpg(proc.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    break
                time.sleep(0.5)
            rc = proc.wait()
        except BaseException:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            raise
        finally:
            thread.join(timeout=15)
            try:
                proc.stdout.close()
            except Exception:
                pass
    board.note(f"$ {' '.join(command)} -> код {rc} (полный лог: {log_path})")
    if check and rc:
        raise RuntimeError(f"шаг «{label}» завершился с кодом {rc}. "
                           f"Подробности в логе {log_path} и на панели выше.")
    return rc


# --------------------------------------------------------------------------- подготовка
def mount_drive(board=None):
    DRIVE_ROOT.parent.mkdir(parents=True, exist_ok=True)
    if DRIVE_ROOT.is_dir():
        return DRIVE_ROOT
    from google.colab import drive
    if board:
        board.set(phase="Подключение Google Drive (разреши доступ в окне браузера)")
    drive.mount("/content/drive")
    if not DRIVE_ROOT.is_dir():
        raise RuntimeError("Google Drive не подключился. Запусти ячейку ещё раз и разреши доступ.")
    return DRIVE_ROOT


def ensure_support_files(board=None):
    """Пайплайн: локально → MyDrive → репозиторий. Скрипты окружения и весов — из этого модуля."""
    missing = []
    for name in SUPPORT_FILES:
        target = CONTENT / name
        if target.is_file() and target.stat().st_size > 512:
            continue
        local_copy = DRIVE_ROOT / name
        if local_copy.is_file() and local_copy.stat().st_size > 512:
            shutil.copy2(local_copy, target)
            if board:
                board.note(f"{name}: взят с Google Drive")
            continue
        if board:
            board.set(phase=f"Скачиваю {name}")
        try:
            download(f"{REPO_RAW}/{name}", target)
            if board:
                board.note(f"{name}: скачан из репозитория")
        except Exception as error:
            missing.append(f"{name} ({type(error).__name__}: {error})")
    if missing:
        raise RuntimeError(
            "Не хватает файла пайплайна: " + ", ".join(missing) +
            f". Положи p3dmm_pipeline.py в корень MyDrive или рядом с ноутбуком в репозитории ({REPO_RAW}).")
    env_path, weights_path = write_scripts()      # ENV_SH и WEIGHTS_PY всегда заново из модуля
    for path, minimum in ((env_path, 2000), (weights_path, 2000)):
        if not path.is_file() or path.stat().st_size < minimum:
            raise RuntimeError(f"Не удалось записать {path.name} "
                               f"({path.stat().st_size if path.is_file() else 0} байт). "
                               f"Обнови /content/p3dmm_ui.py из репозитория и повтори ячейку.")
    if board:
        board.note(f"p3dmm_ui {UI_VERSION}")
        board.note(f"скрипты окружения и весов созданы из p3dmm_ui.py: "
                   f"{env_path.name} ({env_path.stat().st_size} Б), "
                   f"{weights_path.name} ({weights_path.stat().st_size} Б)")
    return True


CONFIG_BASE = {
    "start_from": "",
    "export": "photos",
    "face_filter": True,
    "face_min_score": 0.75,
    "use_flame2023": "true",
    "torch_compile": False,
    "sync_to_drive": True,
    "single_iters": 800,
    "attempts": 3,
    "min_free_gb": 2.0,
    "session_max_hours": 11.0,
    "session_hard_limit_hours": 0.0,
    "batch_timeout_min": 90,
    "keep_preproc": True,
    "unit": "photo",
    "photos_per_chunk": 40,
    "max_photos_per_run": 500,
    "photo_timeout_min": 15,
    "keep_archive": False,
    "flat": True,
    "date_fallback": True,
    "source": "upload",
}


def write_config(**overrides):
    cfg = dict(CONFIG_BASE)
    cfg.update(overrides)
    CFG_FILE.write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
    return cfg


def preflight_report(output_root):
    """Читает _state/preflight.json — что готово, что мешает."""
    path = pathlib.Path(output_root) / "_state/preflight.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def prepare(board, output_root=OUTPUT_DEFAULT, auto_cache=True):
    """Подготовка сессии: Drive, файлы, окружение, веса, проверка, кеш.

    Возвращает словарь с итогом проверки. Всё идемпотентно: кеш с Drive поднимается
    за минуты, а полная сборка случается только в первой сессии.
    """
    out = drive_path(output_root)
    stages = 4
    STOP_FILE.unlink(missing_ok=True)

    board.stage("Google Drive и папки", 1, stages)
    mount_drive(board)
    out.mkdir(parents=True, exist_ok=True)
    board.note(f"результаты: {out}")
    absent = [name for name in FLAME_ZIPS if not (DRIVE_ROOT / name).is_file()]
    if absent:
        raise RuntimeError(
            "В корне Google Drive нет " + ", ".join(absent) +
            ". Положи лицензированные архивы FLAME в корень MyDrive (не в подпапку) и повтори.")

    board.stage("Вспомогательные файлы", 2, stages)
    ensure_support_files(board)
    write_config(input=str(UPLOAD_DIR), output=str(out))
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    for folder in (LOG_DIR, UI_LOG_DIR, CONTENT / "work", CONTENT / "stage",
                   CONTENT / "p3dmm/preprocessed", CONTENT / "p3dmm/tracking"):
        folder.mkdir(parents=True, exist_ok=True)
    env_cfg = pathlib.Path.home() / ".config/pixel3dmm/.env"
    env_cfg.parent.mkdir(parents=True, exist_ok=True)
    env_cfg.write_text("PIXEL3DMM_CODE_BASE=/content/pixel3dmm/\n"
                       "PIXEL3DMM_PREPROCESSED_DATA=/content/p3dmm/preprocessed/\n"
                       "PIXEL3DMM_TRACKING_OUTPUT=/content/p3dmm/tracking/\n", encoding="utf-8")

    board.stage("Окружение: Python 3.9 + CUDA-сборки (самый долгий этап, 1 раз)", 3, stages)
    rc = run_stream(board, ["bash", str(ENV_SH)], "Окружение Pixel3DMM", check=True)
    board.note(f"окружение: код {rc}")

    if not ENV_PYTHON.is_file():
        raise RuntimeError("Не появился /content/p3dmm-env/bin/python — установка окружения не удалась, "
                           "смотри лог на панели.")
    run_stream(board, [str(ENV_PYTHON), "-u", str(WEIGHTS_PY)], "Веса сетей и FLAME", check=True)

    board.set(phase="Проверка готовности (preflight)")
    preflight_rc = run_stream(board, [str(ENV_PYTHON), "-u", str(PIPELINE), "preflight"],
                             "Проверка готовности", check=False)
    report = preflight_report(out)
    problems = list(report.get("problems") or [])
    if preflight_rc and not problems:
        problems = ["preflight завершился с ошибкой — смотри лог выше"]

    board.stage("Кеш окружения для следующих сессий", 4, stages)
    manifest = out / "_env/manifest.json"
    if auto_cache and not manifest.is_file():
        board.note("Кеша ещё нет: упаковываю окружение на Drive (один раз, ~10–20 минут; "
                   "занимает ~10 ГБ. Не нужно — SAVE_ENV_CACHE = False в ячейке 1)")
        board.set(phase="Упаковка кеша окружения на Drive")
        run_stream(board, [str(ENV_PYTHON), "-u", str(PIPELINE), "freeze"], "Кеш окружения",
                   check=False)
    else:
        board.note("Кеш окружения уже сохранён — упаковку пропускаю")
        board.set(phase="Кеш окружения уже сохранён")

    if problems:
        board.warning = True
        for item in problems:
            board.note("! " + item)
    board.set(phase="Готово к загрузке фото", current="",
              stats=f"результаты пойдут в {out}")
    board.state = "warning" if problems else "done"
    return {"output": str(out), "problems": problems, "free_gb": report.get("free_gb"),
            "flame2023": report.get("flame2023")}


# --------------------------------------------------------------------------- загрузка
def _flatten(photo_paths, target):
    """Все фото — в одну папку батча; при совпадении имён добавляем имя подпапки."""
    target.mkdir(parents=True, exist_ok=True)
    seen, kept = {}, []
    for path in photo_paths:
        path = pathlib.Path(path)
        stem, suffix = path.stem, path.suffix.lower()
        name = f"{stem}{suffix}"
        if name in seen:
            seen[name] += 1
            parent = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", path.parent.name)[:24] or "set"
            name = f"{parent}__{stem}{suffix}"
            if name in seen:
                seen[name] += 1
                name = f"{parent}__{stem}_{seen[name]}{suffix}"
        seen.setdefault(name, 1)
        shutil.copy2(path, target / name)
        kept.append(name)
    return kept


def _unzip(archive, dest):
    dest.mkdir(parents=True, exist_ok=True)
    found = []
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            name = info.filename
            if info.is_dir() or "__MACOSX" in name or pathlib.PurePosixPath(name).name.startswith("._"):
                continue
            if pathlib.Path(name).suffix.lower() not in IMG_EXT:
                continue
            out = (dest / name).resolve()
            if not str(out).startswith(str(dest.resolve())):
                raise RuntimeError(f"опасный путь в архиве: {name}")
            out.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, out.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            found.append(out)
    return found


def ask_files():
    """Кнопка выбора файлов Colab: одно фото, несколько фото или .zip с папкой фото."""
    from google.colab import files as colab_files
    return colab_files.upload()


def last_upload():
    """Батчи прошлой загрузки этой сессии (переживают повторный запуск ячейки 2)."""
    data = _read_json(UPLOAD_DIR / "_last_upload.json")
    batches = []
    for item in data.get("batches") or []:
        path = pathlib.Path(str(item.get("path", "")))
        if path.is_dir() and any(path.iterdir()):
            batches.append((str(item.get("name") or path.name), path))
    return batches


def collect_uploads(board, reuse=False):
    """Загрузка кнопкой -> список батчей [(имя, папка с фото)]."""
    if reuse:
        batches = last_upload()
        if batches:
            board.stage("Повтор прошлой загрузки (без окна выбора)", 1, 4)
            total = sum(len(list(p.iterdir())) for _, p in batches)
            board.note(f"беру уже загруженное: {total} фото в {len(batches)} батч(ах); "
                       f"готовые фото пропустятся")
            return batches
        board.warn("сохранённой загрузки нет — открываю окно выбора файлов")
    board.stage("Ждём файл: нажми «Выберите файлы»", 1, 4)
    board.set(phase="Выбери фото или .zip с папкой фото",
              stats="одно фото · несколько фото · архив .zip (структура папок не важна)")
    uploaded = ask_files()
    if not uploaded:
        raise RuntimeError("Файлы не выбраны — нажми ▶︎ у ячейки ещё раз.")
    shutil.rmtree(UPLOAD_RAW, ignore_errors=True)
    UPLOAD_RAW.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    saved = {}
    for name, payload in uploaded.items():
        target = UPLOAD_RAW / pathlib.Path(name).name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        saved[target.name] = target
        board.note(f"получен {target.name} ({len(payload) / 1e6:.1f} МБ)")

    batches, loose_images = [], []
    for name, path in sorted(saved.items()):
        suffix = pathlib.Path(name).suffix.lower()
        if suffix in ARCHIVE_EXT:
            extract_to = UPLOAD_RAW / (pathlib.Path(name).stem + "_unpacked")
            photos = _unzip(path, extract_to)
            if not photos:
                board.warn(f"в архиве {name} нет изображений .jpg/.jpeg/.png")
                continue
            batch = latin(pathlib.Path(name).stem)
            shutil.rmtree(UPLOAD_DIR / batch, ignore_errors=True)   # архив перезалили — старое убираем
            kept = _flatten(photos, UPLOAD_DIR / batch)
            board.note(f"архив {name}: фото {len(kept)} -> батч «{batch}»")
            batches.append((batch, UPLOAD_DIR / batch))
        elif suffix in IMG_EXT:
            loose_images.append(path)
        else:
            board.warn(f"файл {name} пропущен: поддерживаются .jpg/.jpeg/.png и .zip")
    if loose_images:
        batch = "photos_" + datetime.now().strftime("%Y-%m-%d_%H-%M")
        shutil.rmtree(UPLOAD_DIR / batch, ignore_errors=True)
        kept = _flatten(loose_images, UPLOAD_DIR / batch)
        board.note(f"отдельные фото: {len(kept)} -> батч «{batch}»")
        batches.append((batch, UPLOAD_DIR / batch))
    if not batches:
        raise RuntimeError("В загруженных файлах не нашлось ни одного фото .jpg/.jpeg/.png. "
                           "Загрузи фото или .zip с фото.")
    total = sum(len(list(p.iterdir())) for _, p in batches)
    board.note(f"итого к обработке: {total} фото в {len(batches)} батч(ах)")
    (UPLOAD_DIR / "_last_upload.json").write_text(json.dumps(
        {"updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
         "batches": [{"name": name, "path": str(path)} for name, path in batches]},
        ensure_ascii=False, indent=1), encoding="utf-8")
    return batches


# --------------------------------------------------------------------------- обработка
def process(board, output_root=OUTPUT_DEFAULT, auto_upload=True, batches=None,
            skip_collect=False, max_photos=0, reuse_last=False):
    """Загрузка кнопкой → план → реконструкция каждого фото → индекс → сводка."""
    out = drive_path(output_root)
    out.mkdir(parents=True, exist_ok=True)
    STOP_FILE.unlink(missing_ok=True)
    board.stage("Загрузка файлов", 1, 4)
    batches = collect_uploads(board, reuse=reuse_last) if auto_upload else (batches or [])
    if not batches:
        raise RuntimeError("Нет фото на входе.")

    results = []
    for index, (batch, folder) in enumerate(batches, start=1):
        prefix = f"[{index}/{len(batches)}] {batch}: " if len(batches) > 1 else ""
        write_config(input=str(folder), output=str(out),
                     max_photos_per_run=int(max_photos) or CONFIG_BASE["max_photos_per_run"])
        board.stage(f"{prefix}план: даты, лица, порядок", 2, 4)
        plan_rc = run_stream(board, [str(ENV_PYTHON), "-u", str(PIPELINE), "plan"],
                             f"{prefix}План обработки", check=False)
        if plan_rc:
            board.warn(f"{batch}: план не собрался (код {plan_rc}) — батч пропущен")
            continue
        board.stage(f"{prefix}реконструкция: одна FLAME-модель на фото", 3, 4)
        board.set(stats="на каждое фото: предобработка → нормали → UV → трекинг → сохранение на Drive")
        run_rc = run_stream(board, [str(ENV_PYTHON), "-u", str(PIPELINE), "run"],
                            f"{prefix}Реконструкция фото", check=False)
        if run_rc:
            board.warn(f"{batch}: часть фото завершилась с ошибкой (код {run_rc}); "
                       f"готовые результаты сохранены, повторный запуск досчитает остальные")
        if board.stop_requested:
            board.note("остановлено по кнопке — пересчёт продолжится с неготовых фото")
            break
        results.append(batch)

    if not skip_collect:
        board.stage("Индекс и сводка", 4, 4)
        if ENV_PYTHON.is_file():
            run_stream(board, [str(ENV_PYTHON), "-u", str(PIPELINE), "collect"],
                       "Сборка индекса", check=False)
    summary = status(output_root)
    board.set(phase="Результаты на Google Drive", current=str(out),
              stats=f"готово {summary['totals']['done']} из {summary['totals']['photos']} фото")
    board.state = "warning" if (summary["totals"]["failed"] or board.warning) else "done"
    return summary


# --------------------------------------------------------------------------- статус
def _read_json(path):
    try:
        return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def status(output_root=OUTPUT_DEFAULT):
    """Что уже есть на Drive: по батчам и по каждому фото. Не требует GPU и окружения."""
    out = drive_path(output_root)
    log = {}
    log_file = out / "_state/photos_log.jsonl"
    if log_file.is_file():
        for line in log_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if isinstance(record, dict) and record.get("photo"):
                log[(record.get("out") or record.get("id") or "", record["photo"])] = record
    batches, totals = {}, {"photos": 0, "done": 0, "failed": 0, "bytes": 0}
    if out.is_dir():
        for folder in sorted(p for p in out.iterdir() if p.is_dir() and not p.name.startswith("_")):
            photos_dir = folder / "photos"
            entry = {"name": folder.name, "path": str(folder), "photos": [], "done": 0, "failed": 0}
            if photos_dir.is_dir():
                for photo in sorted(p for p in photos_dir.iterdir() if p.is_dir()):
                    files = [p for p in photo.iterdir() if p.is_file()]
                    size = sum(p.stat().st_size for p in files)
                    done = (photo / "done.json").is_file() and (photo / "mesh.ply").is_file()
                    params = _read_json(photo / "params.json")
                    record = log.get((folder.name, photo.name), {})
                    entry["photos"].append({
                        "name": photo.name, "path": str(photo), "done": done,
                        "files": len(files), "mb": round(size / 1e6, 1),
                        "sec": params.get("run", {}).get("sec") or record.get("sec"),
                        "bin": (params.get("pose") or {}).get("bin", ""),
                        "yaw": round((params.get("pose") or {}).get("yaw_deg", 0) or 0, 1),
                        "error": "" if done else str(record.get("err", "") or "нет done.json"),
                    })
                    totals["photos"] += 1
                    totals["bytes"] += size
                    if done:
                        entry["done"] += 1
                        totals["done"] += 1
                    else:
                        entry["failed"] += 1
                        totals["failed"] += 1
            batches[folder.name] = entry
    totals["mb"] = round(totals["bytes"] / 1e6, 1)
    return {"root": str(out), "batches": batches, "totals": totals,
            "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}


def status_html(result, max_rows=400):
    """Компактная таблица «батч → фото → статус» для вывода в ячейке."""
    esc = html.escape
    if not result["batches"]:
        return (f'<section style="font:14px system-ui;border-left:5px solid #a26a00;padding:10px 14px;'
                f'border-radius:8px;background:#fffdf5">Пока пусто: в {esc(result["root"])} нет '
                f'папок с результатами. Запусти ячейку 2 и загрузи фото.</section>')
    rows = []
    for batch, entry in result["batches"].items():
        rows.append(
            f'<tr><td colspan="5" style="padding:10px 8px 4px;font-weight:600;background:#f2f5f9">'
            f'{esc(batch)} — готово {entry["done"]}, проблем {entry["failed"]} · {esc(entry["path"])}</td></tr>')
        for photo in entry["photos"][:max_rows]:
            mark = "✅" if photo["done"] else "⚠️"
            color = "#1a7f4b" if photo["done"] else "#a26a00"
            rows.append(
                f'<tr style="border-top:1px solid #eceff3">'
                f'<td style="padding:4px 8px">{mark} {esc(photo["name"])}</td>'
                f'<td style="padding:4px 8px;color:{color}">{"готово" if photo["done"] else esc(photo["error"])[:60]}</td>'
                f'<td style="padding:4px 8px">{photo["files"]} файлов · {photo["mb"]} МБ</td>'
                f'<td style="padding:4px 8px">{esc(str(photo["bin"]))} · yaw {photo["yaw"]}°</td>'
                f'<td style="padding:4px 8px">{human(photo["sec"]) if photo["sec"] else "—"}</td></tr>')
    totals = result["totals"]
    return (
        f'<section style="font:13px/1.5 -apple-system,system-ui,Segoe UI,Roboto,sans-serif;color:#111">'
        f'<div style="font-weight:600;font-size:15px;margin-bottom:6px">'
        f'Готово {totals["done"]} из {totals["photos"]} фото · {totals["mb"]} МБ · '
        f'<span style="font-weight:400;color:#555">{esc(result["root"])}</span></div>'
        f'<table style="border-collapse:collapse;width:100%;max-width:100%">'
        f'<tr style="text-align:left;color:#555;font-size:12px">'
        f'<th style="padding:4px 8px">фото</th><th style="padding:4px 8px">статус</th>'
        f'<th style="padding:4px 8px">данные</th><th style="padding:4px 8px">ракурс</th>'
        f'<th style="padding:4px 8px">время</th></tr>{"".join(rows)}</table>'
        f'<div style="color:#666;font-size:12px;margin-top:6px">обновлено {esc(result["updated"])}; '
        f'подробности по каждому фото — в params.json и done.json внутри его папки</div></section>')


def legend_html():
    """Что именно извлекается в папку каждого фото."""
    rows = "".join(
        f'<tr style="border-top:1px solid #eceff3"><td style="padding:3px 10px 3px 0;'
        f'font-family:ui-monospace,Menlo,monospace;white-space:nowrap">{html.escape(name)}</td>'
        f'<td style="padding:3px 0">{html.escape(text)}</td></tr>'
        for name, text in PHOTO_LEGEND)
    return (f'<section style="font:13px/1.5 -apple-system,system-ui,Segoe UI,Roboto,sans-serif">'
            f'<div style="font-weight:600;margin-bottom:4px">Что лежит в папке каждого фото</div>'
            f'<table style="border-collapse:collapse">{rows}</table></section>')


def preview_last(output_root=OUTPUT_DEFAULT, count=1):
    """Показать исходник, кроп и наложение последнего готового фото прямо в ноутбуке."""
    from IPython.display import HTML, Image, display
    result = status(output_root)
    photos = [p for entry in result["batches"].values() for p in entry["photos"] if p["done"]]
    if not photos:
        display(HTML('<i>готовых фото пока нет</i>'))
        return
    for photo in photos[-count:]:
        folder = pathlib.Path(photo["path"])
        blocks = []
        for name, title in (("overlay.jpg", "меш поверх фото"), ("crop.jpg", "кроп"),
                            ("segmentation.png", "сегментация"), ("normals.png", "нормали"),
                            ("uv_map.png", "UV-развёртка")):
            path = folder / name
            if path.is_file():
                data = base64.b64encode(path.read_bytes()).decode()
                mime = "image/png" if path.suffix == ".png" else "image/jpeg"
                blocks.append(f'<figure style="margin:0;text-align:center">'
                              f'<img src="data:{mime};base64,{data}" style="width:190px;border-radius:6px">'
                              f'<figcaption style="font-size:11px;color:#555">{title}</figcaption></figure>')
        display(HTML(
            f'<div style="font:13px system-ui;margin-top:8px"><b>{html.escape(photo["name"])}</b> · '
            f'{html.escape(photo["path"])}</div>'
            f'<div style="display:flex;gap:10px;flex-wrap:wrap;margin-top:6px">{"".join(blocks)}</div>'))


# --------------------------------------------------------------------------- скрипты окружения
# Ниже — те же скрипты, что были в v2 (сборка окружения и весов). Они вынесены в отдельные
# файлы, чтобы ноутбук состоял из трёх ячеек, а не из пятнадцати.

ENV_SCRIPT = r'''
set -euo pipefail
trap 'printf "ОШИБКА: строка %s: %s\n" "$LINENO" "$BASH_COMMAND" >&2' ERR
export MAMBA_ROOT_PREFIX=/content/micromamba-root PYTHONUNBUFFERED=1
export PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_PREFER_BINARY=1
ENV=/content/p3dmm-env
PY=$ENV/bin/python
READY=$ENV/.pixel3dmm_setup_v2
PIXEL_REV=fcd1fa973c7715b02a8948dfc679dff53cf85924
MICA_REV=af22e7a5810d474bc28a1433db533723d6bd2b07
FACER_REV=ddd35c76ff840174b8a5403ad1c1255e37b8782b
PIPNET_REV=b9eab58816437403a34aa5bc3adeafe5081fd36b

phase() {
  printf '__P3DMM_STATUS__{"phase":"%s","completed":%d,"total":12}\n' "$2" "$1"
}

configure_toolchain() {
  export CONDA_PREFIX=$ENV CUDA_HOME=$ENV
  export PATH="$ENV/bin:$PATH" LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
  export CC="$ENV/bin/x86_64-conda-linux-gnu-gcc" CXX="$ENV/bin/x86_64-conda-linux-gnu-g++"
  export CUDAHOSTCXX="$CXX" FORCE_CUDA=1
  export MAX_JOBS
  MAX_JOBS=$(python3 -c 'import os; gb=os.sysconf("SC_AVPHYS_PAGES")*os.sysconf("SC_PAGE_SIZE")/2**30; print(max(1,min(2,os.cpu_count() or 2,int((gb-2)/3))))')
  export TORCH_CUDA_ARCH_LIST
  TORCH_CUDA_ARCH_LIST=$("$PY" -c 'import torch; a=torch.cuda.get_device_capability(); print("%d.%d" % a if a <= (9,0) else "8.9+PTX")')
  printf 'CUDA arch: %s | потоков компиляции: %s\n' "$TORCH_CUDA_ARCH_LIST" "$MAX_JOBS"
}

verify_environment() {
  "$PY" - <<'PY'
import sys
import numpy, torch, torchvision, chumpy, cv2, facer, insightface, pixel3dmm
from pytorch3d.ops import knn_points
import nvdiffrast.torch as dr
assert sys.version_info[:2] == (3, 9), sys.version
assert numpy.__version__ == "1.23.0", numpy.__version__
assert torch.__version__ == "2.7.0+cu118", torch.__version__
assert torchvision.__version__ == "0.22.0+cu118", torchvision.__version__
assert torch.cuda.is_available(), "Выбери GPU T4/L4 в настройках Colab"
points = torch.zeros((1, 1, 3), device="cuda", requires_grad=True)
knn_points(points, points, K=1).dists.sum().backward()
vertices = torch.tensor([[[-.5, -.5, 0., 1.], [.5, -.5, 0., 1.], [0., .5, 0., 1.]]],
                        device="cuda", requires_grad=True)
faces = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
raster, _ = dr.rasterize(dr.RasterizeCudaContext(), vertices, faces, resolution=[4, 4])
raster.sum().backward()
torch.cuda.synchronize()
print("Окружение проверено:", torch.__version__, torch.cuda.get_device_name(0), flush=True)
PY
}

phase 0 'Проверка GPU'
if ! nvidia-smi; then
  printf 'Нужен GPU: Runtime -> Change runtime type -> T4/L4\n' >&2
  exit 1
fi

phase 1 'Системные библиотеки: установка только отсутствующих'
missing=()
for pkg in unzip libgl1 libglib2.0-0 build-essential; do
  dpkg -s "$pkg" >/dev/null 2>&1 || missing+=("$pkg")
done
if [ ${#missing[@]} -gt 0 ]; then
  apt-get update -qq
  apt-get install -y -qq "${missing[@]}"
fi

phase 2 'Восстановление сохраненного окружения с Drive'
if [ ! -x "$PY" ] || [ ! -f /content/pixel3dmm/scripts/run_cropping.py ]; then
  python3 -u /content/p3dmm_pipeline.py warm || printf 'Полного кеша нет: собираю недостающее\n'
fi

phase 3 'Проверка существующего окружения без переустановки'
if [ -x "$PY" ] && [ "$(git -C /content/pixel3dmm rev-parse HEAD 2>/dev/null || true)" = "$PIXEL_REV" ]; then
  if configure_toolchain && verify_environment && "$PY" -m pip check; then
    touch "$READY"
    phase 12 'Окружение готово: установка пропущена'
    exit 0
  fi
fi

phase 4 'Micromamba и Python 3.9'
mkdir -p /content/micromamba-bin
[ -x /content/micromamba-bin/bin/micromamba ] || curl -LsSf https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C /content/micromamba-bin bin/micromamba
MM=/content/micromamba-bin/bin/micromamba
[ -x "$PY" ] || "$MM" create -y -p "$ENV" -c conda-forge --override-channels python=3.9 pip
"$PY" -c 'import sys; assert sys.version_info[:2] == (3, 9), "Другой Python в p3dmm-env: перезапусти runtime"'

phase 5 'CUDA 11.8 и GCC 11: совместимый инструмент сборки'
if ! "$ENV/bin/x86_64-conda-linux-gnu-g++" -dumpversion 2>/dev/null | grep -Eq '^11(\.|$)' \
   || ! "$ENV/bin/nvcc" --version 2>/dev/null | grep -q 'release 11.8' \
   || [ ! -s "$ENV/include/cuda_runtime.h" ]; then
  "$MM" install -y -p "$ENV" --override-channels -c nvidia/label/cuda-11.8.0 -c conda-forge cuda-nvcc=11.8 cuda-cccl=11.8 cuda-cudart cuda-cudart-dev libcusparse libcusparse-dev libcublas libcublas-dev libcurand libcurand-dev libcusolver libcusolver-dev gcc_linux-64=11 gxx_linux-64=11
fi

phase 6 'PyTorch cu118 и базовые зависимости'
"$PY" -c 'import torch, torchvision; assert torch.__version__ == "2.7.0+cu118" and torchvision.__version__ == "0.22.0+cu118"' 2>/dev/null \
  || "$PY" -m pip install "torch==2.7.0" "torchvision==0.22.0" --index-url https://download.pytorch.org/whl/cu118
configure_toolchain
"$PY" -m pip install "numpy==1.23.0" "setuptools<81" ninja packaging fvcore iopath gdown "Cython<3"

clone_repo() {
  local url=$1 path=$2 revision=$3
  if [ ! -e "$path" ]; then
    git init -q "$path"
    git -C "$path" remote add origin "$url"
  fi
  if [ "$(git -C "$path" rev-parse HEAD 2>/dev/null || true)" != "$revision" ]; then
    git -C "$path" fetch -q --depth 1 origin "$revision"
    git -C "$path" checkout -q --detach "$revision"
  fi
}

phase 7 'Загрузка только закрепленных ревизий исходного кода'
clone_repo https://github.com/SimonGiebenhain/pixel3dmm.git /content/pixel3dmm "$PIXEL_REV"
BASE=/content/pixel3dmm/src/pixel3dmm/preprocessing
clone_repo https://github.com/FacePerceiver/facer.git "$BASE/facer" "$FACER_REV"
clone_repo https://github.com/Zielon/MICA.git "$BASE/MICA" "$MICA_REV"
clone_repo https://github.com/jhb86253817/PIPNet.git "$BASE/PIPNet" "$PIPNET_REV"
cp -f "$BASE/replacement_code/farl.py" "$BASE/facer/facer/face_parsing/farl.py"
cp -f "$BASE/replacement_code/facer_transform.py" "$BASE/facer/facer/transform.py"
cp -f "$BASE/replacement_code/mica_demo.py" "$BASE/MICA/demo.py"
cp -f "$BASE/replacement_code/mica.py" "$BASE/MICA/micalib/models/mica.py"

phase 8 'PyTorch3D: CUDA-сборка только если проверка не проходит'
"$PY" -c 'import torch; from pytorch3d.ops import knn_points; p=torch.zeros((1,1,3),device="cuda"); knn_points(p,p,K=1)' 2>/dev/null \
  || "$PY" -m pip install --force-reinstall --no-deps --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47"

phase 9 'Nvdiffrast: установка только если CUDA-проверка не проходит'
"$PY" -c 'import nvdiffrast.torch as dr; dr.RasterizeCudaContext()' 2>/dev/null \
  || "$PY" -m pip install --force-reinstall --no-deps --no-build-isolation "git+https://github.com/NVlabs/nvdiffrast.git@v0.3.3"

phase 10 'Зависимости Pixel3DMM и FaceR без замены PyTorch'
printf 'torch==2.7.0+cu118\ntorchvision==0.22.0+cu118\nnumpy==1.23.0\nopencv-python<4.12\nopencv-python-headless<4.12\nCython<3\nsetuptools<81\n' > /content/p3dmm-constraints.txt
"$PY" -m pip install --no-build-isolation -c /content/p3dmm-constraints.txt -r /content/pixel3dmm/requirements.txt -r "$BASE/facer/requirements.txt"
"$PY" -m pip install --no-build-isolation --no-deps -e /content/pixel3dmm
"$PY" -m pip install --no-build-isolation --no-deps -e "$BASE/facer"

phase 11 'Итоговая проверка импортов и реальных CUDA-операций'
"$PY" -m pip check
verify_environment
touch "$READY"
phase 12 'Окружение готово'
'''

WEIGHTS_SCRIPT = r'''
import json, pickle, shutil, stat, subprocess, sys, zipfile
from pathlib import Path, PurePosixPath
import numpy as np
sys.path.insert(0, "/content")
from p3dmm_pipeline import (CODE, DRIVE_ROOT, FLAME_DATA, PIPNET, HF_FILES, HF_REV,
                            apply_env, load_config, progress, verified_asset, write_json)
apply_env(load_config())
BASE = CODE / "src/pixel3dmm/preprocessing"
MICA_REV = "af22e7a5810d474bc28a1433db533723d6bd2b07"
HF = f"https://huggingface.co/alexnasa/pixel3dmm/resolve/{HF_REV}"
completed, total = 0, 21


def step(label, function, *args, **kwargs):
    global completed
    progress(phase=label, current="", completed=completed, total=total)
    function(*args, **kwargs)
    completed += 1
    print("OK:", label, flush=True)
    progress(completed=completed, total=total)


def download(name, target):
    target = Path(target)
    if verified_asset(name, target):
        print("Проверенный файл уже есть:", name, flush=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    subprocess.run(["curl", "--fail", "--location", "--retry", "3", "--connect-timeout", "30",
                    "--speed-time", "60", "--speed-limit", "1024", "--progress-bar",
                    "--output", str(partial), f"{HF}/{name}"], check=True)
    if not verified_asset(name, partial):
        raise ValueError(f"Неверный размер/SHA256 {name}; повтори ячейку 4")
    partial.replace(target)
    verified_asset(name, target)


def valid_model(path):
    if not path.is_file():
        return False
    try:
        # Only the user's licensed FLAME archives are deserialized.
        with path.open("rb") as stream:
            model = pickle.load(stream, encoding="latin1")
        vertices, directions = np.asarray(model["v_template"]), np.asarray(model["shapedirs"])
        return (vertices.shape == (5023, 3) and directions.shape[:2] == (5023, 3)
                and directions.shape[2] >= 400 and np.isfinite(vertices).all()
                and np.isfinite(directions).all())
    except Exception:
        return False


def install_flame(version, filename):
    source, target = DRIVE_ROOT / (version + ".zip"), FLAME_DATA / version / filename
    if not source.is_file():
        raise FileNotFoundError(f"Нет {source}; FLAME берется только из корня MyDrive")
    stamp_path = CODE / ".flame_sources.json"
    try:
        stamp = json.loads(stamp_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        stamp = {}
    if not isinstance(stamp, dict):
        stamp = {}
    source_meta = [source.stat().st_size, source.stat().st_mtime_ns]
    previous = stamp.get(version) or {}
    if previous.get("source") == source_meta and target.is_file():
        target_meta = [target.stat().st_size, target.stat().st_mtime_ns]
        if previous.get("target") == target_meta or valid_model(target):
            print("FLAME уже распакован из этого архива и проверен:", version, flush=True)
            stamp[version] = {"source": source_meta, "target": target_meta}
            write_json(stamp, stamp_path)
            return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    with zipfile.ZipFile(source) as archive:
        for info in archive.infolist():
            path = PurePosixPath(info.filename)
            if path.is_absolute() or ".." in path.parts or stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError(f"Опасный путь в FLAME-архиве: {info.filename}")
        members = [info for info in archive.infolist() if PurePosixPath(info.filename).name == filename
                   and "__MACOSX" not in PurePosixPath(info.filename).parts and not info.is_dir()]
        if len(members) != 1:
            found = [info.filename for info in archive.infolist() if info.filename.lower().endswith(".pkl")]
            raise ValueError(f"В {source.name} нужен ровно один {filename}. Найдены: {found}. "
                             "Нужен обычный FLAME2023, не Open; другая модель не подставляется.")
        with archive.open(members[0]) as incoming, partial.open("wb") as outgoing:
            shutil.copyfileobj(incoming, outgoing, length=8 * 1024 * 1024)
    if not valid_model(partial):
        raise ValueError(f"{source.name} не содержит корректную модель {filename}")
    partial.replace(target)
    stamp[version] = {"source": source_meta, "target": [target.stat().st_size, target.stat().st_mtime_ns]}
    write_json(stamp, stamp_path)


def restore_mica_asset(relative, minimum):
    target = FLAME_DATA / "FLAME2020" / relative
    content = subprocess.run(["git", "-C", str(BASE / "MICA"), "show",
                              f"{MICA_REV}:data/FLAME2020/{relative}"],
                             stdout=subprocess.PIPE, check=True).stdout
    if len(content) < minimum:
        raise ValueError(f"Неполный файл в MICA: {relative}")
    if target.is_file() and target.read_bytes() == content:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    partial.write_bytes(content)
    partial.replace(target)


step("FLAME2020 из MyDrive/FLAME2020.zip", install_flame, "FLAME2020", "generic_model.pkl")
step("FLAME2023 из MyDrive/FLAME2023.zip", install_flame, "FLAME2023", "flame2023_no_jaw.pkl")
for relative, minimum in (("landmark_embedding.npy", 31292), ("FLAME_masks/FLAME_masks.pkl", 215062),
                          ("head_template.obj", 10000)):
    step(relative, restore_mica_asset, relative, minimum)
targets = {"mica.tar": FLAME_DATA / "pretrained/mica.tar",
           "epoch59.pth": PIPNET / "snapshots/WFLW/pip_32_16_60_r18_l2_l1_10_1_nb10/epoch59.pth",
           "uv.ckpt": CODE / "pretrained_weights/uv.ckpt", "normals.ckpt": CODE / "pretrained_weights/normals.ckpt"}
for name, target in targets.items():
    step(name, download, name, target)
for name in HF_FILES:
    if name.startswith("insightface/"):
        step(name, download, name, Path.home() / ("." + name))


def build_faceboxes():
    if list((PIPNET / "FaceBoxesV2/utils").glob("*nms*.so")):
        print("FaceBoxes уже собран", flush=True)
        return
    subprocess.run([sys.executable, "build.py", "build_ext", "--inplace"],
                   cwd=PIPNET / "FaceBoxesV2/utils", check=True)


def verify_cuda():
    import torch, chumpy, cv2, facer, insightface, pixel3dmm
    from pytorch3d.ops import knn_points
    import nvdiffrast.torch as dr
    points = torch.zeros((1, 1, 3), device="cuda")
    knn_points(points, points, K=1)
    dr.RasterizeCudaContext()


step("FaceBoxes CPU NMS", build_faceboxes)
step("Импорты, PyTorch3D CUDA и nvdiffrast JIT", verify_cuda)

# Заранее загружаем lazy-веса FaceR, чтобы они попали в кеш среды.
prewarm_code = r"""
from pathlib import Path
import gc
import torch
import torch.utils.model_zoo
import facer
from facer.face_parsing import farl
base = Path("/content/pixel3dmm/src/pixel3dmm/preprocessing")
assert Path(farl.__file__).read_bytes() == (base / "replacement_code/farl.py").read_bytes(), "Загружен facer без нужного replacement"
cache = Path(torch.hub.get_dir()) / "checkpoints"
assert cache.resolve() == (Path.home() / ".cache/torch/hub/checkpoints").resolve(), cache
detector = facer.face_detector("retinaface/mobilenet", device="cuda")
parser = facer.face_parser("farl/celebm/448", device="cuda")
del detector, parser
gc.collect()
torch.cuda.empty_cache()
"""
exec(compile(prewarm_code, "prewarm_facer.py", "exec"))
print("Lazy-веса FaceR загружены в кеш", flush=True)

print("Все веса и сборки готовы", flush=True)
'''


def write_scripts():
    """Разложить ENV_SCRIPT/WEIGHTS_SCRIPT по файлам в /content."""
    ENV_SH.write_text(ENV_SCRIPT, encoding="utf-8")
    WEIGHTS_PY.write_text(WEIGHTS_SCRIPT, encoding="utf-8")
    return ENV_SH, WEIGHTS_PY
