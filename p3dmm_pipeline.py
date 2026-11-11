# -*- coding: utf-8 -*-
"""Colab pipeline v3: одна примерка на фото, понятная очередь и состояние на Google Drive.

Отличия от v2 (для ноутбука «одна кнопка»):
  * вход может быть локальным каталогом (кнопка загрузки в Colab), а не только папкой MyDrive;
  * если в имени файла нет даты, она берётся из EXIF, иначе из времени файла — фото больше
    не выбрасываются молча;
  * flat=True кладёт результаты прямо в OUTPUT/<батч>/photos/<фото>/ без служебной папки batches;
  * в цикле трекинга печатаются структурированные прогресс-события для живой панели статуса.

Ноутбук запускает plan/status/collect без GPU, а preflight/run — в изолированном окружении.
Готовые папки фото переживают сброс рантайма; freeze/warm кешируют установку, а не результаты.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import posixpath
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

ENV_PY = "/content/p3dmm-env/bin/python"
ENV_ROOT = Path("/content/p3dmm-env")
CODE = Path("/content/pixel3dmm")
PREPRO = Path("/content/p3dmm/preprocessed")
TRACK = Path("/content/p3dmm/tracking")
STAGE = Path("/content/stage")
LOGS = Path("/content/logs")
WORK = Path("/content/work")
CFG_FILE = Path("/content/p3dmm_config.json")
PLAN_FILE = Path("/content/work/plan.json")
LOCAL_LOG = Path("/content/work/batch_log.jsonl")
PHOTO_LOG = Path("/content/work/photos_log.jsonl")     # unit=photo: журнал по каждому фото
STOP_FILE = Path("/content/STOP")
DRIVE_ROOT = Path("/content/drive/MyDrive")
FLAME_DATA = CODE / "src/pixel3dmm/preprocessing/MICA/data"
PIPNET = CODE / "src/pixel3dmm/preprocessing/PIPNet"
IMG_EXT = (".jpg", ".jpeg", ".png")
VIDEO_EXT = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")
# обязательные файлы FLAME: без них трекинг падает сразу после инициализации
REQUIRED_FLAME2020 = ("generic_model.pkl", "landmark_embedding.npy",
                      "FLAME_masks/FLAME_masks.pkl")
FLAME2023_PKL = "FLAME2023/flame2023_no_jaw.pkl"
HF_REV = "7f5141a5edb10fa57caa705e8cb029f44630f788"
HF_FILES = {
    "generic_model.pkl": (53_023_716, "efcd14cc4a69f3a3d9af8ded80146b5b6b50df3bd74cf69108213b144eba725b"),
    "mica.tar": (502_647_880, "4542a467d9e8f7521474a1d00eac89552bebef0b331b72bf7fbd6f065ff64d7b"),
    "epoch59.pth": (49_393_015, "44002daaf3187e6e4b99e0fd9a52f60ffa05e06ec5df3721ace950fd2f2cc120"),
    "uv.ckpt": (2_246_794_577, "dff9d73feec47914b704759f57ebffb8c58d2aef550b426013fe31eae21707b8"),
    "normals.ckpt": (1_469_022_184, "e856799d55db54c7537c8ee3c5a4938c13cc0b24082ce7e4e7f35f0d0f0e28da"),
}
for _model in ("antelopev2", "buffalo_l"):
    for _name, _size, _sha in (
        ("1k3d68.onnx", 143_607_619, "df5c06b8a0c12e422b2ed8947b8869faa4105387f199c477af038aa01f9a45cc"),
        ("2d106det.onnx", 5_030_888, "f001b856447c413801ef5c42091ed0cd516fcd21f2d6b79635b1e733a7109dbf"),
        ("genderage.onnx", 1_322_532, "4fde69b1c810857b88c64a335084f1c3fe8f01246c9a191b48c7bb756d6652fb"),
        ("scrfd_10g_bnkps.onnx" if _model == "antelopev2" else "det_10g.onnx", 16_923_827,
         "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91"),
    ):
        HF_FILES[f"insightface/models/{_model}/{_name}"] = (_size, _sha)
HF_FILES["insightface/models/antelopev2/glintr100.onnx"] = (
    260_665_334, "4ab1d6435d639628a6f3e5008dd4f929edf4c4124b1a7169e1048f9fef534cdf")
HF_FILES["insightface/models/buffalo_l/w600k_r50.onnx"] = (
    174_383_860, "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43")
WEIGHTS = (("pretrained_weights/uv.ckpt", HF_FILES["uv.ckpt"][0]),
           ("pretrained_weights/normals.ckpt", HF_FILES["normals.ckpt"][0]),
           (str(FLAME_DATA.relative_to(CODE)) + "/pretrained/mica.tar", HF_FILES["mica.tar"][0]),
           ("src/pixel3dmm/preprocessing/PIPNet/snapshots/WFLW/pip_32_16_60_r18_l2_l1_10_1_nb10/epoch59.pth", HF_FILES["epoch59.pth"][0]))
# что из предобработки уезжает в Drive вместе с результатом
PREPROC_SUBSET = ("cropped", "seg_og", "PIPnet_landmarks", "p3dmm/normals", "p3dmm/uv_map")
# unit=photo: (имя файла в папке фото, подпапка предобработки, расширение) — всё по одному снимку
PHOTO_PREPROC = (("crop.jpg", "cropped", ".jpg"),
                 ("segmentation.png", "seg_og", ".png"),
                 ("landmarks_pipnet.npy", "PIPnet_landmarks", ".npy"),
                 ("normals.png", "p3dmm/normals", ".png"),
                 ("uv_map.png", "p3dmm/uv_map", ".png"),
                 ("arcface.npy", "arcface", ".npy"))
# MICA пишет на кадр отдельной папкой mica/<номер кадра>/ — это тоже данные фото
PHOTO_MICA = (("mica_identity.npy", "identity.npy"),
              ("mica_mesh.ply", "mesh.ply"),
              ("mica_kpt7.npy", "kpt7.npy"),
              ("mica_kpt68.npy", "kpt68.npy"))
# mica_demo.py берёт кадры с шагом len(cropped)//10: за один вызов больше 10 — и часть фото
# останется без своего mica_identity/arcface, поэтому недостающее предоббатываем группами по 10
MICA_GROUP = 10
# ядро, без которого трекер не читает кадр; остальное (arcface + MICA) даёт тот же вызов MICA
PHOTO_CORE = tuple(p for p in PHOTO_PREPROC if p[1] != "arcface")
# по этому набору решаем, что фото посчитано и папку можно пропускать
PHOTO_REQUIRED = ("mesh.ply", "canonical.ply", "params.json", "done.json")
# --- наблюдаемость лица на снимке: те же классы FaceR, из которых read_data() собирает свои маски ---
# без этой правки потом нельзя отличить «сторона не видна» от «сторона другая»
SEG_FG = (1, 2, 4, 5, 6, 7, 8, 9, 10, 12, 13)      # fg_mask: силуэт головы вообще
SEG_UV = (2, 6, 7, 10, 12, 13)                     # uv_mask: где реально работали UV-сопоставления
SEG_NORMAL = SEG_UV + (11,)                        # normal_mask: где работали нормали (big_normal_mask выкл)
MASK_SIZE = 256                                    # config.size — в этом разрешении трекер видел кадр
# PIPNet WFLW: первые 68 точек идут в порядке iBUG-68, дальше дополнительный контур
PIPNET_ZONES = ((range(0, 17), "jaw"), (range(17, 22), "brow_right"), (range(22, 27), "brow_left"),
                (range(27, 36), "nose"), (range(36, 42), "eye_right"), (range(42, 48), "eye_left"),
                (range(48, 60), "mouth_outer"), (range(60, 68), "mouth_inner"))
# ракурс считаем не «на глаз по 2D»: та же девятибинная таблица и та же математика, что в
# pixel3dmm_analysis.py (разбор корпуса), чтобы метка в params.json и метка при сравнении не разъехались
VIEW_BINS = (
    ("left_profile", -95.0, -50.0, -70.0),
    ("left_deep", -50.0, -40.0, -45.0),
    ("left_mid", -40.0, -25.0, -32.5),
    ("left_light", -25.0, -10.0, -17.5),
    ("frontal", -10.0, 10.0, 0.0),
    ("right_light", 10.0, 25.0, 17.5),
    ("right_mid", 25.0, 40.0, 32.5),
    ("right_deep", 40.0, 50.0, 45.0),
    ("right_profile", 50.0, 95.0, 70.0),
)
POSE_CONVENTION = "yaw positive nose-to-image-right; pitch positive up; roll camera-up"
# tracker.py: MAX_STEPS = min(len(cropped) - start_frame, 1000) — дальше кадры просто не пишутся
MAX_TRACK_FRAMES = 1000
# типичная независимая примерка одного фото на T4 (iters=800) — оценка до первого замера в журнале
PHOTO_DEFAULT_S = 90.0

DEFAULT_CONFIG = {
    "input": "MyDrive/input",
    "output": "MyDrive/output",
    "start_from": "",
    "export": "photos",
    "face_filter": True,
    "face_min_score": 0.75,
    "use_flame2023": "true",
    "torch_compile": False,
    "sync_to_drive": True,
    "single_iters": 200,
    "attempts": 3,
    "min_free_gb": 2.0,
    "session_max_hours": 12.0,
    "session_hard_limit_hours": 0.0,
    "batch_timeout_min": 90,
    "keep_preproc": True,
    "unit": "photo",
    "photos_per_chunk": 1,
    "max_photos_per_run": 500,
    "photo_timeout_min": 15,
    "keep_archive": False,
    # v3: результаты прямо в OUTPUT/<батч>/... и дата фото не только из имени файла
    "flat": False,
    "date_fallback": True,
}


# ----------------------------------------------------------------------------- база

def say(text=""):
    print(text, flush=True)


def progress(**values):
    """Structured updates are consumed by the notebook's persistent status panel."""
    if os.environ.get("P3DMM_NOTEBOOK_STATUS") == "1":
        say("__P3DMM_STATUS__" + json.dumps(values, ensure_ascii=False))
    else:
        text = " | ".join(str(values[k]) for k in ("phase", "current", "stats") if values.get(k))
        if text:
            say(text)


def load_config():
    if not CFG_FILE.is_file():
        raise SystemExit("нет /content/p3dmm_config.json — сначала выполни ячейку «1. ВХОД -> ВЫХОД»")
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(json.loads(CFG_FILE.read_text(encoding="utf-8")))
    if cfg["unit"] != "photo" or cfg["export"] != "photos" or cfg["keep_archive"]:
        raise ValueError("Ноутбук поддерживает только независимые фото без архивов результатов. Повтори ячейку 1.")
    return cfg


def apply_env(cfg):
    """Все переменные, в которые упирались ошибок: пути Pixel3DMM, компилятор nvdiffrast, weights_only."""
    os.environ["PIXEL3DMM_CODE_BASE"] = str(CODE) + "/"
    os.environ["PIXEL3DMM_PREPROCESSED_DATA"] = str(PREPRO) + "/"
    os.environ["PIXEL3DMM_TRACKING_OUTPUT"] = str(TRACK) + "/"
    os.environ.pop("TORCH_FORCE_WEIGHTS_ONLY_LOAD", None)
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
    os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5 8.9"
    os.environ["PATH"] = str(ENV_ROOT / "bin") + ":" + os.environ.get("PATH", "")
    os.environ["CONDA_PREFIX"] = str(ENV_ROOT)
    os.environ["CUDA_HOME"] = str(ENV_ROOT)
    os.environ["LD_LIBRARY_PATH"] = str(ENV_ROOT / "lib") + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    gcc = str(ENV_ROOT / "bin/x86_64-conda-linux-gnu-gcc")
    gxx = str(ENV_ROOT / "bin/x86_64-conda-linux-gnu-g++")
    if os.path.exists(gcc):
        os.environ["CC"] = gcc
        os.environ["CXX"] = gxx
        os.environ["CUDAHOSTCXX"] = gxx
    os.environ["MAX_JOBS"] = "2"
    try:
        import torch
        if torch.cuda.is_available():
            arch = torch.cuda.get_device_capability()
            os.environ["TORCH_CUDA_ARCH_LIST"] = f"{arch[0]}.{arch[1]}" if arch <= (9, 0) else "8.9+PTX"
    except ImportError:
        pass
    for d in (PREPRO, TRACK, STAGE, LOGS, WORK):
        d.mkdir(parents=True, exist_ok=True)


def patch_compile(enable):
    """COMPILE в tracker.py по умолчанию True -> torch.compile/triton. выключаем, если не просили иначе."""
    path = CODE / "src/pixel3dmm/tracking/tracker.py"
    if not path.is_file():
        return "(pixel3dmm ещё не склонирован)"
    text = path.read_text(encoding="utf-8", errors="ignore")
    want = "COMPILE = True" if enable else "COMPILE = False"
    fixed = re.sub(r"^COMPILE = (True|False)$", want, text, count=1, flags=re.M)
    if fixed != text:
        path.write_text(fixed, encoding="utf-8")
    return want


# upstream-трекер умеет начинать с start_frame, но число кадров ограничивает только длиной папки,
# а MICA берёт средним по всем кадрам. для unit="photo" это два blockers: прогон одного кадра
# превращается в прогон «всего, что осталось после start_frame», а форма инициализируется чужими лицами.
NUM_FRAMES_MARK = "# p3dmm-colab: num_frames"
MICA_FRAME_MARK = "# p3dmm-colab: per-frame mica"


def patch_single_frame_support():
    """Делает возможным независимую примерку ровно одного кадра (num_frames=1). Идемпотентен."""
    path = CODE / "src/pixel3dmm/tracking/tracker.py"
    if not path.is_file():
        return "нет tracker.py — выполни ячейку «3. Окружение»"
    text = path.read_text(encoding="utf-8", errors="ignore")
    applied = []
    if NUM_FRAMES_MARK not in text:
        anchor = "        self.BATCH_SIZE = min(self.config.batch_size, self.MAX_STEPS)"
        step = ("        " + NUM_FRAMES_MARK + "\n"
                "        if getattr(self.config, 'num_frames', 0):\n"
                "            self.MAX_STEPS = min(max(self.MAX_STEPS, 1), int(self.config.num_frames))\n")
        if anchor in text:
            text = text.replace(anchor, step + anchor, 1)
            applied.append("num_frames")
    if MICA_FRAME_MARK not in text:
        anchor = "            mica_shape = np.mean(mica_shapes, axis=0)"
        step = ("            " + MICA_FRAME_MARK + "\n"
                "            if getattr(self.config, 'num_frames', 0) and os.path.isdir("
                "f'{mica_folder}/{timestep:05d}'):\n"
                "                mica_shape = np.squeeze(np.load("
                "f'{mica_folder}/{timestep:05d}/identity.npy'))\n"
                "            elif getattr(self.config, 'num_frames', 0):\n"
                "                raise RuntimeError('нет собственной MICA identity для кадра ' + str(timestep))\n"
                "            else:\n"
                "                mica_shape = np.mean(mica_shapes, axis=0)")
        if anchor in text:
            text = text.replace(anchor, step, 1)
            applied.append("per-frame mica")
    if applied:
        path.write_text(text, encoding="utf-8")
    return ", ".join(applied) if applied else "уже на месте (num_frames + per-frame mica)"


def slug(text):
    """Имя батча должно пережить os.system без кавычек и парсинг OmegaConf CLI."""
    out = re.sub(r"[^0-9A-Za-z_-]", "_", str(text)).strip("_")
    return out[:60] or "x"


# OmegaConf.from_cli разбирает одиночное значение как yaml: 123 -> int, true -> bool, null -> None.
# такой id в video_name=... ломает пути (data_folder=<int>) -> добавляем суффикс.
CLI_UNSAFE = {"true", "false", "null", "none", "~", "yes", "no", "on", "off", ""}


def safe_id(text):
    out = slug(text)
    # консервативно: если id не начинается с буквы, yaml может распознать его как число (1_5e3 -> float)
    if not out[:1].isalpha() or out.lower() in CLI_UNSAFE:
        out += "_v"
    return out


def photo_folder(text):
    """Имя папки фото в Drive: оставляем исходное имя файла (в т.ч. кириллицу), режем только запрещённое."""
    stem = Path(str(text)).stem.strip()
    out = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", stem)[:60].strip(" .") or "frame"
    return out


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def free_gb():
    for probe in ("/content", "/tmp", "/"):
        try:
            return round(shutil.disk_usage(probe).free / 1e9, 1)
        except OSError:
            continue
    return -1.0


def human(sec):
    if sec < 90:
        return f"{sec:.0f} с"
    if sec < 5400:
        return f"{sec / 60:.1f} мин"
    return f"{sec / 3600:.1f} ч"


def run_cmd(cmd, log_path, timeout_min=None, cwd=None):
    """Keep the full child log and expose new output while long GPU steps are running."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(c) for c in cmd]
    if cmd[0] in ("python", "python3"):
        cmd[0] = ENV_PY
    seconds = None if not timeout_min else float(timeout_min) * 60
    started = time.monotonic()
    deadline = started + seconds if seconds else None
    with log_path.open("a", encoding="utf-8", errors="ignore") as fh:
        fh.write("\n$ " + " ".join(map(str, cmd)) + "\n")
        fh.flush()
        output_start = fh.tell()
        progress(phase="Выполняется " + Path(cmd[1]).name)
        try:
            child_env = dict(os.environ, PYTHONUNBUFFERED="1")
            proc = subprocess.Popen(cmd, cwd=str(cwd or CODE), stdout=fh, env=child_env,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            with log_path.open("r", encoding="utf-8", errors="replace") as live:
                live.seek(output_start)
                try:
                    while True:
                        remaining = deadline - time.monotonic() if deadline else 2.0
                        if remaining <= 0:
                            raise subprocess.TimeoutExpired(cmd, seconds)
                        try:
                            rc = proc.wait(timeout=min(2.0, remaining))
                        except subprocess.TimeoutExpired:
                            rc = None
                        output = live.read().replace("\r", "\n").splitlines()
                        for line in output[-3:]:
                            if line.strip():
                                say(line[:600])
                        if rc is not None:
                            break
                except (subprocess.TimeoutExpired, KeyboardInterrupt):
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    proc.wait()
                    raise
            reason = "" if rc == 0 else f"код {rc}"
        except subprocess.TimeoutExpired:
            rc, reason = -9, f"таймаут {timeout_min} мин"
        except OSError as error:
            rc, reason = 127, f"{type(error).__name__}: {error}"
        fh.write(f"=== {reason or 'ok'} ===\n")
    return rc, reason


def tail(path, lines=25):
    try:
        return "\n".join(Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()[-lines:])
    except OSError:
        return "(лога нет)"


# ----------------------------------------------------------------------------- вход

def resolve_input(raw):
    """Папка с фото: путь от MyDrive («MyDrive/input», «input») или абсолютный путь.

    Абсолютный путь разрешён, потому что ноутбук складывает загруженный архив и распакованные
    фото в /content/upload — гонять их через Drive было бы медленно и бессмысленно.
    Сканируется только указанный каталог, никогда весь MyDrive.
    """
    raw = str(raw or "").strip().strip('"').strip()
    if not raw:
        raise ValueError("INPUT пуст: нужна папка с фото (MyDrive/input или /content/upload/<батч>)")
    if raw.startswith("/"):
        target = Path(raw).resolve()
    else:
        name = raw[len("MyDrive/"):] if raw.startswith("MyDrive/") else raw
        target = (DRIVE_ROOT / name).resolve()
    if not target.is_dir():
        raise ValueError(f"не найдена папка с фото {target}; архивы и видео не являются входом")
    return target


def unpack_zip(archive, dest=None):
    import zipfile
    archive = Path(archive)
    if dest is None:
        digest = hashlib.sha256(str(archive.resolve()).encode()).hexdigest()[:10]
        dest = WORK / "input_photos" / digest
    dest = Path(dest)
    if archive.resolve().is_relative_to(dest.resolve()):
        raise ValueError("архив не должен лежать в папке распаковки")
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        for n in zf.namelist():
            if n.endswith("/") or Path(n).suffix.lower() not in IMG_EXT:
                continue
            if n.startswith("._") or "/._" in n or "__MACOSX" in n:
                continue
            target = (dest / n).resolve()
            if not str(target).startswith(str(dest.resolve()) + os.sep):
                raise ValueError("опасный путь в архиве: " + n)
            zf.extract(n, dest)
    kids = [p for p in dest.iterdir() if p.is_dir()]
    if len(kids) == 1 and not [p for p in dest.iterdir() if p.is_file()]:
        return kids[0]
    return dest


def is_photo(p):
    return p.is_file() and p.suffix.lower() in IMG_EXT and not p.name.startswith("._")


def photo_moment(path):
    """Return a sortable timestamp from the filename, or None when the date is missing/invalid."""
    stem = Path(path).stem
    patterns = (
        r"(?<!\d)(\d{4})([_\-.]?)(\d{2})\2(\d{2})(?!\d)",
        r"(?<!\d)(\d{2})([_\-.]?)(\d{2})\2(\d{4})(?!\d)",
    )
    match = None
    for pattern in patterns:
        match = re.search(pattern, stem)
        if match:
            break
    if not match:
        return None
    groups = match.groups()
    try:
        if len(groups[0]) == 4:
            year, month, day = int(groups[0]), int(groups[2]), int(groups[3])
        else:
            day, month, year = int(groups[0]), int(groups[2]), int(groups[3])
        date = datetime(year, month, day).date()
    except ValueError:
        return None
    time_match = re.search(r"(?<!\d)(\d{2})[:_\-]?(\d{2})[:_\-]?(\d{2})(?!\d)", stem[match.end():])
    if time_match:
        try:
            parsed_time = datetime.strptime("".join(time_match.groups()), "%H%M%S").time()
            return datetime.combine(date, parsed_time)
        except ValueError:
            pass
    return datetime.combine(date, datetime.min.time())


def photo_moment_fallback(path):
    """Дата, когда её нет в имени файла: EXIF DateTimeOriginal, иначе время изменения файла.

    v2 молча выбрасывал такие фото («без корректной даты не анализируются»), из-за чего
    загруженный архив с именами вида IMG_1234.jpg мог дать ноль результатов.
    """
    path = Path(path)
    raw = ""
    try:
        from PIL import Image
        with Image.open(path) as image:
            exif = image.getexif() or {}
            raw = str(exif.get(36867) or exif.get(306) or "")
    except Exception:
        raw = ""
    if raw:
        try:
            return datetime.strptime(raw.strip()[:19], "%Y:%m:%d %H:%M:%S"), "exif"
        except ValueError:
            pass
    try:
        return datetime.fromtimestamp(path.stat().st_mtime), "mtime"
    except OSError:
        return datetime.min, "none"


def photo_sort_moment(path):
    """Момент съёмки для хронологии: имя файла, иначе EXIF/mtime — как в scan_tree."""
    moment = photo_moment(path)
    if moment is None:
        moment, _ = photo_moment_fallback(path)
    return moment


def scan_tree(root, cfg):
    """Sort dated photos chronologically; undated photos are reported but never analyzed."""
    cfg = cfg or {}
    dated, undated, fallback = [], [], []
    for base, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d != "__MACOSX")
        d = Path(base)
        for name in files:
            if name.startswith(".") or Path(name).suffix.lower() not in IMG_EXT:
                continue
            path = d / name
            if not path.is_file():
                continue
            rel = str(path.relative_to(root))
            moment = photo_moment(path)
            if moment is None and cfg.get("date_fallback", True):
                moment, source = photo_moment_fallback(path)
                fallback.append({"relpath": rel, "source": source})
            if moment is None:
                undated.append(rel)
            else:
                dated.append((moment, path, rel))
    dated.sort(key=lambda item: (item[0], item[2]))
    start = str(cfg.get("start_from", "")).strip()
    if start:
        start_item = None
        for item in dated:
            if item[2] == start or Path(item[2]).name == start:
                start_item = item
                break
        if start_item is None:
            raise ValueError(f"START_FROM не найден среди датированных фото: {start!r}")
        # Фото за тот же день сохраняются все, даже если начальная точка указана
        # как один из вариантов "(2)" или "(3)" за эту дату.
        start_date = start_item[0].date()
        dated = [item for item in dated if item[0].date() >= start_date]
    cfg["_undated_photos"] = sorted(undated)
    cfg["_date_fallback"] = sorted(fallback, key=lambda item: item["relpath"])

    grouped = {}
    order = []
    for _, path, rel in dated:
        folder = path.parent
        if folder not in grouped:
            grouped[folder] = []
            order.append(folder)
        grouped[folder].append((path, rel))
    batches = []
    for folder in order:
        rel_parts = folder.relative_to(root).parts
        # фото лежат прямо в корне входа (так делает ноутбук после распаковки архива):
        # имя батча = имя папки, иначе получился бы безликий «flat__all»
        subject = slug(rel_parts[0]) if len(rel_parts) >= 1 else slug(root.name)
        angle = "__".join(slug(x) for x in rel_parts[1:]) if len(rel_parts) > 1 else "all"
        bid = safe_id(subject if len(rel_parts) <= 1 else f"{subject}__{angle}")
        batches.append({
            "id": bid,
            "subject": subject,
            "angle": angle,
            "kind": "photos",
            "folder": str(folder),
            "n": len(grouped[folder]),
            "frames": [{"orig": str(path), "relpath": rel} for path, rel in grouped[folder]],
        })
    counts = {}
    for b in batches:
        counts[b["id"]] = counts.get(b["id"], 0) + 1
    for b in batches:
        if counts[b["id"]] > 1:
            rel = str(Path(b["folder"]).relative_to(root))
            digest = hashlib.sha256(rel.encode()).hexdigest()[:10]
            b["id"] = f"{b['id']}__{digest}"
    cfg["_photo_count"] = len(dated)
    cfg["_undated_count"] = len(undated)
    return batches


def scan_single(root, cfg):
    """Один файл: видео или фото -> один батч."""
    stem = safe_id(root.stem)
    if root.suffix.lower() in IMG_EXT:
        return [{"id": f"{stem}__one", "subject": stem, "angle": "one", "kind": "photos",
                 "folder": str(root.parent), "n": 1,
                 "frames": [{"orig": str(root), "relpath": root.name}]}]
    if root.suffix.lower() not in VIDEO_EXT:
        raise ValueError(f"неподдерживаемый вход: {root.name}; нужны фото, zip или видео")
    return [{"id": stem, "subject": stem, "angle": "video", "kind": "video",
             "folder": str(root), "n": 0, "frames": []}]


def build_batches(cfg, refresh=True):
    root = resolve_input(cfg["input"])
    batches = scan_tree(root, cfg)
    if refresh:
        for b in batches:
            b["n"] = len(b["frames"])
            stage_photos(b)
    return [b for b in batches if b["n"] > 0]


def stage_video(b, root):
    import cv2
    capture = cv2.VideoCapture(str(root))
    try:
        if not capture.isOpened():
            raise ValueError(f"видео не читается: {root}")
        count = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    finally:
        capture.release()
    if count > MAX_TRACK_FRAMES:
        raise ValueError(f"видео содержит {count} кадров; разбей на части до {MAX_TRACK_FRAMES} перед plan")
    b["n"] = count
    dst = STAGE / (b["id"] + ".mp4")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if Path(root).resolve() != dst.resolve():
        shutil.copy2(root, dst)
    b["target"] = str(dst)
    return str(dst)


def stage_photos(b, keep=None):
    """Stage one batch's frames as numbered images."""
    from PIL import Image, ImageOps
    import numpy as np
    frames = b["frames"] if keep is None else [b["frames"][i] for i in keep]
    dst = STAGE / b["id"]
    if any(Path(item["orig"]).resolve().is_relative_to(dst.resolve()) for item in frames):
        raise ValueError("INPUT находится в папке staging; перенеси исходные фото за пределы /content/stage")
    shutil.rmtree(dst, ignore_errors=True)
    dst.mkdir(parents=True, exist_ok=True)
    kept = []
    for item in frames:
        i = len(kept)
        progress(phase="Подготовка изображений", current=item["relpath"],
                 stats=f"Группа: {i + 1}/{len(frames)}")
        try:
            with Image.open(item["orig"]) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                if min(im.size) < 64 or np.asarray(im).std() < 1.0:
                    item["note"] = (item.get("note", "") + " too_small_or_flat").strip()
        except Exception as error:
            item["note"] = (item.get("note", "") + f" unreadable:{type(error).__name__}").strip()
            b.setdefault("dropped", []).append(dict(item))
            continue
        im.save(dst / f"{i:05d}.jpg", quality=95)
        item["orig_frame"] = item.get("frame")
        item["frame"] = f"{i:05d}"
        kept.append(item)
    b["frames"] = kept
    b["n"] = len(kept)
    b["target"] = str(dst)
    write_manifest(b)
    return str(dst)


MANIFEST_COLS = ("frame", "subject", "angle", "batch", "orig_relpath", "orig_path", "face_score", "note")
# плоские колонки кадра: один список на frames.csv батча и на сводный _index/frames_all.csv, иначе
# ракурс (yaw/pitch/roll + метка из VIEW_BINS) появится в одном месте и пропадёт в другом
FRAMES_CSV_COLS = ["frame", "batch", "subject", "angle", "orig_relpath", "orig_path", "face_score",
                   "mesh_ply", "frame_file", "jaw_norm", "exp_norm", "shape_norm", "t", "fl", "pp",
                   "yaw_deg", "pitch_deg", "roll_deg", "view_bin", "note"]
FRAMES_ALL_CSV_COLS = ["batch"] + [c for c in FRAMES_CSV_COLS if c not in ("batch", "note")]


def manifest_path(b):
    return WORK / "manifest" / f"{b['id']}.csv"


def write_manifest(b):
    path = manifest_path(b)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_COLS)
        writer.writeheader()
        if b["kind"] == "video":
            for i in range(b.get("n_real", 0)):
                writer.writerow({"frame": f"{i:05d}", "subject": b["subject"], "angle": "video",
                                 "batch": b["id"], "orig_relpath": f"{Path(b['folder']).name}#frame={i}",
                                 "orig_path": b["folder"], "face_score": "", "note": "video_frame"})
        else:
            for item in b["frames"]:
                writer.writerow({"frame": item["frame"], "subject": b["subject"], "angle": b["angle"],
                                 "batch": b["id"], "orig_relpath": item["relpath"],
                                 "orig_path": item["orig"], "face_score": item.get("score", ""),
                                 "note": item.get("note", "")})


def read_manifest(bid):
    path = manifest_path({"id": bid})
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as fh:
        return {row["frame"]: row for row in csv.DictReader(fh)}


# ------------------------------------------------------------------- FaceBoxes фильтр

def make_detector():
    sys.path.insert(0, str(CODE / "src"))
    sys.path.insert(0, str(CODE / "src/pixel3dmm/preprocessing/PIPNet/FaceBoxesV2"))
    import torch
    from pixel3dmm.preprocessing.PIPNet.FaceBoxesV2.faceboxes_detector import FaceBoxesDetector
    weights = PIPNET / "FaceBoxesV2/weights/FaceBoxesV2.pth"
    return FaceBoxesDetector("FaceBoxes", str(weights), True, torch.device("cuda"))


def face_score(detector, path):
    import cv2
    image = cv2.imread(str(path))
    if image is None or image.ndim < 3:
        return -1.0
    detections, _ = detector.detect(image, 0.6, 1)
    faces = sorted([d for d in detections if d[0] == "face"], key=lambda x: -float(x[1]))
    return float(faces[0][1]) if faces else 0.0


def filter_faces(batches, cfg):
    """Порог 0.75 из upstream: кадр ниже -> ValueError рвёт весь батч. Отсекаем заранее."""
    try:
        detector = make_detector()
    except Exception as error:
        say(f"  FaceBoxes недоступен ({error}) — фильтр пропущен, плохие кадры поймает цикл")
        return batches, []
    dropped = []
    for b in batches:
        if b["kind"] == "video":
            continue
        keep = []
        for i, item in enumerate(b["frames"]):
            score = face_score(detector, Path(b["target"]) / f"{i:05d}.jpg")
            item["score"] = round(score, 3)
            progress(phase="Поиск лиц (FaceBoxes)", current=item.get("relpath", ""),
                     completed=i + 1, total=len(b["frames"]),
                     stats=f"проверено {i + 1}/{len(b['frames'])} | отброшено {len(dropped)}")
            if score >= float(cfg["face_min_score"]):
                keep.append(i)
            else:
                dropped.append({"batch": b["id"], "photo": item["orig"], "score": item["score"]})
        stage_photos(b, keep)
    return batches, dropped


# ----------------------------------------------------------------------- проверка шагов

def count_files(folder, suffix=None):
    folder = Path(folder)
    if not folder.is_dir():
        return 0
    return sum(1 for p in folder.iterdir()
               if p.is_file() and (suffix is None or p.name.endswith(suffix)))


def count_dirs(folder):
    folder = Path(folder)
    return sum(1 for p in folder.iterdir() if p.is_dir()) if folder.is_dir() else 0


def nonempty_file(path):
    return path.is_file() and path.stat().st_size > 0


def _dir_size(folder):
    folder = Path(folder)
    return sum(p.stat().st_size for p in folder.rglob("*") if p.is_file()) if folder.is_dir() else 0


def resolve_actor_dir(bid):
    cands = [d for d in TRACK.glob(f"{bid}_nV*") if d.is_dir()]
    return max(cands, key=lambda d: d.stat().st_mtime) if cands else None


def preproc_missing(bid, n):
    """Кадры без обязательных артефактов предобработки. network_inference молча глотает
    ошибки по кадрам (try/except pass), поэтому rc=0 ничего не гарантирует."""
    base = PREPRO / bid
    import numpy as np
    mica_entries = list((base / "mica").iterdir()) if (base / "mica").is_dir() else []
    mica_ok = bool(mica_entries)
    for entry in mica_entries:
        try:
            identity = np.load(entry / "identity.npy", allow_pickle=False).squeeze()
            mica_ok = mica_ok and identity.shape == (300,) and bool(np.isfinite(identity).all())
        except (OSError, ValueError):
            mica_ok = False
    stages = {
        "cropped": [i for i in range(n) if not nonempty_file(base / "cropped" / f"{i:05d}.jpg")],
        "seg_og": [i for i in range(n) if not nonempty_file(base / "seg_og" / f"{i:05d}.png")],
        "PIPnet_landmarks": [i for i in range(n) if not nonempty_file(base / "PIPnet_landmarks" / f"{i:05d}.npy")],
        "normals": [i for i in range(n) if not nonempty_file(base / "p3dmm/normals" / f"{i:05d}.png")],
        "uv_map": [i for i in range(n) if not nonempty_file(base / "p3dmm/uv_map" / f"{i:05d}.png")],
        # read_data() берёт среднее по mica/*.npy: пустая папка = падение трекинга
        "mica": [] if mica_ok else [0],
    }
    return stages


def track_missing(actor_dir, n):
    out = {}
    out["mesh"] = [i for i in range(n) if not nonempty_file(actor_dir / "mesh" / f"{i:05d}.ply")]
    out["checkpoint"] = [i for i in range(n) if not nonempty_file(actor_dir / "checkpoint" / f"{i:05d}.frame")]
    if not nonempty_file(actor_dir / "result.mp4"):
        out["result.mp4"] = list(range(n))
    if not nonempty_file(actor_dir / "mesh" / "canonical.ply"):
        out["canonical.ply"] = list(range(n))
    return {k: v for k, v in out.items() if v}


def flame2023_available():
    path = FLAME_DATA / FLAME2023_PKL
    return path.is_file() and path.stat().st_size > 0


def resolve_flame2023(cfg):
    if not flame2023_available():
        return False
    return True


def track_args(cfg, n, use2023):
    if n <= 1:
        return (f"iters={int(cfg['single_iters'])} batch_size=1 save_landmarks=True "
                f"use_flame2023={'True' if use2023 else 'False'}")
    joint = min(int(cfg["joint_iters_max"]),
                max(int(cfg["joint_iters_min"]), int(cfg["joint_iters_per_frame"]) * n))
    return (" ".join([
        f"iters={int(cfg['stage_iters'])}", f"global_iters={joint}",
        f"batch_size={min(int(cfg['batch_size_cap']), n)}",
        "include_neck=False", "w_exp=0.1", "use_mouth_lmk=False", "w_shape=0.01",
        "w_shape_general=0.001", "normal_super=2000.0", "sil_super=1000.0",
        "ignore_mica=True", "is_discontinuous=True", "save_landmarks=True",
        f"use_flame2023={'True' if use2023 else 'False'}"]))


# --------------------------------------------------------------------------- прогон

def preproc_commands(cfg, bid, target):
    """Команды предобработки батча: кроп+ландмарки -> MICA (и arcface-слепки) -> сегментация -> нормали -> uv.

    Первые три пишут в PREPRO/<bid>, последние две — p3dmm/normals и p3dmm/uv_map; порядок важен
    (сегментация берёт cropped, сети — тоже), поэтому он один и тот же и в joint, и в photo-режиме."""
    mica = CODE / "src/pixel3dmm/preprocessing/MICA"
    return (
        (["python", str(CODE / "scripts/run_cropping.py"), "--video_or_images_path", str(target)],
         CODE / "scripts"),
        (["python", str(mica / "demo.py"), "-video_name", bid, "-a", str(PREPRO / bid / "arcface"),
          "-m", str(mica / "data/pretrained/mica.tar")], mica),
        (["python", str(CODE / "scripts/run_facer_segmentation.py"), "--video_name", bid], CODE / "scripts"),
        (["python", "scripts/network_inference.py", "model.prediction_type=normals", f"video_name={bid}"], CODE),
        (["python", "scripts/network_inference.py", "model.prediction_type=uv_map", f"video_name={bid}"], CODE),
    )


def process_batch(cfg, b, use2023):
    t0 = time.time()
    bid = b["id"]
    log = LOGS / f"{bid}.log"
    log.unlink(missing_ok=True)
    args = ""
    b.pop("n_done", None)
    if b["kind"] == "photos" and b["n"] > MAX_TRACK_FRAMES:
        return "fail", f"больше {MAX_TRACK_FRAMES} кадров — разбей папку на отдельные батчи", args, 0.0
    attempts = max(1, int(cfg["attempts"]))
    timeout = float(cfg["batch_timeout_min"])
    deadline = t0 + timeout * 60 if timeout > 0 else None

    def run_step(cmd, cwd=None):
        remaining = (deadline - time.time()) / 60 if deadline is not None else None
        if remaining is not None and remaining <= 0:
            return -9, f"таймаут батча {timeout} мин"
        return run_cmd(cmd, log, timeout_min=remaining, cwd=cwd)

    for attempt in range(attempts):
        shutil.rmtree(PREPRO / bid, ignore_errors=True)
        preprocessing = preproc_commands(cfg, bid, b["target"])[:3]
        for cmd, cwd in preprocessing:
            rc, why = run_step(cmd, cwd=cwd)
            if rc:
                return "fail", f"preprocessing ({Path(cmd[1]).name}): {why}; {tail(log, 12)}", args, time.time() - t0
        produced = count_files(PREPRO / bid / "cropped", ".jpg")
        # ожидаемое число кадров: для фото = сколько застили, для видео = сколько нарезали
        expected = count_files(PREPRO / bid / "rgb", ".jpg") if b["kind"] == "video" else b["n"]
        if b["kind"] == "video":
            b["n_real"] = expected
            b["n"] = expected
            write_manifest(b)
        if produced == 0 or expected == 0:
            return "fail", "preprocessing: ни одного кадра в cropped", args, time.time() - t0
        holes = preproc_missing(bid, expected)
        missing = sorted({i for key in ("cropped", "seg_og", "PIPnet_landmarks") for i in holes[key]})
        if missing and attempt + 1 < attempts:
            keep = [i for i in range(b["n"]) if i not in missing] if b["kind"] == "photos" else []
            if not keep:
                return "fail", f"не прошли предобработку кадры {missing[:10]}", args, time.time() - t0
            say(f"  {bid}: выкидываем кадры {missing[:12]} и повторяем предобработку")
            stage_photos(b, keep)
            expected = b["n"]
            continue
        if missing:
            return "fail", f"не прошли предобработку кадры {missing[:10]}", args, time.time() - t0
        if holes["mica"]:
            return "fail", "предобработка не сделала MICA (папка mica пуста) — трекинг на ней упадёт", args, time.time() - t0
        break

    if expected > MAX_TRACK_FRAMES:
        return "fail", f"видео содержит {expected} кадров; разбей на части до {MAX_TRACK_FRAMES}", args, time.time() - t0

    for cmd, cwd in preproc_commands(cfg, bid, b["target"])[3:]:
        rc, why = run_step(cmd, cwd=cwd)
        if rc:
            return "fail", f"{cmd[2].split('=')[1]}: {why}; {tail(log, 12)}", args, time.time() - t0
    holes = {k: v for k, v in preproc_missing(bid, expected).items() if v and k in ("normals", "uv_map")}
    if holes:
        return "fail", "сеть не сгенерировала часть кадров: " + ", ".join(f"{k}:{len(v)}" for k, v in holes.items()), args, time.time() - t0

    args = track_args(cfg, expected, use2023)
    b["track_args"] = args
    b["tracking_config"] = effective_config(cfg, b, expected, use2023)
    for old_actor in TRACK.glob(f"{bid}_nV*"):
        if old_actor.is_dir():
            shutil.rmtree(old_actor)
    rc, why = run_step(["python", "scripts/track.py", f"video_name={bid}"] + args.split())
    actor = resolve_actor_dir(bid)
    if rc or actor is None:
        return "fail", f"track: {why or 'нет выходного каталога'}; {tail(log, 20)}", args, time.time() - t0
    n_track = expected
    holes = track_missing(actor, n_track)
    if holes:
        return "fail", "трекинг не дописал: " + ", ".join(f"{k}:{len(v)}" for k, v in holes.items()), args, time.time() - t0
    try:
        extract_params(actor, n_track)
    except Exception as error:
        return "fail", f"повреждён checkpoint: {type(error).__name__}: {error}", args, time.time() - t0
    b["n_done"] = n_track
    if not cfg.get("keep_preproc", True):
        shutil.rmtree(PREPRO / bid, ignore_errors=True)
    return "ok", "", args, time.time() - t0


# ---------------------------------------------------------------- извлечение параметров

PARAM_KEYS = ("exp", "shape", "eyes", "eyelids", "jaw", "neck", "R", "R_rotation_matrix", "t")
EXTRA_KEYS = ("joint_transforms", "img_size", "frame_id", "global_step")


def _as_batch(value):
    """Значение из .frame -> (1, ...): размерности матриц (R_base 3x3, joint_transforms) сохраняем."""
    import numpy as np
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.shape[0] == 1:
        return arr
    return arr[None]


def frame_params(actor_dir, key):
    """Один checkpoint кадра -> (строка таблицы, {имя: массив (1, ...)}) — общий код joint и photo."""
    import numpy as np
    import torch
    path = Path(actor_dir) / "checkpoint" / f"{key}.frame"
    d = torch.load(path, map_location="cpu", weights_only=False)  # файл написал наш же трекинг
    d["flame"] = {name: _as_batch(value) for name, value in d["flame"].items()}
    d["camera"] = {name: _as_batch(value) for name, value in d["camera"].items()}
    one = {name: _as_batch(d["flame"][name]) for name in PARAM_KEYS}
    for name, value in d["camera"].items():
        one[name] = _as_batch(value)
    for name in EXTRA_KEYS:
        if name in d:
            one[name] = _as_batch(d[name])
    jaw = np.asarray(d["flame"]["jaw"]).ravel()
    exp = np.asarray(d["flame"]["exp"]).ravel()
    pose = pose_of_payload({"flame": {k: one[k] for k in ("R", "R_rotation_matrix") if k in one},
                            "camera": {k: one[k] for k in ("R_base_0",) if k in one}})
    row = {
        "frame": key,
        "mesh_ply": f"mesh/{key}.ply",
        "frame_file": f"checkpoint/{key}.frame",
        "jaw_norm": round(float(np.linalg.norm(jaw)), 5),
        "exp_norm": round(float(np.linalg.norm(exp)), 5),
        "shape_norm": round(float(np.linalg.norm(np.asarray(d["flame"]["shape"]).ravel())), 5),
        "t": ";".join(f"{x:.5f}" for x in np.asarray(d["flame"]["t"]).ravel()[:3]),
        "fl": round(float(np.asarray(d["camera"]["fl"]).ravel()[0]), 4),
        "pp": ";".join(f"{x:.3f}" for x in np.asarray(d["camera"]["pp"]).ravel()[:2]),
        # ракурс кадра по матрице поворота: тот же источник, что и в params.json, только плоский для csv
        "yaw_deg": pose.get("yaw_deg", ""), "pitch_deg": pose.get("pitch_deg", ""),
        "roll_deg": pose.get("roll_deg", ""), "view_bin": pose.get("bin") or "",
    }
    return row, one


def extract_params(actor_dir, n):
    """Все FLAME-параметры и камера по кадрам -> (строки frames.csv, массивы params.npz)."""
    import numpy as np
    per, rows = {}, []
    for i in range(n):
        row, one = frame_params(actor_dir, f"{i:05d}")
        rows.append(row)
        for name, value in one.items():
            per.setdefault(name, []).append(value)
    arrays = {}
    for name, values in per.items():
        try:
            arrays[name] = np.concatenate(values, axis=0)
        except ValueError:
            say(f"  параметр {name} разной размерности — в params.npz не кладём")
    return rows, arrays


# -------------------------------------------------------------------- ракурс (поза) кадра

_AXES_CACHE = {}


def unit_vector(v):
    """Единичный вектор; нулевой оставляем нулевым — вырожденную геометрию ловит вызывающий код."""
    import numpy as np
    arr = np.asarray(v, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    return arr / norm if norm > 1e-12 else arr


def yaw_bin(yaw):
    """Имя бина по VIEW_BINS: нижняя граница включительно, верхняя — нет (кроме последнего бина)."""
    import math
    if yaw is None:
        return None
    yaw = float(yaw)
    if not math.isfinite(yaw):
        return None
    for index, (name, lower, upper, _) in enumerate(VIEW_BINS):
        if lower <= yaw < upper or (index == len(VIEW_BINS) - 1 and yaw == upper):
            return name
    return None


def rotation_6d_to_matrix(vec):
    """6D-поворот Zhou et al. -> 3x3, как pytorch3d.transforms.rotation_6d_to_matrix.

    Запасной путь: трекер пишет и готовую R_rotation_matrix, но в старых чекпойнтах её может не быть.
    От pytorch3d отличаемся только в вырожденном входе (там нулевой второй вектор -> нечисловая
    матрица, здесь — любая перпендикулярная плоскость), потому что ронять экспорт фото из-за этого нельзя."""
    import numpy as np
    v = np.asarray(vec, dtype=np.float64).ravel()
    if v.size != 6:
        raise ValueError(f"ожидалось 6 чисел поворота, получено {v.size}")
    a1, a2 = v[:3], v[3:6]
    b1 = unit_vector(a1)
    if not np.any(b1):
        raise ValueError("вырожденный 6D-поворот: нулевой первый вектор")
    b2 = unit_vector(a2 - b1 * float(np.dot(b1, a2)))
    if not np.any(b2):
        helper = np.array([0.0, 0.0, 1.0]) if abs(float(b1[2])) < 0.9 else np.array([1.0, 0.0, 0.0])
        b2 = unit_vector(helper - b1 * float(np.dot(b1, helper)))
    return np.stack([b1, b2, np.cross(b1, b2)], axis=0)


def flame2020_dir():
    """generic_model.pkl + landmark_embedding.npy: те же файлы, что проверяет preflight по REQUIRED_FLAME2020."""
    return Path(FLAME_DATA) / "FLAME2020"


def landmark_embedding_axes(flame_dir):
    """Оси лица по шаблону FLAME2020: те же 68 вшитых ландмарок и та же построка, что в
    pixel3dmm_analysis.build_reference (origin между глазами, lateral по скулам, up к макушке,
    front — нормаль к этой плоскости, развёрнутая в сторону носа). Только чистый CPU: pickle + numpy."""
    import numpy as np
    import pickle
    flame_dir = Path(flame_dir)
    with (flame_dir / "generic_model.pkl").open("rb") as stream:
        model = pickle.load(stream, encoding="latin1")
    embedding = np.load(flame_dir / "landmark_embedding.npy", allow_pickle=True, encoding="latin1").item()
    face_ids = np.concatenate([np.asarray(embedding["dynamic_lmk_faces_idx"])[0],
                               np.asarray(embedding["static_lmk_faces_idx"])])
    bary = np.concatenate([np.asarray(embedding["dynamic_lmk_bary_coords"])[0],
                          np.asarray(embedding["static_lmk_bary_coords"])])
    if len(face_ids) != 68 or bary.shape != (68, 3):
        raise ValueError("неожиданная топология FLAME landmark embedding")
    vertices = np.asarray(model["v_template"], dtype=np.float64)
    triangles = np.asarray(model["f"], dtype=int)
    lm = np.sum(vertices[triangles[face_ids.astype(int)]] * bary.astype(float)[..., None], axis=-2)
    origin = (lm[21] + lm[22]) / 2
    lateral = unit_vector(lm[45] - lm[36])
    up = unit_vector(origin - lm[8])
    up = unit_vector(up - lateral * float(np.dot(up, lateral)))
    front = unit_vector(np.cross(lateral, up))
    if float(np.dot(lm[30] - origin, front)) < 0:
        front = -front
    axes = np.column_stack([lateral, up, front])
    if not np.isfinite(axes).all() or abs(float(np.linalg.det(axes))) < 0.5:
        raise ValueError("вырожденная референсная геометрия FLAME")
    return axes


def reference_axes(flame_dir=None):
    """Кэширует результат: generic_model.pkl весит 53 МБ, перечитывать его на каждое фото нельзя.

    Битый файл кэшируем как ошибку (пересчитывать бесполезно), а отсутствие файла — нет: веса
    могут докачать следующей ячейкой, и в том же процессе оси должны появиться."""
    key = str(flame_dir or flame2020_dir())
    cached = _AXES_CACHE.get(key)
    if cached is None:
        try:
            cached = landmark_embedding_axes(Path(key))
        except Exception as error:
            if not (Path(key) / "generic_model.pkl").is_file():
                raise FileNotFoundError(f"нет {Path(key) / 'generic_model.pkl'} — выполни ячейку «4. Веса»")
            cached = error
        _AXES_CACHE[key] = cached
    if isinstance(cached, BaseException):
        raise cached
    return cached


def pose_from_camera(rotation, axes):
    """yaw/pitch/roll по повёрнутым осям лица — дословно та же формула, что в analysis.pose_from_camera."""
    import math
    import numpy as np
    lateral, up, front = (np.asarray(rotation, dtype=np.float64)
                          @ np.asarray(axes, dtype=np.float64)).T
    yaw = math.degrees(math.atan2(front[0], front[2]))
    pitch = math.degrees(math.atan2(front[1], math.hypot(front[0], front[2])))
    roll = math.degrees(math.atan2(lateral[1], lateral[0]))
    return {"yaw_deg": yaw, "pitch_deg": pitch, "roll_deg": roll, "convention": POSE_CONVENTION}


def pose_of_payload(payload):
    """Поза одного кадра: R_total = R_base_0 @ R (как в разборе корпуса), метка — по девяти бинам.

    Берётся из того, что уже лежит в params.json (flame.R / flame.R_rotation_matrix + camera.R_base_0),
    поэтому этим же путём можно дополнить и старые папки фото, не трогая torch и чекпойнт.
    Поза не двигает фото между папками: разметка корпуса остаётся основным ярлыком, это только измерение."""
    import numpy as np
    try:
        axes = reference_axes()
    except Exception as error:
        return {"error": f"оси FLAME2020 не получаются: {type(error).__name__}: {error}"}
    flame = payload.get("flame") or {}
    camera = payload.get("camera") or {}
    rmat, r6, base = flame.get("R_rotation_matrix"), flame.get("R"), camera.get("R_base_0")
    if base is None or (rmat is None and r6 is None):
        return {"error": "в params.json нет camera.R_base_0 или поворота головы"}
    try:
        base = np.asarray(base, dtype=np.float64).reshape(3, 3)
        if rmat is not None:
            rot = np.asarray(rmat, dtype=np.float64).reshape(3, 3)
            source = "camera.R_base_0 @ flame.R_rotation_matrix"
        else:
            rot = rotation_6d_to_matrix(r6)
            source = "camera.R_base_0 @ rotation_6d_to_matrix(flame.R)"
        if not (np.isfinite(base).all() and np.isfinite(rot).all()):
            raise ValueError("не числа в матрицах поворота")
    except Exception as error:
        return {"error": f"матрица поворота кадра не читается: {type(error).__name__}: {error}"}
    pose = pose_from_camera(base @ rot, axes)
    for key in ("yaw_deg", "pitch_deg", "roll_deg"):
        pose[key] = round(float(pose[key]), 2)
    pose["bin"] = yaw_bin(pose["yaw_deg"])
    pose["source"] = source
    pose["reference"] = "FLAME2020 v_template + landmark_embedding (68 вшитых точек), оси analysis.build_reference"
    return pose


def effective_config(cfg, b, n, use2023):
    """Upstream track.py не сохраняет конфиг запуска — сохраняем сами: tracking.yaml + наши CLI-аргументы.
    Иначе потом невозможно воспроизвести, что реально крутилось."""
    args = b.get("track_args", track_args(cfg, n, use2023)).split()
    try:
        from omegaconf import OmegaConf
        base = OmegaConf.load(CODE / "configs" / "tracking.yaml")
        merged = OmegaConf.merge(base, OmegaConf.from_cli([f"video_name={b['id']}"] + args))
        return OmegaConf.to_container(merged, resolve=True)
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}", "cli_args": args}


def batch_metrics(cfg, b, actor_dir, n, use2023, sec):
    import torch
    return {
        "batch": b["id"], "subject": b["subject"], "angle": b["angle"], "kind": b["kind"],
        "frames": n, "photos": [item["orig"] for item in b["frames"]],
        "relpaths": [item["relpath"] for item in b["frames"]],
        "track_args": b.get("track_args", track_args(cfg, n, use2023)), "use_flame2023": bool(use2023),
        "tracking_config": b.get("tracking_config", effective_config(cfg, b, n, use2023)),
        "paths": {"preprocessed": str(PREPRO / b["id"]), "actor": str(actor_dir),
                  "input": cfg["input"], "output": cfg["output"]},
        "actor_dir": actor_dir.name, "sec": round(sec, 1),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "canonical_ply": (Path(actor_dir) / "mesh/canonical.ply").is_file(),
        "landmark_files": sorted(p.name for p in (Path(actor_dir) / "mesh").glob("*.npy"))[:3],
        "preproc_kept": list(PREPROC_SUBSET) if cfg.get("keep_preproc", True) else [],
        "dropped": b.get("dropped", []), "ts": int(time.time()),
    }


# ---------------------------------------------------------------------------- sync

def batches_root(cfg):
    """Папки батчей: OUTPUT/<батч>/ при flat=True, иначе OUTPUT/batches/<батч>/.

    flat=True включает ноутбук: человек видит на Drive путь
    «pixel3dmm_results/<архив>/photos/<фото>/», а не лишний уровень «batches».
    Служебные папки (_state, _index, _logs, _env) начинаются с «_» и с батчами не путаются.
    """
    return Path(cfg["output"]) if cfg.get("flat") else Path(cfg["output"]) / "batches"


def batch_dir(cfg, bid):
    return batches_root(cfg) / bid


def photo_names(rows):
    """кадр -> имя папки в Drive: по исходному файлу; при коллизии имён добавляем номер кадра."""
    base = {}
    for row in rows:
        hint = row.get("orig_relpath") or row.get("orig_path") or ""
        base[row["frame"]] = photo_folder(hint) if hint else f"frame_{row['frame']}"
    counts = {}
    for name in base.values():
        counts[name] = counts.get(name, 0) + 1
    return {frame: (f"{name}__{frame}" if counts[name] > 1 else name) for frame, name in base.items()}


def overlay_frame(video, dst_jpg, index, target_size=None):
    """Сохраняет только центральную панель (оригинал + наложенный меш) из трёхпанельного кадра.
    Размер результата задаётся исходным фото, если оно доступно."""
    try:
        import cv2
    except ImportError:
        return False
    capture = cv2.VideoCapture(str(video))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, image = capture.read()
    finally:
        capture.release()
    if not ok or image is None:
        return False
    h, w = image.shape[:2]
    # Pixel3DMM composes three equal panels side-by-side; take the middle, model-overlay view.
    if w >= h * 2.4:
        left, right = w // 3, (2 * w) // 3
        image = image[:, left:right]
    if target_size and target_size[0] > 0 and target_size[1] > 0:
        image = cv2.resize(image, tuple(map(int, target_size)), interpolation=cv2.INTER_AREA)
    return bool(cv2.imwrite(str(dst_jpg), image, [cv2.IMWRITE_JPEG_QUALITY, 95]))


def photo_payload(b, row, arrays, index):
    """params.json одного фото: FLAME-параметры, камера и измеренный ракурс обычными списками + связь с исходником."""
    import numpy as np
    flame, camera = {}, {}
    for name, values in arrays.items():
        try:
            value = values[index]
        except IndexError:
            continue
        (flame if name in PARAM_KEYS else camera)[name] = np.asarray(value).tolist()
    pose = pose_of_payload({"flame": flame, "camera": camera})
    return {
        "batch": b["id"], "subject": b["subject"], "angle": b["angle"], "frame": row["frame"],
        "source": row.get("orig_path", ""), "relpath": row.get("orig_relpath", ""),
        "face_score": row.get("face_score", ""), "note": row.get("note", ""),
        "summary": {key: row.get(key, "") for key in
                    ("jaw_norm", "exp_norm", "shape_norm", "t", "fl", "pp")},
        "flame": flame, "camera": camera,
        # метка ракурса — по матрице поворота и девяти бинам корпуса, а не по догадкам о 2D-асимметрии
        "pose": pose,
    }


def landmark_artifacts(actor, frame):
    """Ландмарки кадра: upstream пишет их номером без ведущего нуля, а np.save к имени
    '.ply' приклеивает '.npy' — поэтому ibug68_N.ply.npy. Падает в обе ветки экспорта.
    zeropadding на всякий случай перебирается: в другом релизе трекера имена могут оказаться
    и с нулями, а молча потерять ландмарки из папки фото было бы дороже, чем один is_file()."""
    actor = Path(actor)
    variants = [str(frame)]
    if str(frame).isdigit():                       # upstream пишет номер без нулей, но с нулями тоже встречался
        variants = [str(int(frame)), f"{int(frame):05d}"]
    out = []
    for label, pattern in (("landmarks_7.npy", "landmarks_{}_0.npy"),
                           ("landmarks_68.npy", "ibug68_{}.ply.npy"),
                           ("landmarks_eyes.npy", "now_{}.ply.npy")):
        paths = [actor / "mesh" / pattern.format(v) for v in variants]
        out.append((label, next((p for p in paths if nonempty_file(p)), paths[-1])))
    return out


def export_photo_folders(cfg, b, actor, rows, arrays, index_of, tmp):
    """photos/<имя исходного файла>/ — всё извлечённое по каждому фото лежит в своей папке."""
    names = photo_names(rows)
    pre = PREPRO / b["id"]
    canonical = actor / "mesh" / "canonical.ply"
    made = []
    for row in rows:
        frame = row["frame"]
        index = index_of.get(frame, 0)
        dst = tmp / "photos" / names[frame]
        dst.mkdir(parents=True, exist_ok=True)
        source = Path(row.get("orig_path") or "")
        if source.is_file():
            shutil.copy2(source, dst / f"source{source.suffix.lower() or '.jpg'}")
        for src, name in ((actor / "mesh" / f"{frame}.ply", "mesh.ply"),
                          (actor / "checkpoint" / f"{frame}.frame", "checkpoint.frame"),
                          (pre / "cropped" / f"{frame}.jpg", "crop.jpg"),
                          (pre / "seg_og" / f"{frame}.png", "segmentation.png"),
                          (pre / "seg_og" / f"{frame}.jpg", "segmentation.jpg"),
                          (pre / "PIPnet_landmarks" / f"{frame}.npy", "landmarks_pipnet.npy")):
            if src.is_file():
                shutil.copy2(src, dst / name)
        for name, src in landmark_artifacts(actor, frame):
            if src.is_file():
                shutil.copy2(src, dst / name)
        if canonical.is_file():
            shutil.copy2(canonical, dst / "canonical.ply")
        position = int(frame) if str(frame).isdigit() else index   # нумерация кадров в result.mp4 = номер кадра
        overlay_frame(actor / "result.mp4", dst / "overlay.jpg", position)
        payload = photo_payload(b, row, arrays, index)
        payload["visibility"] = write_visibility(dst)
        payload["files"] = sorted(p.name for p in dst.iterdir() if p.is_file())
        (dst / "params.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        made.append(names[frame])
    if canonical.is_file():
        shutil.copy2(canonical, tmp / "canonical.ply")
    lines = [f"# батч {b['id']}: кадров {len(rows)}", "",
             f"вход: {cfg['input']}", "",
             "frames.csv — таблица по кадрам, params.npz — все параметры разом, result.mp4 — "
             "наложение меша" + (", artifacts.tar.gz — mesh/ checkpoint/ предобработка"
                                 if cfg.get("keep_archive", True) else "") + ".",
             "",]
    for row in rows:
        lines.append(f"- {row.get('orig_relpath') or row.get('orig_path') or row['frame']} "
                     f"-> photos/{names[row['frame']]}/")
    (tmp / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return made


def copy_file(src, dst):
    """Один файл в выходную папку: через .part и с проверкой размера — полунки в Drive не остаётся."""
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    partial = dst.parent / (dst.name + ".part")
    shutil.copy2(src, partial)
    if partial.stat().st_size != src.stat().st_size:
        raise OSError(f"неполная копия: {dst}")
    partial.replace(dst)
    return dst.stat().st_size


def write_json(payload, dst):
    """JSON в выходную папку атомарно (через тот же .part)."""
    dst = Path(dst)
    tmp = dst.parent / (dst.name + ".part")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(dst)
    return dst.stat().st_size


def sync_dir_to_drive(src, dst):
    """Папка (photos/...) в Drive: копируем пофайлово через .part — обрезанных файлов на Drive быть не должно."""
    files = [p for p in sorted(src.rglob("*")) if p.is_file()]
    total = 0
    dst.mkdir(parents=True, exist_ok=True)
    for item in files:
        total += copy_file(item, dst / item.relative_to(src))
    return len(files), total


def write_archive(archive, actor_dir, pre, cfg):
    """Не нужен в unit=photo (всё лежит по папкам фото), но остаётся для joint-режима по желанию."""
    with tarfile.open(archive, "w:gz") as tar:
        for folder, arc in ((Path(actor_dir) / "mesh", "mesh"), (Path(actor_dir) / "checkpoint", "checkpoint")):
            if folder.is_dir():
                tar.add(folder, arcname=arc)
        for optional, arc in ((Path(actor_dir) / "config.yaml", "config.yaml"),   # upstream его не пишет
                              (Path(actor_dir) / "logs", "logs")):                  # tensorboard: кривые потерь
            if optional.is_file():
                tar.add(optional, arcname=arc)
            elif optional.is_dir() and _dir_size(optional) < 40_000_000:
                tar.add(optional, arcname=arc)
        for rel in (PREPROC_SUBSET if cfg.get("keep_preproc", True) else ()):
            folder = Path(pre) / rel
            if folder.is_dir():
                tar.add(folder, arcname="preproc/" + rel.replace("/", "_"))
    return archive


def sync_batch(cfg, b, actor_dir, n, sec, use2023, photos=None):
    """Собирает результат батча в Drive: мелкие файлы отдельно + artifacts.tar.gz
    с mesh/, checkpoint/ и предобработкой. photos=None -> по cfg["export"];
    демо просит папки по фото всегда — иначе непонятно, где результат конкретного файла."""
    out = Path(cfg["output"])
    dst = batch_dir(cfg, b["id"])
    tmp = WORK / "sync" / b["id"]
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    rows, arrays = extract_params(actor_dir, n)
    index_of = {row["frame"]: i for i, row in enumerate(rows)}   # позиции в массивах params.npz
    manifest = read_manifest(b["id"])
    for row in rows:
        row.update(manifest.get(row["frame"], {}))
    if manifest:
        rows.sort(key=lambda r: r["frame"])
    with (tmp / "frames.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FRAMES_CSV_COLS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    if manifest_path(b).is_file():
        shutil.copy2(manifest_path(b), tmp / "manifest.csv")
    if arrays:
        import numpy as np
        np.savez_compressed(tmp / "params.npz", **arrays)
    shutil.copy2(actor_dir / "result.mp4", tmp / "result.mp4")
    (tmp / "metrics.json").write_text(json.dumps(batch_metrics(cfg, b, actor_dir, n, use2023, sec),
                                                 ensure_ascii=False, indent=1), encoding="utf-8")
    pre = PREPRO / b["id"]
    if cfg.get("keep_archive", True):
        write_archive(tmp / "artifacts.tar.gz", actor_dir, pre, cfg)
    want_photos = (str(cfg.get("export", "photos")) != "bundle") if photos is None else bool(photos)
    if want_photos:
        made = export_photo_folders(cfg, b, actor_dir, rows, arrays, index_of, tmp)
        say(f"  {b['id']}: папок по фото {len(made)}")
    written = []
    if cfg.get("sync_to_drive", True):
        dst.mkdir(parents=True, exist_ok=True)
        for src in sorted(tmp.iterdir(), key=lambda p: (p.name == "metrics.json", p.name)):
            target = dst / src.name
            if src.is_dir():
                count, size = sync_dir_to_drive(src, target)
                written.append((src.name + "/", size))
                say(f"  {b['id']}: {src.name}/ — {count} файлов в Drive")
                continue
            partial = dst / (src.name + ".part")
            shutil.copy2(src, partial)
            if partial.stat().st_size != src.stat().st_size:
                raise OSError(f"неполная копия в Drive: {src.name}")
            partial.replace(target)
            written.append((src.name, target.stat().st_size))
    if written:
        say(f"  {b['id']}: в Drive {len(written)} файлов")
    log_src = LOGS / f"{b['id']}.log"
    if cfg.get("sync_to_drive", True) and log_src.is_file():
        (out / "_logs").mkdir(parents=True, exist_ok=True)
        shutil.copy2(log_src, out / "_logs" / log_src.name)
    return written


# --------------------------------------------------------------------------- лог

def state_dir(cfg):
    path = Path(cfg["output"]) / "_state"
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_log(cfg, name="batch_log.jsonl", local=None):
    """Журнал прогона:Drive-копия + локальная. name/локальный путь разные для batch и photo."""
    local = LOCAL_LOG if local is None else Path(local)
    records = {}
    for path in (Path(cfg["output"]) / "_state" / name, local):
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if not isinstance(rec, dict):
                continue
            if path == local and "output" not in rec:
                continue
            if rec.get("output", cfg["output"]) != cfg["output"] or rec.get("input", cfg["input"]) != cfg["input"]:
                continue
            uid = rec.get("key") or rec.get("id")     # unit=photo: ключ фото, иначе — id батча
            if not uid:
                continue
            previous = records.get(uid, {})
            if rec.get("updated_ns", 0) >= previous.get("updated_ns", 0):
                records[uid] = rec
    return [records[k] for k in sorted(records)]


def append_log(cfg, rec, name="batch_log.jsonl", local=None):
    rec = dict(rec, input=cfg["input"], output=cfg["output"], updated_ns=time.time_ns())
    local = LOCAL_LOG if local is None else Path(local)
    local.parent.mkdir(parents=True, exist_ok=True)
    with local.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()
    last_error = None
    for _ in range(3):
        try:
            (state_dir(cfg) / name).write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in read_log(cfg, name, local)),
                encoding="utf-8")
            log = LOGS / f"{rec.get('log') or rec.get('id')}.log"
            if rec.get("status") == "fail" and log.is_file():
                (Path(cfg["output"]) / "_logs").mkdir(parents=True, exist_ok=True)
                shutil.copy2(log, Path(cfg["output"]) / "_logs" / log.name)
            return rec
        except OSError as error:
            last_error = error
            time.sleep(1)
    raise RuntimeError(f"журнал не записан на Drive: {last_error}")


BUNDLE_FILES = ("frames.csv", "manifest.csv", "params.npz", "metrics.json", "result.mp4", "artifacts.tar.gz")


def bundle_files(cfg):
    """Что обязательно должно лежать в папке батча: архив нужен только если его явно просили."""
    return tuple(name for name in BUNDLE_FILES
                 if name != "artifacts.tar.gz" or cfg.get("keep_archive", True))


def bundle_missing(folder, synced=(), cfg=None):
    folder = Path(folder)
    expected = dict(synced)
    names = bundle_files(cfg) if cfg is not None else BUNDLE_FILES
    return [name for name in names
            if not (folder / name).is_file() or (folder / name).stat().st_size == 0
            or (name in expected and (folder / name).stat().st_size != expected[name])]


def save_plan(cfg, plan):
    PLAN_FILE.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(plan, ensure_ascii=False, indent=1)
    PLAN_FILE.write_text(text, encoding="utf-8")
    try:
        (state_dir(cfg) / "plan.json").write_text(text, encoding="utf-8")
    except OSError as error:
        say(f"план в Drive не записан: {error}")


# ------------------------------------------------------------------ слепок среды (квота Colab)
# После сброса рантайма /content пуст: сборка энва + скачивание весов занимают ~45–55 минут,
# и всё это время идёт счётчик GPU-квоты. freeze упаковывает готовую среду в OUTPUT/_env,
# warm распаковывает её обратно — минуты вместо десятков минут сборки.
ENV_ARCHIVE_DIR = "_env"
ENV_MARKER = ENV_ROOT / ".pixel3dmm_setup_v2"
CACHE_ROOT = Path.home() / ".cache"
INSIGHTFACE_ROOT = Path.home() / ".insightface"
ENV_CACHE_VERSION = 2
PIXEL_REV = "fcd1fa973c7715b02a8948dfc679dff53cf85924"
EXTRACT_ROOT = "/"              # куда распаковывать слепок; в тестах подменяется на tmp_path


def freeze_parts():
    """(имя архива, что кладём, зачем) — пути читаются из глобалов, чтобы тесты могли их подменить."""
    return (
        ("p3dmm-env.tar.gz", (ENV_ROOT,), "микроокружение python 3.9 с torch cu118, pytorch3d и nvdiffrast в сборе"),
        ("pixel3dmm.tar.gz", (CODE,), "клон pixel3dmm с патчами, FLAME и весами сетей"),
        ("root-cache.tar.gz", (INSIGHTFACE_ROOT, CACHE_ROOT / "insightface", CACHE_ROOT / "torch_extensions",
                               CACHE_ROOT / "huggingface", CACHE_ROOT / "torch"),
         "веса insightface/FARL и JIT-сборка nvdiffrast"),
    )


def env_dir(cfg, make=False):
    path = Path(cfg["output"]) / ENV_ARCHIVE_DIR
    if make:
        path.mkdir(parents=True, exist_ok=True)
    return path


def env_manifest_path(cfg):
    return env_dir(cfg) / "manifest.json"


def read_env_manifest(cfg):
    path = env_manifest_path(cfg)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def archive_prefixes(sources):
    """Префиксы путей внутри tar: распаковка идёт в указанный корень, поэтому чужие пути запрещены."""
    return tuple(str(Path(p).absolute()).lstrip("/") for p in sources)


def cache_excluded(name):
    parts = Path(name).parts
    return (Path(name).name in {"token", "stored_tokens", "lock"}
            or Path(name).name.endswith(".lock") or ".locks" in parts)


def safe_extract(archive, sources, target="/", missing_only=False):
    """Validate the complete archive before extracting, including legacy conda/HF links."""
    prefixes = archive_prefixes(sources)

    def allowed(name):
        return any(name == p or name.startswith(p + "/") for p in prefixes)

    with tarfile.open(archive, "r:*") as tar:
        members = []
        for member in tar.getmembers():
            norm = posixpath.normpath(member.name)
            if (not allowed(norm) or member.name.startswith("/")
                    or ".." in member.name.split("/")):
                raise ValueError(f"{archive.name}: подозрительный путь в архиве: {member.name}")
            if cache_excluded(norm):
                continue
            if not (member.isfile() or member.isdir() or member.issym() or member.islnk()):
                raise ValueError(f"{archive.name}: подозрительный тип файла в архиве: {norm}")
            if member.issym() or member.islnk():
                link = (member.linkname.lstrip("/") if member.linkname.startswith("/")
                        else posixpath.join(posixpath.dirname(norm), member.linkname)
                        if member.issym() else member.linkname)
                link = posixpath.normpath(link)
                if not allowed(link) or cache_excluded(link):
                    raise ValueError(f"{archive.name}: ссылка за пределы кеша: {norm} -> {member.linkname}")
                member.linkname = (posixpath.relpath(link, posixpath.dirname(norm))
                                   if member.issym() else link)
            resolved = (Path(target) / norm).resolve()
            roots = [Path(target).absolute() / p for p in prefixes]
            if not any(resolved.is_relative_to(p) for p in roots):
                raise ValueError(f"{archive.name}: существующая ссылка выводит путь из кеша: {norm}")
            members.append(member)
        links = {m.name for m in members if m.issym() or m.islnk()}
        for member in members:
            if any(str(p) in links for p in Path(member.name).parents):
                raise ValueError(f"{archive.name}: запись внутрь ссылки: {member.name}")
        if missing_only:
            members = [m for m in members if m.isdir() or not (Path(target) / m.name).exists()]
        kwargs = {"members": members, "path": str(target)}
        if sys.version_info >= (3, 12):
            kwargs["filter"] = "fully_trusted"
        tar.extractall(**kwargs)
    return len(members)


def snapshot_signature():
    return {"version": ENV_CACHE_VERSION, "pixel_rev": PIXEL_REV, "hf_rev": HF_REV,
            "cuda_arch": os.environ.get("TORCH_CUDA_ARCH_LIST", ""),
            "flame_archives": {name: [p.stat().st_size, p.stat().st_mtime_ns]
                               for name in ("FLAME2020.zip", "FLAME2023.zip")
                               for p in (DRIVE_ROOT / name,) if p.is_file()}}


def snapshot_parts(cfg, manifest):
    """The manifest cannot choose extraction destinations or arbitrary archive paths."""
    allowed = {name: tuple(map(Path, sources)) for name, sources, _ in freeze_parts()}
    out = []
    for part in manifest.get("parts", []):
        if not isinstance(part, dict):
            raise ValueError("Некорректная часть слепка")
        name = str(part.get("name", ""))
        match = re.fullmatch(r"(p3dmm-env|pixel3dmm|root-cache)(?:-[0-9]+)?\.tar\.gz", name)
        if not match:
            raise ValueError(f"Некорректное имя архива: {name}")
        kind = match[1] + ".tar.gz"
        sources = [Path(p) for p in part.get("sources", [])]
        if not sources or any(p not in allowed[kind] for p in sources):
            raise ValueError(f"Недопустимые пути в слепке: {name}")
        archive = env_dir(cfg) / name
        if not archive.resolve().is_relative_to(env_dir(cfg).resolve()):
            raise ValueError(f"Архив находится за пределами _env: {name}")
        out.append((part, archive, sources, kind))
    return out


def snapshot_ready(cfg, manifest):
    if (manifest.get("invalid") or manifest.get("signature") != snapshot_signature()
            or manifest.get("output") != cfg["output"]):
        return False
    try:
        parts = snapshot_parts(cfg, manifest)
        return ({kind for _, _, _, kind in parts} == {p[0] for p in freeze_parts()}
                and all(archive.is_file() and archive.stat().st_size == part.get("bytes")
                        and re.fullmatch(r"[a-f0-9]{64}", str(part.get("sha256", "")))
                        for part, archive, _, _ in parts))
    except (OSError, ValueError, TypeError):
        return False


def dir_size_gb(folder):
    total = 0
    for root, dirs, files in os.walk(folder):
        for name in files:
            try:
                total += os.path.getsize(Path(root) / name)
            except OSError:
                pass
    return total / 1e9


def free_gb_at(path):
    """Свободно ГБ на устройстве, куда пишем. Для fuse-маунта Drive обычно показывает свободное /content,
    поэтому это только страховка от явного переполнения, а не квота Google Drive."""
    try:
        return shutil.disk_usage(path).free / 1e9
    except OSError:
        return 0.0


def session_elapsed_hours():
    """Сколько живёт текущий рантайм: /proc/uptime — время контейнера, не ноутбука."""
    try:
        return float(Path("/proc/uptime").read_text(encoding="utf-8").split()[0]) / 3600.0
    except (OSError, IndexError, ValueError):
        return 0.0


GUARD_MARGIN_S = 900          # запас на запись последнего батча в Drive перед принудительным отключением


def session_deadline(cfg, now=None):
    """До какого момента можно считать: минимум из бюджета ячейки и остатка жизни рантайма Colab.

    Бесплатная среда отключается принудительно (~12–13 ч), и обрыв посреди батча означал бы
    потерю всего времени, потраченного на него, поэтому цикл останавливается сам чуть раньше."""
    now = time.time() if now is None else now
    limits = []
    budget = float(cfg.get("session_max_hours", 0) or 0) * 3600
    hard = float(cfg.get("session_hard_limit_hours", 0) or 0) * 3600
    if budget > 0:
        limits.append(budget)
    if hard > 0:
        limits.append(max(0.0, hard - session_elapsed_hours() * 3600 - GUARD_MARGIN_S))
    return now + (min(limits) if limits else 365 * 86400.0)


def median_sec_per_frame(records):
    """Медиана секунд на кадр по успешно посчитанным батчам; 0.0 если опыта ещё нет."""
    vals = sorted(float(r["sec"]) / int(r["n"]) for r in records
                  if r.get("status") == "ok" and int(r.get("n") or 0) > 0 and float(r.get("sec") or 0) > 0)
    if not vals:
        return 0.0
    mid = len(vals) // 2
    return vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2.0


def weights_stamp():
    """Файл-отметка проверенных весов (живёт в /content/pixel3dmm, то есть и в слепке среды тоже)."""
    return CODE / ".weights_sha256.json"


_VERIFIED_ASSETS = set()


def verified_asset(name, path):
    """Reuse a SHA256 verification only for the same revision, size and nanosecond mtime."""
    path = Path(path)
    size, expected = HF_FILES[name]
    if not path.is_file():
        return False
    stat = path.stat()
    if stat.st_size != size:
        return False
    token = (expected, stat.st_dev, stat.st_ino, size, stat.st_mtime_ns)
    try:
        stamp = json.loads(weights_stamp().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        stamp = {}
    if not isinstance(stamp, dict) or stamp.get("revision") != HF_REV:
        stamp = {"revision": HF_REV, "files": {}}
    if not isinstance(stamp.get("files"), dict):
        stamp["files"] = {}
    files = stamp["files"]
    row = {"sha256": expected, "size": size, "mtime_ns": stat.st_mtime_ns}
    if files.get(str(path)) == row:
        _VERIFIED_ASSETS.add(token)
        return True
    if token not in _VERIFIED_ASSETS:
        progress(phase="Проверка SHA256", current=name)
        if file_sha256(path) != expected:
            return False
        _VERIFIED_ASSETS.add(token)
    files[str(path)] = row
    write_json(stamp, weights_stamp())
    return True


def check_weights():
    """Проверяет веса по размеру и SHA256; повторное хеширование 4+ ГБ не нужно, если файл не менялся.

    Отметка живёт в /content/pixel3dmm/.weights_sha256.json и поэтому попадает и в слепок среды: после
    warm/preflight считает только то, что реально изменилось. Возвращает список (rel, размер, ок, вывод)."""
    out = []
    for rel, minimum in WEIGHTS:
        path = CODE / rel
        name = Path(rel).name
        if not path.is_file():
            out.append((rel, 0, False, "НЕТ"))
            continue
        size = path.stat().st_size
        ok = verified_asset(name, path)
        out.append((rel, size, ok, "ok" if ok else "НЕТ"))
    return out


# ------------------------------------------------------------------ стадии

def st_preflight(cfg):
    import torch
    say("=" * 72)
    say("ПРЕДФЛАЙТ — готов ли рантайм к корпусу")
    say("=" * 72)
    problems = []
    say(f"python   : {sys.executable}")
    if not sys.executable.startswith(str(ENV_ROOT)):
        problems.append("ядро пайплайна запускается не из /content/p3dmm-env")
    say(f"torch    : {torch.__version__} | cuda: {torch.cuda.is_available()} | "
        f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'БЕЗ GPU (Runtime -> Change runtime type -> T4)'}")
    if not torch.cuda.is_available():
        problems.append("GPU недоступен — трекинг не пойдёт")
    if not CODE.is_dir():
        problems.append("нет /content/pixel3dmm — выполни ячейку «3. Окружение»")
    needed = dict(WEIGHTS)
    for rel, size, ok, flag in check_weights():
        if not ok:
            problems.append(f"{rel} — неверный размер/SHA256 ({size} байт, ожидается {needed[rel]}), выполни ячейку «4. Веса»")
        say(f"вес      : {flag:<3} {rel} ({size / 1e6:.0f} МБ)")
    for rel in REQUIRED_FLAME2020:
        path = FLAME_DATA / "FLAME2020" / rel
        if not path.is_file() or not path.stat().st_size:
            problems.append(f"FLAME2020/{rel} нет — повтори ячейку «4. Веса»")
        say(f"flame    : {'ok' if path.is_file() else 'НЕТ ':<3} {rel}")
    use2023 = flame2023_available()
    try:
        use2023 = resolve_flame2023(cfg)
        say(f"flame2023: {'ok' if use2023 else 'НЕТ ':<3} {FLAME2023_PKL} -> "
            f"батчи пойдут с use_flame2023={'True' if use2023 else 'False'}")
    except ValueError as error:
        problems.append(str(error))
        say(f"flame2023: НЕТ {FLAME2023_PKL} (в CONFIG стоит use_flame2023=True)")
    nms = list((PIPNET / "FaceBoxesV2/utils").rglob("*nms*.so"))
    say(f"faceboxes: {'ok' if nms else 'НЕТ ':<3} {', '.join(p.name for p in nms) or 'не собран (ячейка 4)'}")
    if not nms:
        problems.append("FaceBoxes не собран — выполни ячейку «4. Веса»")
    for name, (size, digest) in HF_FILES.items():
        if name.startswith("insightface/"):
            path = Path.home() / ("." + name)
            if not path.is_file() or path.stat().st_size != size:
                problems.append(f"{name} отсутствует или недокачан — повтори ячейку «4. Веса»")
    say("torch.compile -> " + patch_compile(bool(cfg.get("torch_compile"))))
    say("трекинг одного кадра -> " + patch_single_frame_support())
    try:
        import nvdiffrast.torch as dr
        dr.RasterizeCudaContext()
        say("nvdiffrast: JIT-плагин собран ( gcc 11 из энва )")
    except Exception as error:
        problems.append(f"nvdiffrast не собирается: {error}")
        say(f"nvdiffrast: FAIL {type(error).__name__}: {str(error)[:200]}")
    try:
        import chumpy, cv2, facer, insightface, pixel3dmm  # noqa: F401
        from pytorch3d import _C  # noqa: F401
        from pytorch3d.ops import knn_points
        if torch.cuda.is_available():
            points = torch.zeros((1, 1, 3), device="cuda")
            knn_points(points, points, K=1)
            make_detector()
        say("импорты  : ok (chumpy cv2 facer insightface pixel3dmm pytorch3d)")
    except Exception as error:
        problems.append(f"импорт пакета: {error}")
    say(f"диск     : свободно {free_gb()} ГБ на /content")
    if free_gb() < float(cfg["min_free_gb"]):
        problems.append(f"свободно меньше {cfg['min_free_gb']} ГБ — освободи диск перед run")
    root = None
    try:
        root = resolve_input(cfg["input"])
        say(f"вход     : {root}")
    except ValueError as error:
        problems.append(str(error))
    out = Path(cfg["output"])
    out.mkdir(parents=True, exist_ok=True)
    say(f"выход    : {out} | {'доступен' if out.is_dir() else 'НЕ ДОСТУПЕН — проверь монтирование Drive'}")
    say(f"режим    : source={cfg.get('source', 'drive')} | unit={cfg.get('unit', 'folder')} | "
        f"папки батчей в {batches_root(cfg)} | "
        + (f"на каждое фото: {batches_root(cfg)}/<папка>/photos/<имя файла>/ "
           "строго по одному фото: полный комплект перед следующим"
           if str(cfg.get('unit', 'folder')) == 'photo' else
           f"результат на фото: {'папка по имени файла' if str(cfg.get('export', 'photos')) != 'bundle' else 'только архив'}"))
    if root is not None and root.is_dir():
        photos = sum(1 for p in root.rglob("*") if is_photo(p))
        say(f"фото     : {photos} в {root}")
    manifest = read_env_manifest(cfg)
    if manifest.get("parts"):
        gb = sum(int(p.get("bytes", 0)) for p in manifest["parts"]) / 1e9
        age = max(0.0, time.time() - float(manifest.get("ts", 0) or 0))
        say(f"слепок   : {len(manifest['parts'])} архива, {gb:.1f} ГБ, {human(age)} назад — "
            f"перезапуск среды займёт минуты (ячейка 3 распакует сама)")
    else:
        say("слепок   : OUTPUT/_env пуст — после первой успешной сборки жми ячейку freeze, "
            "и дальше каждый перезапуск среды будет в разы дешевле по квоте")
    state_dir(cfg)
    (state_dir(cfg) / "preflight.json").write_text(json.dumps(
        {"problems": problems, "flame2023": bool(use2023), "free_gb": free_gb(),
         "torch": torch.__version__, "ts": int(time.time())}, ensure_ascii=False, indent=1), encoding="utf-8")
    say("-" * 72)
    if problems:
        say("ЧТО МЕШАЕТ:")
        for item in problems:
            say("  ! " + item)
        return 1
    say("ГОТОВО: можно гнать корпус (стадия plan -> run)")
    return 0


def st_plan(cfg):
    say("=" * 72)
    say("ПЛАН — сканирование входа, фильтр лиц, staging, манифест")
    say("=" * 72)
    t0 = time.time()
    batches = build_batches(cfg)
    if cfg["face_filter"] and any(b["kind"] == "photos" for b in batches):
        say("проверяю лица FaceBoxes (одна загрузка модели на всю выборку)...")
        batches, dropped = filter_faces(batches, cfg)
        if dropped:
            (WORK / "face_dropped.json").write_text(json.dumps(dropped, ensure_ascii=False, indent=1), encoding="utf-8")
            say(f"отброшено кадров без лица/с низким скором: {len(dropped)} (список в /content/work/face_dropped.json)")
    batches = [b for b in batches if b["n"] or b["kind"] == "video"]
    unit = str(cfg.get("unit", "folder"))
    if unit == "photo":
        batches = chunk_batches(batches, cfg)
        say("unit=photo: строго одно фото за цикл — предобработка, трекинг и полный экспорт до следующего фото")
    use2023 = resolve_flame2023(cfg)
    total = sum(b["n"] for b in batches)
    save_plan(cfg, {"batches": batches, "use_flame2023": use2023, "total_photos": total, "config": cfg})
    undated = cfg.get("_undated_photos", [])
    if undated:
        write_json({"undated": undated, "count": len(undated)}, state_dir(cfg) / "undated.json")
        say(f"Без корректной даты в имени: {len(undated)} фото. Они не анализируются.")
        for name in undated[:20]:
            say("  " + name)
        if len(undated) > 20:
            say(f"  ... ещё {len(undated) - 20}")
    dated = cfg.get("_photo_count", total)
    say(f"К анализу: {dated} фото; без даты в имени (не анализируются): {len(undated)}")
    fallback = cfg.get("_date_fallback") or []
    if fallback:
        sources = Counter(item["source"] for item in fallback)
        say(f"Дата не в имени файла — взята из EXIF/времени файла у {len(fallback)} фото "
            f"({', '.join(f'{k}: {v}' for k, v in sources.items())}). Порядок обработки: по этим датам.")
        for item in fallback[:5]:
            say(f"  {item['relpath']} <- {item['source']}")
    first = batches[0]["frames"][0]["relpath"] if batches and batches[0]["frames"] else ""
    if first:
        write_json({"next_photo": {"relpath": first}, "last_completed": None,
                    "updated": int(time.time())}, state_dir(cfg) / "resume.json")
    say(f"{'батч':<26}{'кадров':>8}  {'аргументы трекинга'}")
    for b in batches:
        preview = (" ".join(photo_track_args(cfg, b["frames"][0].get("frame", "0"), use2023))
                   if unit == "photo" and b["kind"] == "photos" else track_args(cfg, b["n"], use2023))
        say(f"{b['id']:<26}{b['n']:>8}  {preview[:88]}")
    say("-" * 72)
    over = [b["id"] for b in batches if b["n"] > MAX_TRACK_FRAMES]
    if over:
        say(f"!! больше {MAX_TRACK_FRAMES} кадров — разбей папки перед run: "
            + ", ".join(over))
    say(f"батчей: {len(batches)} | кадров: {total} | flame2023: {use2023} | готовилось {human(time.time() - t0)}")
    photo_log = read_log(cfg, "photos_log.jsonl", PHOTO_LOG) if unit == "photo" else []
    per_frame = median_sec_per_frame(photo_log or read_log(cfg))
    source = ("по журналу фото" if photo_log else "по журналу Drive") if per_frame else "оценка по умолчанию"
    default_per = PHOTO_DEFAULT_S if unit == "photo" else 45.0
    say(f"ориентир по времени: ~{total * (per_frame or default_per) / 3600:.1f} ч GPU "
        f"({(per_frame or default_per):.0f} с/кадр, {source})")
    say(f"свободно на /content: {free_gb()} ГБ")
    return 1 if not batches or over else 0


def load_plan(cfg):
    for path in (PLAN_FILE, Path(cfg["output"]) / "_state/plan.json"):
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(data, dict) or not isinstance(data.get("batches"), list):
                continue
            saved = data.get("config", {})
            if any(saved.get(key) != cfg.get(key) for key in ("input", "output", "source", "subjects",
                                                              "max_batches", "face_filter", "face_min_score",
                                                              "unit", "photos_per_chunk")):
                continue
            for b in data["batches"]:
                if not re.fullmatch(r"[A-Za-z0-9_-]+", str(b.get("id", ""))) or b.get("kind") not in ("photos", "video"):
                    raise ValueError("план содержит некорректный ID/тип батча — пересоздай plan")
                b["target"] = str(STAGE / b["id"] if b["kind"] == "photos" else STAGE / (b["id"] + ".mp4"))
            return data
    return None


def st_test1(cfg):
    say("=" * 72)
    say("ТЕСТ 1 ФОТО — полный цикл на одном кадре (корпус не трогаем)")
    say("=" * 72)
    root = resolve_input(cfg["input"])
    if root.is_file() and root.suffix.lower() == ".zip":
        root = unpack_zip(root)
    video_source = None
    if root.is_file() and root.suffix.lower() in VIDEO_EXT:
        import cv2
        capture = cv2.VideoCapture(str(root))
        try:
            success, image = capture.read()
        finally:
            capture.release()
        if not success:
            say(f"видео не читается: {root}")
            return 1
        video_source = root
        root = WORK / "test1_video_frame.jpg"
        if not cv2.imwrite(str(root), image):
            raise OSError("не удалось сохранить тестовый кадр видео")
    photos = sorted(p for p in (root.rglob("*") if root.is_dir() else [root]) if is_photo(p))
    if not photos:
        say("фото не найдены — проверь INPUT")
        return 1
    src = photos[len(photos) // 2] if len(photos) > 1 else photos[0]
    say(f"тестируем: {src}")
    if str(cfg.get("unit", "folder")) == "photo":
        return test1_photo(cfg, src, video_source)
    b = {"id": "test1", "subject": "test", "angle": "one", "kind": "photos",
          "folder": str(src.parent), "frames": [{"orig": str(src), "relpath": src.name}], "n": 1}
    if video_source:
        b["frames"][0]["relpath"] = video_source.name + "#frame=0"
    stage_photos(b)
    use2023 = resolve_flame2023(cfg)
    status, err, args, sec = process_batch(cfg, b, use2023)
    actor = resolve_actor_dir("test1")
    say(f"test1 -> {status} за {human(sec)} | args: {args or '(не дошли)'}")
    if status != "ok":
        say(err)
        say(tail(LOGS / "test1.log", 30))
        return 1
    n = b.get("n_done", b["n"])
    rows, arrays = extract_params(actor, n)
    say(f"артефакты: mesh {count_files(actor / 'mesh', '.ply')} ply, "
        f"{count_files(actor / 'checkpoint', '.frame')} .frame, "
        f"ландмарок {count_files(actor / 'mesh', '.npy')} npy, "
        f"canonical.ply {(actor / 'mesh/canonical.ply').is_file()}, параметров {len(arrays)} ключей")
    say("файлы батча: " + ", ".join(sorted(p.name for p in actor.iterdir())))
    written = sync_batch(cfg, b, actor, n, sec, use2023)
    append_log(cfg, {"id": b["id"], "status": "ok", "n": n, "sec": round(sec, 1),
                     "args": args, "flame2023": use2023, "synced": written, "ts": int(time.time())})
    say("ИТОГ: УСПЕХ — можно запускать plan + run")
    return 0


def write_demo_preview(cfg, b, sec):
    """Локальное превью + репорт: ячейка «показать результат» читает /content, а не Drive —
    иначе на примонтированном Drive просмотр демо превращается в минуту ожидания."""
    src = WORK / "sync" / b["id"] / "photos"
    dst = WORK / "preview"
    shutil.rmtree(dst, ignore_errors=True)
    if src.is_dir():
        shutil.copytree(src, dst)
    entries = []
    for folder in sorted(p for p in dst.iterdir() if p.is_dir()) if dst.is_dir() else []:
        params = folder / "params.json"
        payload = json.loads(params.read_text(encoding="utf-8")) if params.is_file() else {}
        entries.append({"folder": folder.name, "frame": payload.get("frame", ""),
                        "source": payload.get("source", ""), "relpath": payload.get("relpath", ""),
                        "summary": payload.get("summary", {}),
                        "flame_keys": sorted(payload.get("flame", {})),
                        "files": sorted(p.name for p in folder.iterdir() if p.is_file())})
    report = {"batch": b["id"], "status": "ok", "sec": round(sec, 1),
              "drive": str(batch_dir(cfg, b["id"])), "preview": str(dst),
              "bundle": str(WORK / "sync" / b["id"]),
              "input": cfg["input"], "output": cfg["output"],
              "photos": entries}
    (WORK / "demo_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return report


def st_demo(cfg):
    """ДЕМО — один полный цикл на первом фото из INPUT: preprocessing → сети → трекинг → папка в Drive.
    Результат раскладывается по папкам с именами исходных фото (photos/<имя>/)."""
    say("=" * 72)
    say("ДЕМО — полный цикл на первом фото из INPUT")
    say("=" * 72)
    t0 = time.time()
    try:
        batches = build_batches(cfg, refresh=False)
    except ValueError as error:
        say(f"вход недоступен: {error}")
        return 1
    batches = [b for b in batches if b["kind"] == "photos" and b["n"]]
    if not batches:
        say(f"фото не найдены: поправь INPUT (сейчас это {cfg['input']!r}) — нужна папка с фото на Drive")
        return 1
    b = batches[0]
    limit = max(1, int(cfg.get("demo_photos", 1) or 1))
    total_in_folder = int(b["n"])
    if total_in_folder > limit:
        say(f"в первой папке {total_in_folder} фото — демо берёт первые {limit}")
        if str(cfg.get("unit", "folder")) == "photo":
            say("   unit=photo: демо считает ровно эти фото, и run их пропустит — засчитано")
        else:
            say("   demo считает батч целиком (joint-трекинг), поэтому run пересчитает эту папку заново —")
            say("   так что демо не «кусает» корпус, а проверяет цикл на малом числе кадров")
        b["frames"] = b["frames"][:limit]
        b["n"] = limit
    say(f"батч {b['id']}: " + ", ".join(Path(item["orig"]).name for item in b["frames"][:8]))
    stage_photos(b)
    use2023 = resolve_flame2023(cfg)
    if str(cfg.get("unit", "folder")) == "photo":
        return demo_photos(cfg, b, use2023)
    status, err, args, sec = process_batch(cfg, b, use2023)
    actor = resolve_actor_dir(b["id"])
    say(f"демо -> {status} за {human(sec)} | args: {args or '(не дошли)'}")
    if status != "ok":
        say(err)
        say(tail(LOGS / f"{b['id']}.log", 30))
        append_log(cfg, {"id": b["id"], "status": "fail", "n": b["n"], "sec": round(sec, 1), "args": args,
                         "flame2023": use2023, "synced": [], "err": err[:500],
                         "partial": b["n"] < total_in_folder, "n_total": total_in_folder,
                         "ts": int(time.time())})
        return 1
    n = b.get("n_done", b["n"])
    partial = n < total_in_folder          # демо считало часть папки: run обязан пересчитать её целиком
    written = sync_batch(cfg, b, actor, n, sec, use2023, photos=True)
    append_log(cfg, {"id": b["id"], "status": "ok", "n": n, "sec": round(sec, 1), "args": args,
                     "flame2023": use2023, "synced": written, "partial": partial,
                     "n_total": total_in_folder, "ts": int(time.time())})
    report = write_demo_preview(cfg, b, time.time() - t0)
    say("-" * 72)
    say(f"папка результата в Drive: {report['drive']}")
    for entry in report["photos"]:
        name = Path(entry["source"]).name if entry["source"] else f"кадр {entry['frame']}"
        say(f"  {name} -> photos/{entry['folder']}/ [{', '.join(entry['files']) or 'пусто'}]")
    say(f"превью для показа в ноутбуке: {report['preview']}")
    say("ИТОГ: цикл проходит — дальше жми plan + run по всей папке из INPUT")
    return 0


def demo_photos(cfg, b, use2023):
    """ДЕМО в unit=photo: те же независимые примерки, что и у корпуса; посчитанное засчитывается —
    run потом пропустит эти папки по done.json, пересчёт не нужен."""
    names = photo_names_map(b)
    ok, fail, done_folders = 0, 0, []
    for i, item in enumerate(b["frames"]):
        name = names[item.get("frame", f"{i:05d}")]
        status, err, sec, args, extra = fit_one_photo(cfg, b, item, name, use2023)
        folder, files, shape = photo_folder_report(cfg, b, name)
        say(f"  {Path(item['orig']).name} -> {status} за {human(sec)} -> {folder}")
        say(f"    файлы: {', '.join(files) or 'пусто'}" + (f" | shape_norm {shape}" if shape else ""))
        if status == "ok":
            ok += 1
            done_folders.append(str(folder))
        else:
            fail += 1
            say(f"    ! {err[:240]}")
    preview = WORK / "preview"
    shutil.rmtree(preview, ignore_errors=True)
    src = photos_root(cfg, b.get("out", b["id"]))
    if src.is_dir():
        shutil.copytree(src, preview)
    report = {"batch": b["id"], "unit": "photo", "status": "ok" if not fail else "fail",
              "drive": str(src), "preview": str(preview), "ok": ok, "fail": fail,
              "folders": done_folders, "input": cfg["input"], "output": cfg["output"]}
    write_json(report, WORK / "demo_report.json")
    say("-" * 72)
    say(f"превью для показа в ноутбуке: {preview}")
    say("ИТОГ: цикл проходит — жми plan + run по всей папке, эти фото уже посчитаны" if not fail
        else "ИТОГ: есть провалы — смотри log выше")
    return 1 if fail else 0


def fit_one_photo(cfg, b, item, name, use2023):
    """Один полный фото-цикл без плана и без журнала (демо и test1): предобработка + примерка + папка."""
    frame = item["frame"]
    if not preproc_ok(PREPRO / b["id"], frame):
        preprocess_subset(cfg, b, [frame])
    sync_preproc(cfg, b, frame, name)
    return process_photo(cfg, b, item, name, use2023)


def photo_folder_report(cfg, b, name):
    """Что легло в папку фото и чем этот каноникал отличается от чужого (shape-норма)."""
    folder = photo_dir(cfg, b.get("out", b["id"]), name)
    files = sorted(p.name for p in folder.iterdir() if p.is_file()) if folder.is_dir() else []
    shape = ""
    params = folder / "params.json"
    if params.is_file():
        try:
            shape = str(json.loads(params.read_text(encoding="utf-8")).get("summary", {}).get("shape_norm", ""))
        except (OSError, ValueError):
            shape = ""
    return folder, files, shape


def test1_photo(cfg, src, video_source):
    """test1 в unit=photo — ровно тот цикл, что и для корпуса, но на одном снимке и без журнала."""
    rel = video_source.name + "#frame=0" if video_source else src.name
    b = {"id": "test1", "out": "test1", "subject": "test", "angle": "one", "kind": "photos",
         "folder": str(src.parent), "frames": [{"orig": str(src), "relpath": rel}], "n": 1}
    stage_photos(b)
    use2023 = resolve_flame2023(cfg)
    frame = b["frames"][0]["frame"]
    name = photo_names_map(b)[frame]
    status, err, sec, args, extra = fit_one_photo(cfg, b, b["frames"][0], name, use2023)
    say(f"test1 -> {status} за {human(sec)} | args: {' '.join(args)}")
    if status != "ok":
        say(err)
        say(tail(LOGS / f"test1__{frame}.log", 30))
        return 1
    folder, files, shape = photo_folder_report(cfg, b, name)
    say(f"папка фото: {folder}")
    say("файлы: " + ", ".join(files))
    holes = [n for n in PHOTO_REQUIRED if not nonempty_file(folder / n)]
    if holes:
        say(f"не хватает: {holes}")
        return 1
    say(f"каноническая модель этого фото: shape_norm {shape} — она своя на каждый снимок")
    say("ИТОГ: цикл проходит — дальше жми plan + run по корпусу")
    return 0


def ensure_staged(b):
    """После перезапуска рантайма /content пуст: раскладываем фото заново по плану.
    Нумерация кадров детерминирована (тот же порядок b["frames"]), манифест перезаписывается."""
    target = Path(b.get("target") or (STAGE / b["id"]))
    if b["kind"] == "video":
        if not target.is_file():
            try:
                stage_video(b, Path(b["folder"]))
            except (OSError, ValueError):
                return False
        return Path(b["target"]).is_file()
    expected = {f"{i:05d}.jpg" for i in range(b["n"])}
    staged = {p.name for p in target.glob("*.jpg") if p.is_file()}
    manifest = read_manifest(b["id"])
    if not (staged == expected and len(manifest) == b["n"]
            and all(manifest.get(f"{i:05d}", {}).get("orig_path") == item["orig"]
                    for i, item in enumerate(b["frames"]))):
        original_frames, original_n = b["frames"], b["n"]
        stage_photos(b)
        if b["n"] == 0:
            b["frames"], b["n"] = original_frames, original_n
            return False
    return b["n"] > 0


# --------------------------------------------------------- unit=photo: папка и прогон на каждое фото

def chunk_batches(batches, cfg):
    """unit=photo: пачка предобработки = чанк не больше photos_per_chunk, а результат всё равно
    лежит в папке исходной папки (b["out"]), чтобы фото не разъезжались по разным местам."""
    size = 1  # полный цикл каждого фото завершается до начала следующего
    out = []
    for b in batches:
        out_id = b.setdefault("out", b["id"])      # папка результата = исходная папка, а не чанк
        if b["kind"] != "photos" or b["n"] <= size:
            out.append(b)
            continue
        for j in range(0, b["n"], size):
            piece = dict(b)
            piece["frames"] = b["frames"][j:j + size]
            piece["n"] = len(piece["frames"])
            piece["out"] = out_id
            piece["id"] = f"{out_id}__c{1 + j // size:02d}"
            piece["target"] = str(STAGE / piece["id"])        # чанк стейджится сам, по своим номерам
            piece.pop("track_args", None)
            out.append(piece)
    return out


def photos_root(cfg, out_id):
    """Папка фото исходной папки: без синка в Drive — локально в /content/work, чтобы тесты не трогали Drive."""
    base = batches_root(cfg) if cfg.get("sync_to_drive", True) else WORK / "sync"
    return base / out_id / "photos"


def photo_dir(cfg, out_id, name):
    return photos_root(cfg, out_id) / name


def photo_key(out_id, name):
    return f"{out_id}/photos/{name}"


def photo_done(folder):
    """Фото считается готовым, только если в папке есть все обязательные файлы и done.json —
    именно по этому признаку run пропускает фото после сброса среды."""
    folder = Path(folder)
    return folder.is_dir() and all(nonempty_file(folder / name) for name in PHOTO_REQUIRED)


def photo_names_map(b):
    """Номер кадра чанка -> имя папки фото (та же логика, что и в joint-экспорте)."""
    rows = [{"frame": item.get("frame", f"{i:05d}"), "orig_relpath": item.get("relpath", ""),
             "orig_path": item.get("orig", "")} for i, item in enumerate(b["frames"])]
    return photo_names(rows)


def preproc_files(base, frame, mica=True):
    """(имя в папке фото, путь в предобработке) для одного кадра — то, что трекер читает по номеру.

    mica=False отдаёт только ядро: MICA делает identity по подвыборке кадров, поэтому отсутствие
    своей папки mica — не пропуск, а просто инициализация формы средним по чанку."""
    base = Path(base)
    out = [(name, base / sub / f"{frame}{ext}") for name, sub, ext in PHOTO_CORE]
    if not mica:
        return out
    out += [("arcface.npy", base / "arcface" / f"{frame}.npy")]
    out += [(name, base / "mica" / frame / rel) for name, rel in PHOTO_MICA]
    return out


def preproc_ok(base, frame):
    return all(nonempty_file(path) for _, path in preproc_files(base, frame, mica=True))


def resolve_run_dir(run_root):
    """output_folder=<TRACK>/<чанок>__<кадр>, а внутри ещё каталог актёра с суффиксами конфига."""
    run_root = Path(run_root)
    if not run_root.is_dir():
        return None
    if (run_root / "mesh").is_dir():
        return run_root
    cands = [d for d in run_root.iterdir() if d.is_dir() and (d / "mesh").is_dir()]
    return max(cands, key=lambda d: d.stat().st_mtime) if cands else None


def photo_missing(actor, frame):
    """Чего не довёз трекер по этому кадру (пустой список = кадр полный)."""
    holes = []
    if not nonempty_file(actor / "mesh" / f"{frame}.ply"):
        holes.append("mesh.ply")
    if not nonempty_file(actor / "mesh" / "canonical.ply"):
        holes.append("canonical.ply")
    if not nonempty_file(actor / "checkpoint" / f"{frame}.frame"):
        holes.append("checkpoint.frame")
    return holes


def photo_track_args(cfg, frame, use2023):
    """Аргументы независимой примерки одного кадра: ровно один кадр, свои итерации, свои ландмарки."""
    return (f"start_frame={int(frame)}", "num_frames=1", f"iters={int(cfg['single_iters'])}",
            "batch_size=1", "save_landmarks=True",
            f"use_flame2023={'True' if use2023 else 'False'}")


def restore_preproc(cfg, b, need):
    """Достаёт предобработку кадров из их папок в Drive — после сброса среды её не пересчитываем.
    Верит только метке preproc.json своего чанка: сдвинулась нумерация — считаем заново."""
    base = PREPRO / b["id"]
    got = set()
    for frame, name in need:
        folder = photo_dir(cfg, b.get("out", b["id"]), name)
        marker = folder / "preproc.json"
        if not marker.is_file():
            continue
        try:
            meta = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if str(meta.get("cid")) != b["id"] or str(meta.get("frame")) != str(frame):
            continue
        if not all(nonempty_file(folder / photo_name)
                   for photo_name, _ in preproc_files(base, frame, mica=False)):
            continue
        for photo_name, src in preproc_files(base, frame):        # MICA может не быть — кладём что есть
            if nonempty_file(src) or not nonempty_file(folder / photo_name):
                continue
            src.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(folder / photo_name, src)
        got.add(frame)
    return got


def preprocess_subset(cfg, b, frames):
    """Предобработка только нужных кадров, раскладка по АБСОЛЮТным номерам чанка.

    Пересчитывать весь чанк после каждого сброса среды — значит каждый раз платить десятки минут
    на уже готовые фото; поэтому во временный батч кладывается только недостающее, а результаты
    переносятся под своими номерами, и нумерация start_frame не съезжает. Временных батчей всегда
    несколько по <= MICA_GROUP кадров: иначе MICA проредит кадры и часть фото останется без своего
    mica_identity, а >10 кадров за один вызов mica_demo.py обрабатывает как раз только с прореживанием."""
    by_frame = {item["frame"]: item for item in b["frames"]}
    subset = [by_frame[f] for f in frames if f in by_frame]
    if not subset:
        return
    base = PREPRO / b["id"]
    base.mkdir(parents=True, exist_ok=True)
    log = LOGS / f"{b['id']}.log"
    timeout = float(cfg.get("batch_timeout_min", 0)) or None
    for j in range(0, len(subset), MICA_GROUP):
        group = subset[j:j + MICA_GROUP]
        want = [item["frame"] for item in group]          # stage_photos перенумерует кадры внутри группы
        tid = f"{b['id']}__pre{j // MICA_GROUP:02d}"
        tmp = {"id": tid, "subject": b["subject"], "angle": b["angle"], "kind": "photos",
               "folder": b["folder"], "frames": [dict(i) for i in group], "n": len(group)}
        stage_photos(tmp)
        shutil.rmtree(PREPRO / tid, ignore_errors=True)
        for cmd, cwd in preproc_commands(cfg, tid, tmp["target"]):
            rc, why = run_cmd(cmd, log, timeout_min=timeout, cwd=cwd)
            if rc:
                raise RuntimeError(f"предобработка ({Path(cmd[1]).name}) {tid}: {why}; {tail(log, 10)}")
        src = PREPRO / tid
        for index, tmp_item in enumerate(tmp["frames"]):
            here = f"{index:05d}"
            dst_frame = tmp_item.get("orig_frame")
            if dst_frame is None:
                continue
            for (_, dst), (_, source) in zip(preproc_files(base, dst_frame), preproc_files(src, here)):
                if nonempty_file(source):
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, dst)
            shutil.rmtree(base / "mica" / dst_frame, ignore_errors=True)
            if (src / "mica" / here).is_dir():
                shutil.copytree(src / "mica" / here, base / "mica" / dst_frame)
        shutil.rmtree(src, ignore_errors=True)
        shutil.rmtree(STAGE / tid, ignore_errors=True)


def sync_preproc(cfg, b, frame, name):
    """Кладёт предобработку кадра в папку фото и ставит метку чанка — это и страховка от сброса среды."""
    folder = photo_dir(cfg, b.get("out", b["id"]), name)
    copied = []
    for photo_name, path in preproc_files(PREPRO / b["id"], frame):
        if nonempty_file(path):
            copy_file(path, folder / photo_name)
            copied.append(photo_name)
    write_json({"cid": b["id"], "frame": frame, "relpath": name, "items": sorted(copied),
                "ts": int(time.time())}, folder / "preproc.json")
    return copied


def seg_masks(seg_path, size=MASK_SIZE):
    """Маски наблюдаемости по ровно тем правилам классов, что и read_data(): fg / uv / normal.

    Считается по уже сохранённой segmentation.png, поэтому папку фото можно дополнить ими и потом,
    без GPU и без пересчёта трекинга."""
    import numpy as np
    from PIL import Image
    with Image.open(seg_path) as im:
        seg = np.asarray(im.convert("L").resize((size, size), Image.NEAREST))
    return {"fg": np.isin(seg, SEG_FG), "uv": np.isin(seg, SEG_UV), "normal": np.isin(seg, SEG_NORMAL)}


def write_visibility(folder, size=MASK_SIZE):
    """Фиксирует, какая сторона лица реально наблюдалась на этом снимке — чтобы сравнение потом
    шло по пересечению видимых областей, а не по всей модели.

    left/right здесь — стороны КАДРА (как видит зритель), а не стороны головы: их связь с анатомией
    даёт pose.yaw_deg / pose.bin из params.json, а не признак ниже. Всё берётся из segmentation.png
    и landmarks_pipnet.npy этой же папки."""
    import numpy as np
    from PIL import Image
    folder = Path(folder)
    if not nonempty_file(folder / "segmentation.png"):
        return {"error": "нет segmentation.png — наблюдаемость не зафиксирована"}
    try:
        masks = seg_masks(folder / "segmentation.png", size)
    except Exception as error:      # битая сегментация не имеет права ронять экспорт уже посчитанного фото
        return {"error": f"segmentation.png не читается: {type(error).__name__}"}
    for key, mask in masks.items():
        Image.fromarray((mask.astype(np.uint8)) * 255, mode="L").save(folder / f"mask_{key}.png")
    out = {"size": size, "masks": [f"mask_{k}.png" for k in masks],
           "fg_fraction": round(float(masks["fg"].mean()), 4),
           "uv_fraction": round(float(masks["uv"].mean()), 4),
           "normal_fraction": round(float(masks["normal"].mean()), 4)}
    ys, xs = np.nonzero(masks["fg"])
    if len(xs):
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
        out["fg_box"] = [x0, y0, x1, y1]
        mid = (x0 + x1) // 2
        for side, sl in (("left", slice(x0, max(x0 + 1, mid))), ("right", slice(mid, x1 + 1))):
            fg_part = int(masks["fg"][y0:y1 + 1, sl].sum())
            uv_part = int(masks["uv"][y0:y1 + 1, sl].sum())
            out[f"{side}_uv_of_fg"] = round(uv_part / fg_part, 4) if fg_part else 0.0
        out["visible_side"] = max(("left", "right"), key=lambda s: out.get(f"{s}_uv_of_fg", 0.0))
    lms_path = folder / "landmarks_pipnet.npy"
    if not nonempty_file(lms_path):
        return out
    try:
        lms = np.load(lms_path, allow_pickle=False).astype(np.float64)
        if lms.ndim != 2 or lms.shape[1] < 2 or not len(lms):
            return out
    except Exception as error:
        out["points_error"] = f"landmarks_pipnet.npy не читается: {type(error).__name__}"
        return out
    inside = (lms[:, 0] >= 0) & (lms[:, 0] <= 1) & (lms[:, 1] >= 0) & (lms[:, 1] <= 1)
    px = np.clip(np.rint(lms * (size - 1)), 0, size - 1).astype(int)
    hit_fg = np.where(inside, masks["fg"][px[:, 1], px[:, 0]], False).astype(np.uint8)
    hit_uv = np.where(inside, masks["uv"][px[:, 1], px[:, 0]], False).astype(np.uint8)
    np.save(folder / "visibility_points.npy",
            np.stack([px[:, 0], px[:, 1], hit_fg, hit_uv], axis=1).astype(np.float32))
    out["pipnet_points"] = int(len(px))
    out["points_out_of_frame"] = int((~inside).sum())
    zones = {}
    for index_range, name in PIPNET_ZONES:
        idx = [i for i in index_range if i < len(px)]
        if not idx:
            continue
        zones[name] = {"points": len(idx),
                       "in_silhouette": int(hit_fg[idx].sum()),
                       "in_uv": int(hit_uv[idx].sum()),
                       "fraction_uv": round(float(hit_uv[idx].mean()), 3)}
    out["zones"] = zones
    if len(px) > 45 and inside[30] and inside[36] and inside[45]:
        outer_l, outer_r = int(px[36, 0]), int(px[45, 0])       # внешние уголки правого и левого глаза
        span = outer_r - outer_l
        if abs(span) > 8:
            # 2D-диагностика кривой подгонки (нос относительно середины глаз), не ярлык ракурса:
            # метку ракурса даёт только pose по матрице поворота и девяти бинам корпуса
            out["yaw_proxy"] = round(float((px[30, 0] - (outer_l + outer_r) / 2.0) / span), 4)
    return out


def ensure_visibility(folder):
    """Дописывает в уже готовую папку фото то, что считается на CPU: маски наблюдаемости и метку ракурса.
    Пересчёт трекинга и torch не нужны (поза берётся из матриц, которые уже лежат в params.json),
    поэтому collect может довести старые корпуса без нового прогона."""
    params = Path(folder) / "params.json"
    if not params.is_file():
        return False
    try:
        payload = json.loads(params.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    changed = False
    if not (payload.get("visibility") and nonempty_file(Path(folder) / "mask_uv.png")) \
            and nonempty_file(Path(folder) / "segmentation.png"):
        payload["visibility"] = write_visibility(folder)
        changed = True
    pose = payload.get("pose") or {}
    if not pose.get("bin"):            # старые папки: учим ракурс по матрицам, которые уже лежат в params.json
        fresh = pose_of_payload(payload)
        if not fresh.get("error"):
            payload["pose"] = fresh
            changed = True
    if not changed:
        return False
    payload["files"] = sorted(p.name for p in Path(folder).iterdir() if p.is_file())
    write_json(payload, params)
    return True


def export_photo(cfg, b, item, name, actor, sec, args, use2023):
    """Пишет в папку фото всё, что даёт библиотека: матрицы поворота и ракурс по ним, каноническую
    форму, параметры, наблюдаемость, ландмарки."""
    folder = photo_dir(cfg, b.get("out", b["id"]), name)
    frame = item["frame"]
    row, one = frame_params(actor, frame)
    for key in ("orig", "relpath", "face_score", "note"):
        if item.get(key) is not None:
            row[key] = item.get(key)
    source = Path(item.get("orig") or "")
    if source.is_file():
        copy_file(source, folder / f"source{source.suffix.lower() or '.jpg'}")
    for label, src in (("mesh.ply", actor / "mesh" / f"{frame}.ply"),
                       ("canonical.ply", actor / "mesh" / "canonical.ply"),
                       ("checkpoint.frame", actor / "checkpoint" / f"{frame}.frame")):
        if nonempty_file(src):
            copy_file(src, folder / label)
    for label, src in landmark_artifacts(actor, frame):
        if nonempty_file(src):
            copy_file(src, folder / label)
    target_size = None
    if source.is_file():
        try:
            import cv2
            original = cv2.imread(str(source))
            if original is not None:
                target_size = (original.shape[1], original.shape[0])
        except ImportError:
            pass
    overlay_frame(actor / "result.mp4", folder / "overlay.jpg", 0, target_size)
    payload = photo_payload(b, row, {key: value for key, value in one.items()}, 0)
    payload["unit"] = "photo"
    payload["canonical"] = "canonical.ply"
    payload["run"] = {"args": " ".join([f"video_name={b['id']}"] + list(args)), "sec": round(sec, 1),
                      "chunk": b["id"], "frame": frame}
    payload["tracking_config"] = effective_config(cfg, dict(b, track_args=" ".join(args)), 1, use2023)
    payload["mica_own"] = nonempty_file(folder / "mica_identity.npy")
    # кривые потерь (tensorboard) живут только в папке прогона и удаляются вместе с ней: это единственный
    # измеримый признак того, насколько меш сел именно на этом фото
    events = sorted((actor / "logs").glob("events*")) if (actor / "logs").is_dir() else []
    if events:
        copy_file(events[0], folder / "tb_events.tfevents")
        payload["loss_curves"] = "tb_events.tfevents"
    payload["visibility"] = write_visibility(folder)
    payload["files"] = sorted(p.name for p in folder.iterdir() if p.is_file())
    write_json(payload, folder / "params.json")
    run_log = LOGS / f"{b['id']}__{frame}.log"
    if run_log.is_file():
        copy_file(run_log, folder / "run.log")
    return folder, row, payload


def process_photo(cfg, b, item, name, use2023):
    """Одно фото: независимая примерка -> экспорт в свою папку -> done.json -> запись в журнал."""
    frame = item["frame"]
    args = photo_track_args(cfg, frame, use2023)
    run_root = TRACK / f"{b['id']}__{frame}"
    log = LOGS / f"{b['id']}__{frame}.log"
    # Если предыдущий запуск упал после дорогого трекинга, не удаляем уже посчитанный actor:
    # сначала проверяем, можно ли экспортировать его повторно без нового GPU.
    actor = resolve_run_dir(run_root)
    holes = photo_missing(actor, frame) if actor else ["нет выходного каталога трекинга"]
    rc, why, sec = 0, "", 0.0
    if holes:
        shutil.rmtree(run_root, ignore_errors=True)
        log.unlink(missing_ok=True)
        t0 = time.time()
        rc, why = run_cmd(["python", "scripts/track.py", f"video_name={b['id']}",
                           f"output_folder={run_root}"] + list(args), log,
                          timeout_min=float(cfg.get("photo_timeout_min", 15)))
        sec = time.time() - t0
        actor = resolve_run_dir(run_root)
        holes = photo_missing(actor, frame) if actor else ["нет выходного каталога трекинга"]
    # Лог трекинга сразу кладём на Drive, иначе при сбросе runtime он исчезает вместе с /content/logs.
    folder = photo_dir(cfg, b.get("out", b["id"]), name)
    folder.mkdir(parents=True, exist_ok=True)
    if log.is_file():
        try:
            copy_file(log, folder / "run.log")
        except OSError as error:
            say(f"  {name}: run.log не сохранён на Drive: {error}")
    if rc or holes:
        return "fail", (why or "") + (" | нет: " + ", ".join(holes) if holes else ""), sec, args, {}
    try:
        folder, row, payload = export_photo(cfg, b, item, name, actor, sec, args, use2023)
        vis, pose = payload["visibility"], payload["pose"]
        if not cfg.get("keep_preproc", True):
            shutil.rmtree(PREPRO / b["id"] / "mica" / frame, ignore_errors=True)
        try:
            done_payload = {"key": photo_key(b.get("out", b["id"]), name), "batch": b.get("out", b["id"]),
                            "chunk": b["id"], "frame": frame, "photo": name,
                            "source": item.get("orig", ""), "relpath": item.get("relpath", ""),
                            "sec": round(sec, 1), "args": " ".join([f"video_name={b['id']}"] + list(args)),
                            "files": sorted(p.name for p in folder.iterdir() if p.is_file()),
                            "has_canonical": (folder / "canonical.ply").is_file(),
                            "pipeline": "unit=photo", "ts": int(time.time())}
            write_json(done_payload, folder / "done.json")
        except Exception as error:
            raise RuntimeError(f"done.json не записан: {type(error).__name__}: {error}") from error
    except Exception as error:
        return "fail", f"экспорт папки фото: {type(error).__name__}: {error}", sec, args, {}
    shutil.rmtree(run_root, ignore_errors=True)
    return "ok", "", sec, args, {"summary": {key: row.get(key, "") for key in
                                             ("jaw_norm", "exp_norm", "shape_norm", "t", "fl", "pp")},
                                 "yaw_deg": pose.get("yaw_deg", ""), "pitch_deg": pose.get("pitch_deg", ""),
                                 "roll_deg": pose.get("roll_deg", ""), "view_bin": pose.get("bin") or "",
                                 "pose_error": pose.get("error", ""),
                                 "visible_side": vis.get("visible_side", ""),
                                 "yaw_proxy": vis.get("yaw_proxy", ""),
                                 "uv_fraction": vis.get("uv_fraction", "")}


def photo_todo(cfg, b, records, attempts):
    """Кадры чанка, которые ещё не посчитаны: готовая папка с done.json пропускается насовсем."""
    names = photo_names_map(b)
    out, skipped, exhausted = [], 0, []
    for i, item in enumerate(b["frames"]):
        frame = item.get("frame", f"{i:05d}")
        name = names[frame]
        key = photo_key(b.get("out", b["id"]), name)
        if photo_done(photo_dir(cfg, b.get("out", b["id"]), name)):
            skipped += 1
            continue
        if int(records.get(key, {}).get("fails", 0)) >= attempts:
            exhausted.append(name)
            continue
        out.append((item, name, key))
    return out, skipped, exhausted


def photo_record(cfg, records, b, item, name, key, status, err="", sec=0.0, args_text="", extra=None):
    """Одна запись в photos_log.jsonl: по ней status/collect видят прогресс, а fails — сколько раз
    фото уже пробовали (после cfg["attempts"] неудач его больше не трогаем)."""
    previous = int(records.get(key, {}).get("fails", 0) or 0)
    rec = {"key": key, "id": b["id"], "out": b.get("out", b["id"]), "photo": name,
           "frame": item.get("frame", ""), "relpath": item.get("relpath", ""), "n": 1,
           "log": f"{b['id']}__{item.get('frame', '')}", "status": status,
           "fails": previous + (0 if status == "ok" else 1), "err": str(err)[:400],
           "sec": round(sec, 1), "args": args_text, "ts": int(time.time())}
    rec.update(extra or {})
    append_log(cfg, rec, "photos_log.jsonl", PHOTO_LOG)
    records[key] = rec
    return rec


def run_photos(cfg, plan, batches, use2023):
    """Цикл unit=photo: по одной примерке на фото, журнал и пропуск по папкам фото, без архивов."""
    name_log, local_log = "photos_log.jsonl", PHOTO_LOG
    records = {r["key"]: r for r in read_log(cfg, name_log, local_log) if r.get("key")}
    resume = {}
    try:
        resume = json.loads((state_dir(cfg) / "resume.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        resume = {}
    if resume.get("next_photo", {}).get("relpath"):
        say(f"Продолжение с: {resume['next_photo']['relpath']}")

    def write_resume(current=None, completed=None, nxt=None):
        payload = {"updated": int(time.time())}
        if current:
            payload["current"] = {"relpath": current, "ts": int(time.time())}
        if completed:
            payload["last_completed"] = {"relpath": completed, "ts": int(time.time())}
        if nxt:
            payload["next_photo"] = {"relpath": nxt}
        try:
            write_json(payload, state_dir(cfg) / "resume.json")
        except OSError as error:
            say(f"Точка продолжения не записана: {error}")

    attempts = max(1, int(cfg.get("attempts", 3)))
    work, total_done, gave_up = [], 0, 0
    for b in batches:
        if not ensure_staged(b):
            say(f"{b['id']}: нет исходников — проверь INPUT и пересобери plan")
            continue
        todo, skipped, exhausted = photo_todo(cfg, b, records, attempts)
        total_done += skipped
        for name in exhausted:
            say(f"  {name}: пробовали {attempts} раз(а) и бросили — правка в журнале {name_log}")
        gave_up += len(exhausted)
        if todo or exhausted:
            work.append((b, todo))
    left = sum(len(todo) for _, todo in work)
    say("=" * 72)
    say(f"КОРПУС (unit=photo): фото всего {sum(b['n'] for b in batches)} | готово {total_done} | "
        f"считаем {left} | свободно {free_gb()} ГБ | flame2023={use2023}")
    say("Обработка строго по одному фото: предобработка → трекинг → полный экспорт на Drive → следующее фото")
    say("=" * 72)
    if not left:
        if gave_up:
            say(f"считать нечего: готово {total_done}, брошено после {attempts} попыток ещё {gave_up} фото "
                f"(правь fails в OUTPUT/_state/{name_log}, если их надо повторить)")
            return 1
        say("всё посчитано — ничего не делаем; свежие результаты смотри в status/collect")
        return 0

    per_photo = median_sec_per_frame(records.values())
    deadline = session_deadline(cfg)
    max_photos = int(cfg.get("max_photos_per_run", 500) or 500)
    ok_n, fail_n, new_attempts, stop_run = 0, 0, 0, False
    queue = [(photo_sort_moment(Path(item.get("orig", ""))) or datetime.min,
              b, item, name, key) for b, todo in work for item, name, key in todo]
    queue.sort(key=lambda x: (x[0], x[2].get("relpath", "")))
    for index, (_, b, item, name, key) in enumerate(queue):
        if new_attempts >= max_photos:
            say(f"лимит запуска: {max_photos} фото — остальное в следующий запуск")
            break
        if STOP_FILE.exists():
            say(f"найден {STOP_FILE} — останавливаюсь перед {name}")
            break
        if time.time() > deadline:
            say(f"бюджет сессии исчерпан — стоп перед {name}")
            break
        est = (per_photo or float(cfg.get("photo_timeout_min", 15)) * 60) + GUARD_MARGIN_S
        if est > deadline - time.time():
            say(f"не начинаю {name}: недостаточно времени по бюджету сессии")
            break

        base = PREPRO / b["id"]
        folder = photo_dir(cfg, b.get("out", b["id"]), name)
        if photo_done(folder):
            continue
        frame = item.get("frame", "00000")
        if not preproc_ok(base, frame):
            progress(phase="Предобработка фото", current=name,
                     stats=f"фото {index + 1}/{len(queue)} · до этого полностью готово {total_done + ok_n}")
            say(f"[фото {index + 1}/{len(queue)}] {name}: предобработка (кроп, PIPNet, MICA, нормали, UV)")
            restored = restore_preproc(cfg, b, [(frame, name)])
            if frame not in restored and not preproc_ok(base, frame):
                try:
                    preprocess_subset(cfg, b, [frame])
                except (RuntimeError, OSError, ValueError) as error:
                    photo_record(cfg, records, b, item, name, key, "fail", err=str(error)[:400])
                    fail_n += 1
                    say(f"  {name}: предобработка не удалась: {str(error)[:300]} — останавливаюсь, чтобы не перескакивать фото")
                    break
            if not preproc_ok(base, frame):
                holes = [p.name for _, p in preproc_files(base, frame, mica=False) if not nonempty_file(p)]
                photo_record(cfg, records, b, item, name, key, "fail", err="предобработка: нет " + ", ".join(holes[:6]))
                fail_n += 1
                say(f"  {name}: предобработка не полная — останавливаюсь, чтобы не перескакивать фото")
                break
            sync_preproc(cfg, b, frame, name)

        progress(phase="Трекинг фото", current=name,
                 stats=f"фото {index + 1}/{len(queue)} · полностью готово {total_done + ok_n} · осталось {len(queue)-index}")
        say(f"[фото {index + 1}/{len(queue)}] {name}: трекинг и экспорт")
        write_resume(current=item.get("relpath", name))
        new_attempts += 1
        status, err, sec, args, extra = process_photo(cfg, b, item, name, use2023)
        photo_record(cfg, records, b, item, name, key, status, err=err, sec=sec,
                     args_text=" ".join(args), extra=dict(extra, canonical=status == "ok"))
        if status == "ok":
            ok_n += 1
            total_done += 1
            per_photo = per_photo or sec
            write_resume(completed=item.get("relpath", name))
            say(f"  {name} → полный комплект сохранён на Drive за {human(sec)}")
        else:
            fail_n += 1
            say(f"  {name} → ошибка за {human(sec)}: {err[:200]}")
            # Не переходить к следующему фото: пользователь просил завершать комплекты строго по очереди.
            break
    save_plan(cfg, plan)
    say("-" * 72)
    say(f"сессия: полностью готово {ok_n} | провалов {fail_n}; готовые папки с done.json пропускаются")
    save_plan(cfg, plan)
    say("-" * 72)
    say(f"сессия: ok {ok_n} | провалов {fail_n}. Жми ту же ячейку run — посчитанные фото пропускаются "
        f"по папкам (done.json) и по журналу OUTPUT/_state/{name_log}")
    return 1 if fail_n else 0


def run_folders(cfg, plan, use2023, batches=None):
    """Joint-режим: один прогон на всю папку (для видео другого и не бывает)."""
    batches = plan["batches"] if batches is None else batches
    records = {r["id"]: r for r in read_log(cfg)}
    done, sync_failed = set(), set()
    for b in batches:
        rec = records.get(b["id"], {})
        if rec.get("status") not in ("ok", "ok_no_sync"):
            continue
        if rec.get("partial"):
            say(f"{b['id']}: демо посчитало {rec.get('n')} из {b['n']} кадров — joint-трекинг не дополняется, "
                f"пересчитываю батч целиком")
            continue
        folder = (batch_dir(cfg, b["id"]) if cfg.get("sync_to_drive", True)
                  else WORK / "sync" / b["id"])
        if rec.get("status") == "ok" and not bundle_missing(folder, rec.get("synced", []), cfg):
            done.add(b["id"])
            continue
        actor = resolve_actor_dir(b["id"])
        n = int(rec.get("n", 0))
        if actor and n > 0 and not track_missing(actor, n):
            say(f"досылаю результат {b['id']} (в выходной папке он неполный)")
            manifest = read_manifest(b["id"])
            if b["kind"] == "photos" and len(manifest) == n:
                b["frames"] = [{"orig": row["orig_path"], "relpath": row["orig_relpath"],
                                "frame": row["frame"], "note": row.get("note", ""),
                                "score": row.get("face_score", "")} for row in manifest.values()]
                b["n"] = n
            b["n_done"] = n
            rec = dict(rec)
            try:
                b["track_args"] = rec.get("args") or b.get("track_args", track_args(cfg, n, use2023))
                rec["synced"] = sync_batch(cfg, b, actor, n, rec.get("sec", 0), rec.get("flame2023", use2023))
                rec["status"] = "ok"
                rec["err"] = ""
                done.add(b["id"])
            except Exception as error:
                rec["status"] = "ok_no_sync"
                rec["err"] = f"результат есть, но не записан: {type(error).__name__}: {error}"[:500]
                sync_failed.add(b["id"])
                say(f"  {b['id']}: {rec['err']}")
            append_log(cfg, rec)
            save_plan(cfg, plan)
        else:
            say(f"{b['id']}: полного результата нет ни в выходной папке, ни на /content — пересчитываю")
    todo = [b for b in batches if b["id"] not in (done | sync_failed)]
    say("=" * 72)
    say(f"КОРПУС: батчей {len(batches)} | готово {len(done)} | в работе {len(todo)} | "
        f"кадров {sum(b['n'] for b in todo)} | свободно {free_gb()} ГБ | flame2023={use2023}")
    say(f"бюджет сессии {cfg['session_max_hours']} ч, рантайм живёт {session_elapsed_hours():.1f} ч "
        f"из {cfg['session_hard_limit_hours']} ч, стоп-файл: {STOP_FILE} (создай, чтобы остановить цикл)")
    say("=" * 72)
    if not todo:
        say("все батчи уже в логе — ничего не делаем; свежие результаты смотри в status/collect")
        return 1 if sync_failed else 0
    if str(cfg["input"]).lower().endswith(".zip") and any(
            not Path(item["orig"]).is_file() for b in todo for item in b["frames"]):
        root = unpack_zip(resolve_input(cfg["input"]))
        for b in todo:
            for item in b["frames"]:
                item["orig"] = str(root / item["relpath"])
    deadline = session_deadline(cfg)
    per_frame = median_sec_per_frame(records.values())
    failed = len(sync_failed)
    ok_now = 0
    for k, b in enumerate(todo, 1):
        if STOP_FILE.exists():
            say(f"[{k}/{len(todo)}] найден {STOP_FILE} — останавливаюсь на {b['id']}")
            break
        if time.time() > deadline:
            say(f"[{k}/{len(todo)}] бюджет сессии исчерпан — стоп на {b['id']}")
            break
        if free_gb() < float(cfg["min_free_gb"]):
            say(f"[{k}/{len(todo)}] свободно {free_gb()} ГБ (< {cfg['min_free_gb']}) — стоп на {b['id']}")
            break
        # батч нельзя прервать на середине и дополнить позже, поэтому не начинаем тот,
        # который заведомо не влезает в остаток сессии — иначе потеряем и его целиком, и квоту
        est = per_frame * b["n"] + GUARD_MARGIN_S if per_frame else float(cfg["batch_timeout_min"]) * 60
        left = deadline - time.time()
        if est > left:
            say(f"[{k}/{len(todo)}] не начинаю {b['id']} ({b['n']} кадр(ов)): оценка {human(est)} "
                f"больше остатка {human(max(left, 0.0))} — дождись свежей среды и жми run ещё раз")
            break
        say(f"[{k}/{len(todo)}] {b['id']} ({b['n']} кадр(ов)) старт, свободно {free_gb()} ГБ")
        planned = b["n"]
        if not ensure_staged(b):
            rec = {"id": b["id"], "n": 0, "planned": planned, "status": "fail",
                   "err": f"нет исходников для {b['id']} (проверь INPUT / пересобери plan)",
                   "args": "", "sec": 0.0, "flame2023": bool(use2023), "ts": int(time.time()),
                   "synced": [], "notes": []}
            append_log(cfg, rec)
            failed += 1
            say(f"[{k}/{len(todo)}] {b['id']} -> fail: {rec['err']}")
            continue
        started = time.time()
        try:
            status, err, args, sec = process_batch(cfg, b, use2023)
        except Exception as error:
            status, err, args, sec = "fail", f"{type(error).__name__}: {error}", "", time.time() - started
        n_done = b.get("n_done", 0)
        plan["total_photos"] = sum(batch["n"] for batch in plan["batches"])
        save_plan(cfg, plan)
        rec = {"id": b["id"], "n": n_done, "planned": planned, "status": status, "err": err[:500],
               "args": args, "sec": round(sec, 1), "flame2023": bool(use2023), "ts": int(time.time()),
               "synced": [], "notes": [i["relpath"] for i in b["frames"] if i.get("note")][:20]}
        if status == "ok":
            actor = resolve_actor_dir(b["id"])
            try:
                rec["synced"] = sync_batch(cfg, b, actor, n_done, sec, use2023)
            except Exception as error:
                import traceback
                traceback.print_exc()
                status = "ok_no_sync"
                rec["status"] = status
                rec["err"] = f"результат есть, но в Drive не записан: {type(error).__name__}: {error}"[:500]
        append_log(cfg, rec)
        if status != "ok":
            failed += 1
            say(f"[{k}/{len(todo)}] {b['id']} -> {status} за {human(sec)} | {rec['err'][:240]}")
        else:
            ok_now += 1
            left = sum(x["n"] for x in todo[k:])
            per = sec / max(n_done, 1)
            say(f"[{k}/{len(todo)}] {b['id']} -> ok за {human(sec)} ({per:.0f} с/кадр) | "
                f"осталось ~{human(left * per)} | свободно {free_gb()} ГБ | в Drive {len(rec['synced'])} файлов")
    say("-" * 72)
    say(f"цикл завершен за сессию: ok {ok_now}, провалов/без синка {failed}. "
        f"Жми ту же ячейку run ещё раз — готовые батчи пропускаются (лог живёт в Drive).")
    say("Если среда сбросилась: 1 → 2 → 3 (она сама распакует слепок _env) → 4 → run; "
        "не считанное не потеряно — курсор в OUTPUT/_state/batch_log.jsonl")
    return 1 if failed else 0


def st_run(cfg):
    plan = load_plan(cfg)
    if not plan:
        say("плана нет или INPUT/OUTPUT/фильтры изменились — сначала выполни стадию plan")
        return 1
    use2023 = resolve_flame2023(cfg)
    plan["use_flame2023"] = use2023
    if str(cfg.get("unit", "folder")) != "photo":
        return run_folders(cfg, plan, use2023)
    photos = [b for b in plan["batches"] if b["kind"] == "photos"]
    others = [b for b in plan["batches"] if b["kind"] != "photos"]
    rc = run_photos(cfg, plan, photos, use2023) if photos else 0
    if others:
        rc = max(rc, run_folders(cfg, plan, use2023, others))
    return rc


def photo_groups(cfg, plan, records):
    """unit=photo: сводка по исходным папкам — сколько фото посчитано, сколько сдохло.
    Источник истины — папки в выходе (done.json), журнал даёт скорости и ошибки."""
    groups = {}
    for b in (plan["batches"] if plan else []):
        if b["kind"] != "photos":
            continue
        names = photo_names_map(b)
        out = b.get("out", b["id"])
        group = groups.setdefault(out, {"total": 0, "done": 0, "fail": 0, "errors": []})
        for i, item in enumerate(b["frames"]):
            name = names[item.get("frame", f"{i:05d}")]
            group["total"] += 1
            if photo_done(photo_dir(cfg, out, name)):
                group["done"] += 1
                continue
            rec = records.get(photo_key(out, name), {})
            if int(rec.get("fails", 0)) >= max(1, int(cfg.get("attempts", 3))):
                group["fail"] += 1
                group["errors"].append(f"{name}: {str(rec.get('err', ''))[:70]}")
    return groups


def status_photos(cfg, plan):
    log = read_log(cfg, "photos_log.jsonl", PHOTO_LOG)
    records = {r["key"]: r for r in log if r.get("key")}
    groups = photo_groups(cfg, plan, records)
    resume = {}
    try:
        resume = json.loads((state_dir(cfg) / "resume.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        resume = {}
    if resume.get("current"):
        say(f"Текущее фото: {resume['current'].get('relpath')}")
    if resume.get("last_completed"):
        say(f"Последнее готовое: {resume['last_completed'].get('relpath')}")
    if resume.get("next_photo"):
        say(f"Следующее фото: {resume['next_photo'].get('relpath')}")
    total = sum(g["total"] for g in groups.values())
    done = sum(g["done"] for g in groups.values())
    say(f"{'папка':<26}{'фото':>7}{'готово':>8}{'с броском':>11}")
    for out, group in sorted(groups.items()):
        say(f"{out:<26}{group['total']:>7}{group['done']:>8}{group['fail']:>11}")
        for line in group["errors"][:3]:
            say(f"    ! {line}")
    say("-" * 72)
    per = median_sec_per_frame(records.values())
    left = total - done
    say(f"фото: посчитано {done} из {total} | осталось {left} | свободно {free_gb()} ГБ | "
        f"журнал: OUTPUT/_state/photos_log.jsonl")
    if left:
        estimate = per or PHOTO_DEFAULT_S
        say(f"по {'медиане журнала' if per else 'умолчанию'} {estimate:.0f} с/фото — осталось "
            f"~{human(estimate * left)} GPU-времени (без учёта перезапусков среды)")
    hard = float(cfg.get("session_hard_limit_hours", 0) or 0)
    snap = read_env_manifest(cfg)
    snap_age = max(0.0, time.time() - float(snap.get("ts", 0) or 0)) if snap.get("parts") else 0.0
    say(f"среда живёт {session_elapsed_hours():.1f} ч" + (f" из {hard:.0f} ч" if hard else "") +
        f" | слепок среды: {f'{human(snap_age)} назад' if snap.get('parts') else 'нет (сделай ячейку 4b)'}")
    if not groups:
        say("плана с фото нет (стадия plan) — показываю только журнал")
        for key, rec in list(records.items())[:40]:
            say(f"  {key:<50}{rec.get('status', '?'):<8}{human(rec.get('sec', 0))}")
    return 0


def st_status(cfg):
    plan = load_plan(cfg)
    records = {r["id"]: r for r in read_log(cfg)}
    say("=" * 72)
    say("ДАШБОРД")
    say("=" * 72)
    if str(cfg.get("unit", "folder")) == "photo":
        return status_photos(cfg, plan)
    batches = plan["batches"] if plan else []
    if not batches:
        say("плана нет (стадия plan) — показываю только лог")
    say(f"{'батч':<26}{'кадр':>6} {'статус':<10}{'mesh':>6}{'frame':>7}{'лмк':>5} {'в Drive':>9}  {'время':>8}")
    totals = {"ok": 0, "fail": 0, "todo": 0, "photos": 0, "done_photos": 0}
    out = Path(cfg["output"])
    for b in batches:
        rec = records.get(b["id"])
        actor = resolve_actor_dir(b["id"]) if actor_root_exists() else None
        meshes = count_files(actor / "mesh", ".ply") if actor else 0
        frames = count_files(actor / "checkpoint", ".frame") if actor else 0
        lmks = count_files(actor / "mesh", ".npy") if actor else 0
        folder = batch_dir(cfg, b["id"])
        drive = len(list(folder.iterdir())) if folder.is_dir() else 0
        if rec is None:
            state, sec = "не сделан", ""
            totals["todo"] += 1
        else:
            base = rec.get("status", "?")
            state = f"демо {rec.get('n')}/{b['n']}" if base == "ok" and rec.get("partial") else base
            sec = human(rec.get("sec", 0))
            totals["ok" if base == "ok" else "fail"] += 1
            totals["done_photos"] += rec.get("n", 0) if base == "ok" else 0
        totals["photos"] += b["n"]
        say(f"{b['id']:<26}{b['n']:>6} {state:<10}{meshes:>6}{frames:>7}{lmks:>5} {drive:>9}  {sec:>8}")
        if rec and rec.get("err"):
            say(f"    ! {rec['err'][:200]}")
    say("-" * 72)
    say(f"батчей: {len(batches)} | ok {totals['ok']} | провал {totals['fail']} | "
        f"не сделано {totals['todo']} | кадров {totals['done_photos']}/{totals['photos']} | "
        f"свободно {free_gb()} ГБ | Drive: {out.is_dir()}")
    per = median_sec_per_frame(records.values())
    left_frames = max(0, totals["photos"] - totals["done_photos"])
    if per and left_frames:
        say(f"по медиане {per:.0f} с/кадр осталось ~{human(per * left_frames)} GPU-времени "
            f"(без учёта перезапусков среды)")
    hard = float(cfg.get("session_hard_limit_hours", 0) or 0)
    snap = read_env_manifest(cfg)
    snap_age = max(0.0, time.time() - float(snap.get("ts", 0) or 0)) if snap.get("parts") else 0.0
    say(f"среда живёт {session_elapsed_hours():.1f} ч" + (f" из {hard:.0f} ч" if hard else "") +
        f" | слепок среды: {f'{human(snap_age)} назад' if snap.get('parts') else 'нет (сделай ячейку 4b)'}")
    if not plan:
        for bid, rec in records.items():
            say(f"  {bid:<26}{rec.get('n', 0):>6} {rec.get('status', '?'):<10} {human(rec.get('sec', 0))}")
    return 0


def actor_root_exists():
    return TRACK.is_dir()


def folder_pose(folder):
    """Ракурс из params.json готовой папки фото — на случай журнала, написанного до появления позы."""
    params = Path(folder) / "params.json"
    if not params.is_file():
        return {}
    try:
        return json.loads(params.read_text(encoding="utf-8")).get("pose") or {}
    except (OSError, ValueError):
        return {}


def label_bin(text):
    """Бин из разметки корпуса: имя файла/папки вида `right_light__2016_05_20.jpg`.
    Метка человека остаётся главной — сверяем её с измеренной позой, чтобы видеть расход, а не перезаписывать."""
    low = str(text or "").lower()
    for name, _, _, _ in VIEW_BINS:
        if re.search(rf"(?<![a-z]){name}(?![a-z])", low):
            return name
    return ""


def collect_photos(cfg, out):
    """unit=photo: индекс строится по журналу фото, папки сверяются одним списком каталога."""
    records = {r["key"]: r for r in read_log(cfg, "photos_log.jsonl", PHOTO_LOG) if r.get("key")}
    present = set()
    root = batches_root(cfg)
    topped_up = 0
    if root.is_dir():
        for batch in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")):
            photos = batch / "photos"
            if not photos.is_dir():
                continue
            for name in sorted(p.name for p in photos.iterdir() if p.is_dir()):
                present.add(photo_key(batch.name, name))
                topped_up += 1 if ensure_visibility(photos / name) else 0
    cols = ["key", "batch", "photo", "frame", "relpath", "status", "fails", "sec",
            "yaw_deg", "pitch_deg", "roll_deg", "view_bin", "label_bin", "bin_matches_label",
            "visible_side", "uv_fraction", "yaw_proxy",
            "jaw_norm", "exp_norm", "shape_norm", "t", "fl", "pp", "folder", "ts"]
    pose_keys = ("yaw_deg", "pitch_deg", "roll_deg", "visible_side", "yaw_proxy", "uv_fraction")
    rows = []
    for key, rec in sorted(records.items()):
        row = {"key": key, "batch": rec.get("out", ""), "photo": rec.get("photo", ""),
               "frame": rec.get("frame", ""), "relpath": rec.get("relpath", ""),
               "status": rec.get("status", ""), "fails": rec.get("fails", 0),
               "sec": rec.get("sec", ""), "folder": "есть" if key in present else "НЕТ",
               "ts": rec.get("ts", "")}
        pose = rec
        if not rec.get("view_bin") and rec.get("status") == "ok":
            pose = folder_pose(photos_root(cfg, rec.get("out", "")) / rec.get("photo", "")) or rec
        row.update({k: pose.get(k, rec.get(k, "")) for k in pose_keys})
        row["view_bin"] = pose.get("view_bin") or pose.get("bin") or ""
        row["label_bin"] = label_bin(rec.get("relpath") or rec.get("photo"))
        row["bin_matches_label"] = ("" if not row["label_bin"] or not row["view_bin"] else
                                   "да" if row["label_bin"] == row["view_bin"] else "нет")
        row.update({k: v for k, v in (rec.get("summary") or {}).items() if k in cols})
        rows.append(row)
    (out / "_index").mkdir(parents=True, exist_ok=True)
    with (out / "_index/photos.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    gone = sorted(present - set(records))
    ok = sum(1 for r in records.values() if r.get("status") == "ok")
    bins = {name: sum(1 for r in rows if r["view_bin"] == name) for name, *_ in VIEW_BINS}
    bins["без метки"] = sum(1 for r in rows if not r["view_bin"])
    check = Counter(r["bin_matches_label"] for r in rows)
    confusion = Counter(f"{r['label_bin']} -> {r['view_bin']}" for r in rows if r["bin_matches_label"] == "нет")
    report = {"config": {k: cfg.get(k) for k in ("input", "output", "unit", "photos_per_chunk",
                                                 "single_iters", "use_flame2023", "subjects")},
              "totals": {"photos_in_log": len(records), "ok": ok,
                         "fail": len(records) - ok, "folders_in_drive": len(present),
                         "without_log_entry": len(gone), "free_gb": free_gb()},
              "view_bins": bins,
              "bin_vs_label": {"совпало": check["да"], "разошлось": check["нет"],
                              "не с чем сверять": check[""],
                              "топ расхождений": dict(confusion.most_common(8))},
              "per_batch": {batch: sum(1 for r in records.values() if r.get("out") == batch and
                                       r.get("status") == "ok")
                            for batch in sorted({r.get("out") for r in records.values() if r.get("out")})}}
    (out / "_index/report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                            encoding="utf-8")
    say(f"фото в журнале: {len(records)} (ok {ok}) | папок в выходе: {len(present)} | "
        f"папок без записи в журнале: {len(gone)}")
    for key in gone[:10]:
        say(f"  {key}: папка есть, в photos_log.jsonl её нет — так и оставим, run её не тронет")
    say(f"индекс: {out}/_index/photos.csv ({len(rows)} строк) и report.json")
    say("ракурсы по матрице поворота (9 бинов корпуса): " +
        ", ".join(f"{name} {count}" for name, count in bins.items() if count))
    say(f"сверка с твоей разметкой в именах файлов: совпало {check['да']}, разошлось {check['нет']}, "
        f"без метки ракурса в имени {check['']}")
    for pair, count in confusion.most_common(5):
        say(f"  расхождение {pair}: {count}")
    say("в папке каждого фото: source, crop.jpg, segmentation.png, landmarks_pipnet.npy, normals.png, "
        "uv_map.png, arcface.npy, mica_* (identity/mesh/kpt7/kpt68), mesh.ply, canonical.ply, "
        "landmarks_7/68/eyes.npy, checkpoint.frame, mask_fg/uv/normal.png, visibility_points.npy, "
        "params.json (в нём pose: yaw/pitch/roll + bin), overlay.jpg, run.log, done.json")
    if topped_up:
        say(f"дописал наблюдаемости в {topped_up} папк(ах) — это был CPU без пересчёта трекинга")
    return 1 if (len(records) - ok) else 0


def st_collect(cfg):
    out = Path(cfg["output"])
    say("=" * 72)
    say(f"СБОРКА ИНДЕКСА в {out}")
    say("=" * 72)
    if str(cfg.get("unit", "folder")) == "photo":
        return collect_photos(cfg, out)
    records = {r["id"]: r for r in read_log(cfg)}
    root = batches_root(cfg)
    batch_dirs = sorted(p for p in root.glob("*") if p.is_dir() and not p.name.startswith("_")) \
        if root.is_dir() else []
    index_rows = []
    report = {"config": {k: cfg[k] for k in ("input", "output", "subjects", "use_flame2023",
                                             "face_filter", "torch_compile")},
              "batches": {}, "missing": []}
    for folder in batch_dirs:
        bid = folder.name
        files = {p.name: p.stat().st_size for p in folder.iterdir() if p.is_file()}
        frames_file = folder / "frames.csv"
        rows = 0
        if frames_file.is_file():
            with frames_file.open(encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    row["batch"] = bid
                    index_rows.append(row)
                    rows += 1
        archive_ok = (not cfg.get("keep_archive", True) or
                      ((folder / "artifacts.tar.gz").is_file() and
                       tarfile.is_tarfile(folder / "artifacts.tar.gz")))
        missing_files = bundle_missing(folder, records.get(bid, {}).get("synced", []), cfg)
        report["batches"][bid] = {"files": files, "frames": rows, "log_status": records.get(bid, {}).get("status"),
                                  "archive": archive_ok, "missing_files": missing_files}
        if not archive_ok or rows == 0 or missing_files:
            report["missing"].append(bid)
        say(f"{bid:<26} кадров {rows:>5} | archive {'ok' if archive_ok else 'НЕТ':<4} | "
            f"лог: {records.get(bid, {}).get('status', 'нет записи')}")
    plan = load_plan(cfg)
    expected = set(records) | {b["id"] for b in plan["batches"]} if plan else set(records)
    for bid in sorted(expected):
        if bid not in report["batches"]:
            report["missing"].append(bid)
            say(f"{bid:<26} В ДРАЙВЕ НЕТ — перезапусти run для этого батча")
    (out / "_index").mkdir(parents=True, exist_ok=True)
    with (out / "_index/frames_all.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FRAMES_ALL_CSV_COLS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(index_rows, key=lambda r: (r["batch"], r.get("frame", ""))))
    report["frames_indexed"] = len(index_rows)
    report["totals"] = {"batches_in_drive": len(batch_dirs), "frames": len(index_rows),
                        "log_ok": sum(1 for r in records.values() if r.get("status") == "ok"),
                        "log_fail": sum(1 for r in records.values() if r.get("status") not in ("ok", "ok_no_sync")),
                        "free_gb": free_gb()}
    (out / "_index").mkdir(parents=True, exist_ok=True)
    (out / "_index/report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    say("-" * 72)
    say(f"в Drive батчей: {len(batch_dirs)} | строк в _index/frames_all.csv: {len(index_rows)} | "
        f"проблемных батчей: {len(report['missing'])}")
    say(f"что лежит по каждому батчу: frames.csv (кадр -> исходное фото + параметры + ракурс), manifest.csv, "
        f"params.npz (exp/shape/jaw/neck/eyes/eyelids/R/t/camera/joint_transforms), metrics.json (в т.ч. "
        f"эффективный конфиг трекинга), result.mp4"
        + (", artifacts.tar.gz (mesh/ checkpoint/ logs/ preproc/)" if cfg.get("keep_archive", True) else "")
        + f", summary.md и photos/<имя фото>/ (mesh.ply, params.json, ландмарки, кроп, overlay)")
    return 1 if report["missing"] else 0


def st_freeze(cfg):
    """Publish a complete cache generation last; failed saves never replace the old manifest."""
    previous = read_env_manifest(cfg)
    if snapshot_ready(cfg, previous):
        say("Проверенный кеш среды уже сохранен: повторная упаковка не нужна")
        return 0
    if not ENV_MARKER.is_file() or not (ENV_ROOT / "bin/python").is_file() or not CODE.is_dir():
        say("Нет проверенного окружения: сначала выполни ячейки 3 и 4")
        return 1
    if any(not ok for _, _, ok, _ in check_weights()):
        say("Веса не прошли проверку: неполный кеш не публикуется")
        return 1
    dest = env_dir(cfg, make=True)
    generation = str(time.time_ns())
    parts, created = [], []
    roots = tuple(p.absolute() for _, sources, _ in freeze_parts() for p in sources)

    def include(info):
        if cache_excluded(info.name):
            return None
        return info

    try:
        for index, (base, sources, note) in enumerate(freeze_parts()):
            present = [Path(s) for s in sources if Path(s).is_dir()]
            if not present:
                raise ValueError(f"Нет данных для обязательной части {base}")
            raw = sum(dir_size_gb(p) for p in present)
            if free_gb_at(dest) < raw + 1.0:
                raise OSError(f"Мало места для {base}: нужно до {raw + 1:.1f} ГБ свободного места")
            # Dereference only links into the fixed cache roots, never user data or credentials.
            for source in present:
                for parent, dirs, files in os.walk(source):
                    dirs[:] = [d for d in dirs if not cache_excluded(str(Path(parent) / d))]
                    for filename in dirs + files:
                        path = Path(parent) / filename
                        if not cache_excluded(str(path)) and path.is_symlink():
                            resolved = path.resolve()
                            if not any(resolved.is_relative_to(root) for root in roots):
                                raise ValueError(f"Ссылка за пределы кеша: {path} -> {resolved}")
            name = base.removesuffix(".tar.gz") + "-" + generation + ".tar.gz"
            archive, partial = dest / name, dest / (name + ".part")
            created.extend((partial, archive))
            progress(phase="Сохранение кеша: " + base, current=f"{raw:.2f} ГБ до сжатия",
                     completed=index, total=len(freeze_parts()))
            digest = hashlib.sha256()
            with partial.open("wb") as stream:
                class HashWriter:
                    def write(self, data):
                        digest.update(data)
                        return stream.write(data)

                    def __getattr__(self, key):
                        return getattr(stream, key)

                with tarfile.open(fileobj=HashWriter(), mode="w:gz", compresslevel=1, dereference=True) as tar:
                    for source in present:
                        tar.add(source, arcname=str(source.absolute()).lstrip("/"), filter=include)
            partial.replace(archive)
            parts.append({"name": name, "bytes": archive.stat().st_size, "sha256": digest.hexdigest(),
                          "raw_gb": round(raw, 2), "sources": [str(p) for p in present], "note": note})
            say(f"Сохранен {base}: {archive.stat().st_size / 1e9:.2f} ГБ")
        manifest = {"ts": int(time.time()), "output": cfg["output"], "parts": parts,
                    "pixel_rev": PIXEL_REV, "signature": snapshot_signature()}
        write_json(manifest, env_manifest_path(cfg))
    except (OSError, tarfile.TarError, ValueError, RecursionError) as error:
        for path in created:
            path.unlink(missing_ok=True)
        say(f"Кеш не сохранен: {type(error).__name__}: {error}. Предыдущий слепок оставлен без изменений.")
        return 1
    # Only archives named in our previous manifest are retired after successful publication.
    try:
        for _, archive, _, _ in snapshot_parts(cfg, previous):
            if archive.name not in {p["name"] for p in parts}:
                archive.unlink(missing_ok=True)
    except (OSError, ValueError, TypeError):
        pass
    progress(completed=len(parts), total=len(parts), current="")
    say(f"Кеш следующей сессии: {sum(p['bytes'] for p in parts) / 1e9:.2f} ГБ в {dest}")
    return 0


def st_warm(cfg):
    """Restore fixed cache roots; a separate CUDA verification decides whether setup is usable."""
    manifest = read_env_manifest(cfg)
    if not manifest.get("parts"):
        say("Кеша среды пока нет: первая установка будет выполнена полностью")
        return 1
    if manifest.get("output") != cfg["output"] or manifest.get("pixel_rev") != PIXEL_REV:
        say("Кеш другого выхода или другой ревизии кода: восстановление пропущено")
        return 1
    try:
        parts = snapshot_parts(cfg, manifest)
    except (ValueError, TypeError) as error:
        say(f"Некорректный кеш: {error}")
        return 1
    t0 = time.time()
    restored, bad = [], []
    for index, (part, archive, sources, kind) in enumerate(parts):
        if kind == "p3dmm-env.tar.gz" and (ENV_ROOT / "bin/python").is_file() and ENV_MARKER.is_file():
            continue
        local = WORK / "env_restore" / archive.name
        progress(phase="Восстановление кеша: " + kind, current=archive.name, completed=index, total=len(parts))
        try:
            if not archive.is_file() or archive.stat().st_size != part.get("bytes"):
                raise ValueError("архив отсутствует или имеет неправильный размер")
            if free_gb() < archive.stat().st_size / 1e9 + float(part.get("raw_gb", 0)) + 1:
                raise OSError("мало места на /content для восстановления")
            local.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            with archive.open("rb") as incoming, local.open("wb") as outgoing:
                for block in iter(lambda: incoming.read(8 * 1024 * 1024), b""):
                    digest.update(block)
                    outgoing.write(block)
            if part.get("sha256") and digest.hexdigest() != part["sha256"]:
                raise ValueError("SHA256 архива не совпадает: кеш поврежден")
            safe_extract(local, sources, EXTRACT_ROOT, missing_only=kind == "root-cache.tar.gz")
        except (OSError, tarfile.TarError, ValueError) as error:
            bad.append(f"{part['name']}: {type(error).__name__}: {error}")
            say("FAIL: " + bad[-1])
        else:
            restored.append(kind)
        finally:
            local.unlink(missing_ok=True)
    if bad:
        write_json(dict(manifest, invalid=bad), env_manifest_path(cfg))
    progress(completed=len(parts), total=len(parts), current="")
    say(f"Восстановлено за {human(time.time() - t0)}: {', '.join(restored) or 'ничего'}")
    say("Далее проверяются версии, зависимости и CUDA; наличие Python само по себе не означает успех.")
    return 0 if not bad and (ENV_ROOT / "bin/python").is_file() and CODE.is_dir() else 1


STAGES = {"preflight": st_preflight, "plan": st_plan,
          "run": st_run, "status": st_status, "collect": st_collect,
          "freeze": st_freeze, "warm": st_warm}


def main():
    parser = argparse.ArgumentParser(description="Pixel3DMM colab pipeline")
    parser.add_argument("stage", choices=sorted(STAGES))
    args = parser.parse_args()
    cfg = load_config()
    if args.stage in {"preflight", "run", "freeze"}:
        apply_env(cfg)
    else:
        for folder in (WORK, LOGS):
            folder.mkdir(parents=True, exist_ok=True)
    try:
        rc = STAGES[args.stage](cfg)
    except KeyboardInterrupt:
        say("остановлено вручную")
        rc = 130
    except Exception as error:
        import traceback
        traceback.print_exc()
        say(f"ПАДЕНИЕ на стадии {args.stage}: {type(error).__name__}: {error}")
        rc = 1
    sys.exit(rc)


if __name__ == "__main__":
    main()
