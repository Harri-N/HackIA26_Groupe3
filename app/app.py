"""
AI Workshop — Flask app with live inference streaming via SSE.
"""

import os, time, datetime, json, base64, threading, uuid, collections
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms.v2 as transforms
import yaml
from PIL import Image
import face_recognition
from flask import Flask, render_template, Response, jsonify, request, send_from_directory

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUTS_DIR = "./outputs"
AUTHORIZED_FACES_DIR = "./authorized_faces"
MODELS_DIR = "./models"
CONFIG_PATH = "./config.yaml"
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(AUTHORIZED_FACES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

with open(CONFIG_PATH, "r", encoding="utf-8") as _f:
    CONFIG = yaml.safe_load(_f)

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL PATHS — driven by config.yaml
# ═══════════════════════════════════════════════════════════════════════════════
MODELS = {
    "fire_localizer":   os.path.join(MODELS_DIR, CONFIG["firezone"]["weights"]),
    "object_detector":  os.path.join(MODELS_DIR, CONFIG["object_detection"]["weights"]),
}

# Fire classifiers — trained on FIRE_DATABASE_4 with the same 3-class layout.
FIRE_CLASSES   = CONFIG["fire"]["classes"]
FIRE_IMGSZ     = CONFIG["fire"]["image_size"]
IMAGENET_MEAN  = CONFIG["fire"]["normalization"]["mean"]
IMAGENET_STD   = CONFIG["fire"]["normalization"]["std"]
# YOLO firezone names — explicit so we never display raw class indices.
FIREZONE_NAMES = {int(k): v for k, v in CONFIG["firezone"]["class_names"].items()}
FIREZONE_CONF  = float(CONFIG["firezone"]["confidence"])

app = Flask(__name__)

ts = lambda: datetime.datetime.now().strftime("%H:%M:%S")


def frame_to_b64jpg(frame, max_w=960, quality=65):
    """Encode a BGR frame to base64 JPEG, resized to max_w.

    Smaller payloads + lower quality reduce decode time in the browser,
    which is the main source of jank on the Jetson.
    """
    h, w = frame.shape[:2]
    if w > max_w:
        scale = max_w / w
        frame = cv2.resize(frame, (max_w, int(h * scale)))
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode()


def sse_event(data):
    """Format a dict as an SSE data line."""
    return f"data: {json.dumps(data)}\n\n"


# ═══════════════════════════════════════════════════════════════════════════════
# STREAM CONTROL — pause / resume / stop signalling shared by all SSE generators
# Each running generator owns a `sid` (stream id) sent by the client; the client
# can flip the state via POST /stream/control. The generators check the state
# on every iteration. On stop, the matching module's models are released so the
# GPU memory is freed.
# ═══════════════════════════════════════════════════════════════════════════════
_streams_lock = threading.Lock()
_streams = {}  # sid -> {"paused": bool, "stopped": bool, "module": str}


def _ensure_stream(sid, module):
    with _streams_lock:
        _streams[sid] = {"paused": False, "stopped": False, "module": module}


def _stream_state(sid):
    with _streams_lock:
        return dict(_streams.get(sid, {}))


def _set_stream(sid, **kw):
    with _streams_lock:
        if sid in _streams:
            _streams[sid].update(kw)


def _drop_stream(sid):
    with _streams_lock:
        return _streams.pop(sid, None)


def _resolve_source(source):
    """Map a source string to what cv2.VideoCapture expects. `webcam` → 0;
    anything else is a file path."""
    return 0 if str(source).strip().lower() == "webcam" else source


def release_fire_models():
    global _cls_models, _yolo_fire
    _cls_models = {}
    _yolo_fire = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def release_obj_models():
    global _yolo_obj
    _yolo_obj = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def release_fall_models():
    global _fall_model, _fall_processor
    _fall_model = None
    _fall_processor = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


_RELEASE_FNS = {
    "fire":    release_fire_models,
    "objects": release_obj_models,
    "fall":    release_fall_models,
}


# ═══════════════════════════════════════════════════════════════════════════════
# FACE RECOGNITION — login via webcam, gated by ./authorized_faces/*.jpg
# ═══════════════════════════════════════════════════════════════════════════════
_auth_encodings = []
_auth_names = []


def load_authorized_faces():
    global _auth_encodings, _auth_names
    _auth_encodings.clear()
    _auth_names.clear()
    for fname in os.listdir(AUTHORIZED_FACES_DIR):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        img = face_recognition.load_image_file(os.path.join(AUTHORIZED_FACES_DIR, fname))
        encs = face_recognition.face_encodings(img)
        if encs:
            _auth_encodings.append(encs[0])
            _auth_names.append(os.path.splitext(fname)[0].replace("_", " ").title())
    print(f"Loaded {len(_auth_encodings)} authorized face(s): {_auth_names}")


_camera = None
_camera_lock = threading.Lock()
_logged_in_user = None


def get_camera():
    global _camera
    with _camera_lock:
        if _camera is None or not _camera.isOpened():
            _camera = cv2.VideoCapture(0)
    return _camera


def release_camera():
    global _camera
    with _camera_lock:
        if _camera and _camera.isOpened():
            _camera.release()
        _camera = None


LOGIN_TIMEOUT = 30  # seconds


def gen_login_frames():
    global _logged_in_user
    cam = get_camera()
    frame_count = 0
    t_start = time.time()
    while True:
        if _logged_in_user or (time.time() - t_start > LOGIN_TIMEOUT):
            break
        ret, frame = cam.read()
        if not ret:
            break
        display = frame.copy()
        if frame_count % 3 == 0:
            small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            locs = face_recognition.face_locations(rgb_small)
            encs = face_recognition.face_encodings(rgb_small, locs)
            for (t, r, b, l), enc in zip(locs, encs):
                t, r, b, l = t*2, r*2, b*2, l*2
                if _auth_encodings:
                    matches = face_recognition.compare_faces(_auth_encodings, enc, tolerance=0.5)
                    dists = face_recognition.face_distance(_auth_encodings, enc)
                    if any(matches):
                        idx = int(np.argmin(dists))
                        name = _auth_names[idx]
                        cv2.rectangle(display, (l, t), (r, b), (0, 200, 0), 3)
                        cv2.putText(display, f"{name} ({1-dists[idx]:.0%})", (l, t-12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
                        _logged_in_user = name
                    else:
                        cv2.rectangle(display, (l, t), (r, b), (0, 0, 200), 3)
                        cv2.putText(display, "Unknown", (l, t-12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), 2)
        frame_count += 1
        _, jpeg = cv2.imencode(".jpg", display)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
    release_camera()


# ═══════════════════════════════════════════════════════════════════════════════
# FIRE DETECTION — live SSE stream
# Pipeline: classifier first (selectable: Swin-B, ResNet-50, EffNet V2-S/M); if
# not no_fire → run firezone YOLO for bounding boxes AND Grad-CAM on the
# classifier for explainability.
# ═══════════════════════════════════════════════════════════════════════════════
_cls_models = {}  # classifier_name -> loaded nn.Module (cached, gradcam-hooked)
_yolo_fire = None

_fire_transform = transforms.Compose([
    transforms.Resize(int(FIRE_IMGSZ * 1.15)),
    transforms.CenterCrop(FIRE_IMGSZ),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def _build_swin_b(num_classes: int) -> nn.Module:
    model = torchvision.models.swin_b(weights=None)
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model


def _build_resnet50(num_classes: int) -> nn.Module:
    model = torchvision.models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _build_effnet_v2_s(num_classes: int) -> nn.Module:
    model = torchvision.models.efficientnet_v2_s(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def _build_effnet_v2_m(num_classes: int) -> nn.Module:
    model = torchvision.models.efficientnet_v2_m(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


# Each entry describes one classifier: builder, Grad-CAM target layer, and
# whether activations are channels-last (Swin) or channels-first (CNNs). The
# weights filename and the active choice come from config.yaml.
_FIRE_WEIGHTS = CONFIG["fire"]["weights"]
FIRE_CLASSIFIERS = {
    "swin_b": {
        "filename":      _FIRE_WEIGHTS["swin_b"],
        "builder":       _build_swin_b,
        "target":        lambda m: m.norm,
        "channels_last": True,
        "label":         "Swin-B",
    },
    "resnet50": {
        "filename":      _FIRE_WEIGHTS["resnet50"],
        "builder":       _build_resnet50,
        "target":        lambda m: m.layer4,
        "channels_last": False,
        "label":         "ResNet-50",
    },
    "efficientnet_v2_s": {
        "filename":      _FIRE_WEIGHTS["efficientnet_v2_s"],
        "builder":       _build_effnet_v2_s,
        "target":        lambda m: m.features[-1],
        "channels_last": False,
        "label":         "EfficientNet V2-S",
    },
    "efficientnet_v2_m": {
        "filename":      _FIRE_WEIGHTS["efficientnet_v2_m"],
        "builder":       _build_effnet_v2_m,
        "target":        lambda m: m.features[-1],
        "channels_last": False,
        "label":         "EfficientNet V2-M",
    },
}
ACTIVE_FIRE_CLASSIFIER = CONFIG["fire"]["classifier"]
if ACTIVE_FIRE_CLASSIFIER not in FIRE_CLASSIFIERS:
    raise ValueError(f"config.yaml: fire.classifier='{ACTIVE_FIRE_CLASSIFIER}' is not one of {list(FIRE_CLASSIFIERS)}")


def _register_gradcam_hooks(model, target_layer, channels_last):
    """Hook target_layer to capture activations and their gradients during a
    grad-enabled forward+backward done inside `_compute_gradcam_overlay`.
    State is attached per-model so multiple cached classifiers stay independent.
    Forward passes done under torch.no_grad() are skipped."""
    model._gradcam_state = {"activations": None, "gradients": None,
                            "channels_last": channels_last}

    def fwd_hook(_m, _inp, out):
        if not out.requires_grad:
            return
        out.retain_grad()
        model._gradcam_state["activations"] = out

        def _grad_hook(grad):
            model._gradcam_state["gradients"] = grad.detach()
        out.register_hook(_grad_hook)

    target_layer.register_forward_hook(fwd_hook)


def _compute_gradcam_overlay(model, input_tensor, target_class_idx, orig_bgr):
    """Grad-CAM overlay. `input_tensor` is the preprocessed (1,3,H,W) tensor
    that was just classified; `orig_bgr` is the raw camera frame."""
    state = model._gradcam_state
    model.zero_grad(set_to_none=True)
    state["activations"] = None
    state["gradients"]   = None
    x = input_tensor.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        logits = model(x)
        logits[0, target_class_idx].backward()

    acts  = state["activations"]
    grads = state["gradients"]
    if acts is None or grads is None:
        return orig_bgr.copy()

    if state["channels_last"]:
        # Swin: (1, H, W, C). Reduce over spatial dims for channel weights.
        weights = grads.mean(dim=(1, 2), keepdim=True)
        cam = (weights * acts).sum(dim=-1)
    else:
        # CNNs: (1, C, H, W).
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1)
    cam = torch.relu(cam)[0]
    cam_min, cam_max = cam.min(), cam.max()
    if cam_max - cam_min > 1e-8:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = torch.zeros_like(cam)
    cam_np = cam.detach().cpu().numpy()

    h, w = orig_bgr.shape[:2]
    cam_np = cv2.resize(cam_np, (w, h))
    heatmap = cv2.applyColorMap((cam_np * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(orig_bgr, 0.55, heatmap, 0.45, 0)


def _check_model_file(path, name):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {name} at {path}. Place the file in ./models/ or update MODELS dict in app.py."
        )


def load_fire_models(classifier_name=None):
    """Load the configured fire classifier (plain state_dict .pt file) and the
    firezone YOLO model. Each classifier is cached on first use."""
    global _yolo_fire
    if classifier_name is None or classifier_name not in FIRE_CLASSIFIERS:
        classifier_name = ACTIVE_FIRE_CLASSIFIER
    spec = FIRE_CLASSIFIERS[classifier_name]

    if classifier_name not in _cls_models:
        path = os.path.join(MODELS_DIR, spec["filename"])
        _check_model_file(path, f"fire classifier ({spec['label']})")
        m = spec["builder"](num_classes=len(FIRE_CLASSES))
        sd = torch.load(path, map_location=DEVICE, weights_only=True)
        m.load_state_dict(sd)
        m.to(DEVICE).eval()
        _register_gradcam_hooks(m, spec["target"](m), spec["channels_last"])
        _cls_models[classifier_name] = m

    if _yolo_fire is None:
        _check_model_file(MODELS["fire_localizer"], "fire localizer")
        from ultralytics import YOLO
        _yolo_fire = YOLO(MODELS["fire_localizer"])

    return _cls_models[classifier_name], _yolo_fire


def _draw_firezone_boxes(frame, results):
    """Draw firezone boxes with readable class names (never raw 0/1 indices)."""
    annotated = frame.copy()
    detections = []
    for box in results[0].boxes:
        cls_idx = int(box.cls[0])
        name = FIREZONE_NAMES.get(cls_idx, f"class_{cls_idx}")
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        color = (0, 0, 255) if name == "fire" else (180, 180, 180)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{name} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        detections.append((name, conf))
    return annotated, detections


def gen_fire_sse(video_path, sid=None):
    if sid:
        _ensure_stream(sid, "fire")
    cap = None
    try:
        spec = FIRE_CLASSIFIERS[ACTIVE_FIRE_CLASSIFIER]
        yield sse_event({"type": "log", "text": f"[{ts()}] Chargement des modèles de feu (classifier : {spec['label']})..."})
        cls_model, yolo_model = load_fire_models(ACTIVE_FIRE_CLASSIFIER)

        source = _resolve_source(video_path)
        is_webcam = isinstance(source, int)
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            yield sse_event({"type": "log", "text": f"[{ts()}] ERREUR : impossible d'ouvrir la source : {video_path}"})
            yield sse_event({"type": "done"})
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w, h = int(cap.get(3)), int(cap.get(4))
        if is_webcam:
            yield sse_event({"type": "log", "text": f"[{ts()}] Webcam : {w}x{h} @ {fps:.0f}ips (temps réel)"})
        else:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            yield sse_event({"type": "log", "text": f"[{ts()}] Vidéo : {w}x{h} @ {fps:.0f}ips, {total} images"})

        idx = 0
        cls_label, cls_conf = "no_fire", 0.0
        t_start = time.time()

        while True:
            st = _stream_state(sid) if sid else {}
            if st.get("stopped"):
                yield sse_event({"type": "log", "text": f"[{ts()}] Arrêté par l'utilisateur"})
                break
            if st.get("paused"):
                time.sleep(0.1)
                continue

            ret, frame = cap.read()
            if not ret:
                break

            # 1) Classification with the Swin-B fire model.
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            t = _fire_transform(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                probs = torch.softmax(cls_model(t)[0], dim=0)
            cls_idx = int(probs.argmax().item())
            cls_label = FIRE_CLASSES[cls_idx]
            cls_conf = float(probs.max().item())
            if cls_label != "no_fire" and cls_label != "nofire":
                yield sse_event({"type": "log", "text": f"[{ts()}] Image {idx} ({idx/fps:.1f}s) : {cls_label} ({cls_conf:.0%})"})

            # 2) Only when something is detected: run firezone YOLO + Grad-CAM.
            xai_overlay = None
            if cls_label != "no_fire" and cls_label != "nofire":
                results = yolo_model(frame, conf=FIREZONE_CONF, verbose=False)
                annotated, dets = _draw_firezone_boxes(frame, results)
                for name, conf in dets:
                    yield sse_event({"type": "log", "text": f"[{ts()}] Image {idx} ({idx/fps:.1f}s) : FireZone {name} ({conf:.0%})"})
                xai_overlay = _compute_gradcam_overlay(cls_model, t, cls_idx, frame)
            else:
                annotated = frame.copy()

            color = (0, 0, 255) if cls_label == "fire" else (0, 165, 255) if cls_label == "start_fire" else (0, 255, 0)
            cv2.putText(annotated, f"{cls_label} ({cls_conf:.0%})", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            if xai_overlay is not None:
                cv2.putText(xai_overlay, f"Grad-CAM: {cls_label} ({cls_conf:.0%})", (10, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

            if idx % 2 == 0:
                if not is_webcam:
                    target_time = t_start + idx / fps
                    wait = target_time - time.time()
                    if wait > 0:
                        time.sleep(wait)
                yield sse_event({"type": "frame", "data": frame_to_b64jpg(annotated)})
                if xai_overlay is not None:
                    yield sse_event({"type": "frame_xai", "data": frame_to_b64jpg(xai_overlay)})
                else:
                    yield sse_event({"type": "xai_clear"})

            idx += 1

        yield sse_event({"type": "log", "text": f"[{ts()}] Terminé — {idx} images traitées"})
        if sid and _stream_state(sid).get("stopped"):
            release_fire_models()
            yield sse_event({"type": "log", "text": f"[{ts()}] Modèles déchargés (mémoire GPU libérée)"})
    except Exception as e:
        yield sse_event({"type": "log", "text": f"[{ts()}] ERREUR : {e}"})
    finally:
        if cap is not None:
            cap.release()
        if sid:
            _drop_stream(sid)
    yield sse_event({"type": "done"})


# ═══════════════════════════════════════════════════════════════════════════════
# OBJECT DETECTION
# ═══════════════════════════════════════════════════════════════════════════════
FR_NAMES = dict(CONFIG["object_detection"]["french_names"])
OBJ_CONF = float(CONFIG["object_detection"]["confidence"])
_yolo_obj = None


def load_obj_model():
    global _yolo_obj
    if _yolo_obj is None:
        _check_model_file(MODELS["object_detector"], "object detector")
        from ultralytics import YOLO
        _yolo_obj = YOLO(MODELS["object_detector"])
    return _yolo_obj


def _friendly_name(cls_name):
    return FR_NAMES.get(cls_name, cls_name)


def gen_objects_sse(video_path, sid=None):
    if sid:
        _ensure_stream(sid, "objects")
    cap = None
    try:
        yield sse_event({"type": "log", "text": f"[{ts()}] Chargement du modèle de détection d'objets..."})
        model = load_obj_model()

        class_list = [_friendly_name(n) for n in model.names.values()]
        yield sse_event({"type": "log", "text": f"[{ts()}] Classes détectées : {', '.join(class_list)}"})

        source = _resolve_source(video_path)
        is_webcam = isinstance(source, int)
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            yield sse_event({"type": "log", "text": f"[{ts()}] ERREUR : impossible d'ouvrir la source : {video_path}"})
            yield sse_event({"type": "done"})
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w, h = int(cap.get(3)), int(cap.get(4))
        if is_webcam:
            yield sse_event({"type": "log", "text": f"[{ts()}] Webcam : {w}x{h} @ {fps:.0f}ips (temps réel)"})
        else:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            yield sse_event({"type": "log", "text": f"[{ts()}] Vidéo : {w}x{h} @ {fps:.0f}ips, {total} images"})

        idx = 0
        t_start = time.time()

        while True:
            st = _stream_state(sid) if sid else {}
            if st.get("stopped"):
                yield sse_event({"type": "log", "text": f"[{ts()}] Arrêté par l'utilisateur"})
                break
            if st.get("paused"):
                time.sleep(0.1)
                continue

            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=OBJ_CONF, verbose=False)
            annotated = frame.copy()

            for box in results[0].boxes:
                cls_name = model.names[int(box.cls[0])]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                friendly = _friendly_name(cls_name)
                yield sse_event({"type": "log", "text": f"[{ts()}] Image {idx} ({idx/fps:.1f}s) : {friendly} ({conf:.0%})"})
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, f"{friendly} {conf:.0%}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if idx % 2 == 0:
                if not is_webcam:
                    target_time = t_start + idx / fps
                    wait = target_time - time.time()
                    if wait > 0:
                        time.sleep(wait)
                yield sse_event({"type": "frame", "data": frame_to_b64jpg(annotated)})

            idx += 1

        yield sse_event({"type": "log", "text": f"[{ts()}] Terminé — {idx} images traitées"})
        if sid and _stream_state(sid).get("stopped"):
            release_obj_models()
            yield sse_event({"type": "log", "text": f"[{ts()}] Modèles déchargés (mémoire GPU libérée)"})
    except Exception as e:
        yield sse_event({"type": "log", "text": f"[{ts()}] ERREUR : {e}"})
    finally:
        if cap is not None:
            cap.release()
        if sid:
            _drop_stream(sid)
    yield sse_event({"type": "done"})


# ═══════════════════════════════════════════════════════════════════════════════
# FALL DETECTION — VideoMAE (Phase 3 module pour le groupe)
# Modèle 7-classes d'actions humaines pré-entraîné Kinetics puis fine-tuné chute :
# {FallDown, LyingDown, SitDown, Sitting, StandUp, Standing, Walking}.
# On déclenche l'alerte uniquement sur FallDown.
# ═══════════════════════════════════════════════════════════════════════════════
FALL_MODEL_ID = "yadvender12/videomae-base-finetuned-kinetics-finetuned-fall-detect"
FALL_ALERT_LABELS = {"falldown"}  # casefold-comparé à id2label
# Risk-bar tuning (mirrors Fall detection/inference.py).
FALL_EMA_ALPHA = 0.3   # smoothing factor on the fall-class probability
FALL_THRESHOLD = 1 / 7  # uniform-prior baseline for the 7-class model
_fall_processor = None
_fall_model = None
_fall_clip_len = 16


def load_fall_model():
    """Lazy load the VideoMAE fall-detection model from HuggingFace, cached
    after first call. Returns (image_processor, model, expected_clip_length)."""
    global _fall_processor, _fall_model, _fall_clip_len
    if _fall_model is None:
        from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
        _fall_processor = VideoMAEImageProcessor.from_pretrained(FALL_MODEL_ID)
        _fall_model = VideoMAEForVideoClassification.from_pretrained(FALL_MODEL_ID)
        _fall_model = _fall_model.to(DEVICE).eval()
        _fall_clip_len = int(getattr(_fall_model.config, "num_frames", _fall_clip_len))
    return _fall_processor, _fall_model, _fall_clip_len


def gen_fall_sse(video_path, sid=None):
    """SSE generator. Buffers `clip_len` RGB frames in a sliding window, runs
    VideoMAE every `stride` frames, EMA-smooths the fall-class probability and
    overlays a risk gauge + alert banner. Alert is held for ~1s after dropping
    below threshold (cooldown) so the banner doesn't flicker."""
    if sid:
        _ensure_stream(sid, "fall")
    cap = None
    try:
        yield sse_event({"type": "log", "text": f"[{ts()}] Chargement du modèle VideoMAE de détection de chute..."})
        processor, model, clip_len = load_fall_model()
        id2label = getattr(model.config, "id2label", {})
        # Locate the fall class index (any label containing "fall"); fall back to 1.
        fall_class_idx = next(
            (i for i, label in id2label.items() if "fall" in str(label).lower()),
            1,
        )
        yield sse_event({"type": "log", "text": f"[{ts()}] Modèle prêt — fenêtre = {clip_len} frames, classes = {list(id2label.values())}"})

        source = _resolve_source(video_path)
        is_webcam = isinstance(source, int)
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            yield sse_event({"type": "log", "text": f"[{ts()}] ERREUR : impossible d'ouvrir la source : {video_path}"})
            yield sse_event({"type": "done"})
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w, h = int(cap.get(3)), int(cap.get(4))
        if is_webcam:
            yield sse_event({"type": "log", "text": f"[{ts()}] Webcam : {w}x{h} @ {fps:.0f}ips (temps réel)"})
        else:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            yield sse_event({"type": "log", "text": f"[{ts()}] Vidéo : {w}x{h} @ {fps:.0f}ips, {total} images"})

        clip_buf = collections.deque(maxlen=clip_len)
        # Re-classify every clip_len/2 frames to balance latency and GPU load.
        stride = max(1, clip_len // 2)
        cur_label, cur_conf = "—", 0.0
        smoothed_fall_prob = 0.0
        frames_to_keep_alert = max(1, int(fps))  # hold alert ~1s after release
        alert_cooldown = 0
        was_alert_active = False
        idx = 0
        t_start = time.time()

        while True:
            st = _stream_state(sid) if sid else {}
            if st.get("stopped"):
                yield sse_event({"type": "log", "text": f"[{ts()}] Arrêté par l'utilisateur"})
                break
            if st.get("paused"):
                time.sleep(0.1)
                continue

            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            clip_buf.append(rgb)

            # Inference once the buffer is full and at every stride boundary.
            if len(clip_buf) == clip_len and idx % stride == 0:
                inputs = processor(list(clip_buf), return_tensors="pt")
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                with torch.no_grad():
                    logits = model(**inputs).logits[0]
                    probs = torch.softmax(logits, dim=0)
                pred_idx = int(probs.argmax().item())
                cur_label = id2label.get(pred_idx, f"class_{pred_idx}")
                cur_conf = float(probs.max().item())
                raw_fall_prob = float(probs[fall_class_idx].item())
                smoothed_fall_prob = (FALL_EMA_ALPHA * raw_fall_prob
                                      + (1 - FALL_EMA_ALPHA) * smoothed_fall_prob)
                if pred_idx == fall_class_idx:
                    yield sse_event({"type": "log", "text": f"[{ts()}] Image {idx} ({idx/fps:.1f}s) : Chute détectée — {cur_label} ({cur_conf:.0%})"})

            is_falling = smoothed_fall_prob > FALL_THRESHOLD
            if is_falling:
                alert_cooldown = frames_to_keep_alert
            elif alert_cooldown > 0:
                alert_cooldown -= 1
            is_alert_active = is_falling or (alert_cooldown > 0)

            # Log only on the alert rising edge to avoid flooding the panel.
            if is_alert_active and not was_alert_active:
                yield sse_event({"type": "log", "text": f"[{ts()}] ⚠ ALERTE CHUTE Image {idx} ({idx/fps:.1f}s) : risque = {smoothed_fall_prob*100:.1f}% — {cur_label} ({cur_conf:.0%})"})
            was_alert_active = is_alert_active

            # HUD: semi-transparent black panel for legibility.
            display = frame.copy()
            panel_x2 = min(w - 10, 470)
            overlay = display.copy()
            cv2.rectangle(overlay, (10, 10), (panel_x2, 130), (0, 0, 0), -1)
            display = cv2.addWeighted(overlay, 0.6, display, 0.4, 0)

            status_text = "ALERTE CHUTE DETECTEE !" if is_alert_active else "Surveillance active"
            status_color = (0, 0, 255) if is_alert_active else (0, 200, 0)
            cv2.putText(display, status_text, (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2, cv2.LINE_AA)
            cv2.putText(display, f"{cur_label} ({cur_conf:.0%})", (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(display, f"Risque : {smoothed_fall_prob*100:.1f}%", (20, 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            # Risk gauge bar (green → red), to the right of the risk text.
            bar_x, bar_y, bar_w, bar_h = 180, 100, 250, 18
            p = max(0.0, min(1.0, smoothed_fall_prob))
            fill_w = int(bar_w * p)
            jauge_color = (0, int(255 * (1 - p)), int(255 * p))
            cv2.rectangle(display, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                          (255, 255, 255), 1)
            cv2.rectangle(display, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h),
                          jauge_color, -1)
            # Threshold tick on the gauge.
            tick_x = bar_x + int(bar_w * FALL_THRESHOLD)
            cv2.line(display, (tick_x, bar_y - 3), (tick_x, bar_y + bar_h + 3),
                     (255, 255, 0), 1)

            # Blinking red border while the alert is active.
            if is_alert_active and (idx // 10) % 2 == 0:
                cv2.rectangle(display, (0, 0), (w - 1, h - 1), (0, 0, 255), 15)

            if idx % 2 == 0:
                if not is_webcam:
                    target_time = t_start + idx / fps
                    wait = target_time - time.time()
                    if wait > 0:
                        time.sleep(wait)
                yield sse_event({"type": "frame", "data": frame_to_b64jpg(display)})

            idx += 1

        yield sse_event({"type": "log", "text": f"[{ts()}] Terminé — {idx} images traitées"})
        if sid and _stream_state(sid).get("stopped"):
            release_fall_models()
            yield sse_event({"type": "log", "text": f"[{ts()}] Modèles déchargés (mémoire GPU libérée)"})
    except Exception as e:
        yield sse_event({"type": "log", "text": f"[{ts()}] ERREUR : {e}"})
    finally:
        if cap is not None:
            cap.release()
        if sid:
            _drop_stream(sid)
    yield sse_event({"type": "done"})


# ═══════════════════════════════════════════════════════════════════════════════
# FLASK ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(gen_login_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/check_login")
def check_login():
    if _logged_in_user:
        return jsonify({"success": True, "name": _logged_in_user})
    return jsonify({"success": False, "name": "Guest"})


@app.route("/logout", methods=["POST"])
def logout():
    global _logged_in_user
    _logged_in_user = None
    return jsonify({"ok": True})


# ── File upload endpoints (save file, return path token) ──
_uploads = {}  # uid -> {"path": str, "time": float}


@app.route("/api/upload", methods=["POST"])
def api_upload():
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "no file"}), 400
    uid = str(uuid.uuid4())[:8]
    ext = os.path.splitext(f.filename)[1]
    path = os.path.join(OUTPUTS_DIR, f"upload_{uid}{ext}")
    f.save(path)
    _uploads[uid] = {"path": path, "time": time.time()}
    # Clean up stale uploads (older than 10 minutes)
    stale = [k for k, v in _uploads.items() if time.time() - v["time"] > 600]
    for k in stale:
        try:
            os.remove(_uploads[k]["path"])
        except OSError:
            pass
        del _uploads[k]
    return jsonify({"id": uid})


# ── SSE streaming endpoints ──


def _pop_upload(uid):
    """Pop an uploaded file path by uid, or return None."""
    if uid:
        entry = _uploads.pop(uid, None)
        if entry:
            return entry["path"]
    return None


def _resolve_stream_source(args, default_path):
    """For a stream request: pick webcam if requested, else the uploaded file
    if any, else the module's default video path."""
    if args.get("source") == "webcam":
        return "webcam"
    return _pop_upload(args.get("upload")) or default_path


@app.route("/stream/fire")
def stream_fire():
    src = _resolve_stream_source(request.args, "data/test/fire.mp4")
    sid = request.args.get("sid")
    return Response(gen_fire_sse(src, sid=sid), mimetype="text/event-stream")


@app.route("/stream/objects")
def stream_objects():
    src = _resolve_stream_source(request.args, "data/test/Objects.MOV")
    sid = request.args.get("sid")
    return Response(gen_objects_sse(src, sid=sid), mimetype="text/event-stream")


@app.route("/stream/fall")
def stream_fall():
    src = _resolve_stream_source(request.args, "data/test/falling0.mp4")
    sid = request.args.get("sid")
    return Response(gen_fall_sse(src, sid=sid), mimetype="text/event-stream")


@app.route("/stream/control", methods=["POST"])
def stream_control():
    """Pause / resume / stop a running stream by sid. `release` releases the
    model weights for a given module (called by the UI's Stop button even when
    no stream is active, e.g. to free GPU memory after a stream ended)."""
    data = request.get_json(silent=True) or {}
    sid = data.get("sid")
    action = data.get("action")
    module = data.get("module")
    if action == "pause":
        _set_stream(sid, paused=True)
    elif action == "resume":
        _set_stream(sid, paused=False)
    elif action == "stop":
        _set_stream(sid, stopped=True)
    elif action == "release":
        fn = _RELEASE_FNS.get(module)
        if fn:
            fn()
    else:
        return jsonify({"error": "unknown action"}), 400
    return jsonify({"ok": True})


@app.route("/outputs/<path:filename>")
def serve_output(filename):
    return send_from_directory(OUTPUTS_DIR, filename)


@app.route("/data/<path:filename>")
def serve_data(filename):
    return send_from_directory("data", filename)


def preload_models():
    """Warm up heavy models at startup so the first inference click is fast."""
    print("Préchargement des modèles...")
    t0 = time.time()
    for name, loader in [
        ("fire", load_fire_models),
        ("objects", load_obj_model),
        ("fall", load_fall_model),
    ]:
        t1 = time.time()
        try:
            loader()
            print(f"  ✓ {name} prêt ({time.time()-t1:.1f}s)")
        except Exception as e:
            print(f"  ⚠ {name} non chargé : {e}")
    print(f"Préchargement terminé en {time.time()-t0:.1f}s")


if __name__ == "__main__":
    load_authorized_faces()  # disabled: no webcam available
    preload_models()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
