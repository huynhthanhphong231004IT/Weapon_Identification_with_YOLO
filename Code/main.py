from collections import defaultdict, deque
import threading
import time

import numpy as np
import cv2

try:  
    import tkinter as tk
    from tkinter import filedialog, ttk
    from PIL import Image, ImageTk
except Exception: 
    tk = ttk = filedialog = Image = ImageTk = None

MODEL_PATH = "../Model/train_run/weights/best.pt"
SMOOTH_FRAMES = 25  

C = {
    "bg": "#0a0e16", "panel": "#111827", "panel2": "#0f172a", "border": "#1f2a3d",
    "text": "#e5e7eb", "muted": "#94a3b8", "accent": "#22d3ee", "accent2": "#3b82f6",
    "ok": "#10b981", "warn": "#f59e0b", "danger": "#ef4444", "orange": "#fb923c",
    "yellow": "#facc15", "violet": "#e879f9", "log_bg": "#060910", "log_fg": "#5eead4",
}

CLASS_HEX = {"Gun": C["danger"], "Knife": C["orange"], "Scissors": C["yellow"], "Uncertain": C["violet"], "Weapon": C["danger"]}
CLASS_BGR = {"Gun": (68, 68, 239), "Knife": (60, 146, 251), "Scissors": (21, 204, 250), "Uncertain": (249, 121, 232), "Weapon": (68, 68, 239)}

def canon(name):
    n = str(name).lower()
    if any(k in n for k in ("gun", "pistol", "firearm", "rifle", "handgun", "revolver", "shotgun")):
        return "Gun"
    if any(k in n for k in ("knife", "blade", "dagger", "sword", "machete")):
        return "Knife"
    if any(k in n for k in ("scissor", "scissors", "shear")):
        return "Scissors"
    if any(k in n for k in ("weapon", "threat", "arm")):
        return "Weapon"
    return str(name).capitalize()

def analyze_threat_level(family, feats=None, conf=0.0):
    fam = str(family).lower()
    if "gun" in fam:
        return "CRITICAL", 3, "HIGH RISK: Firearm detected. Extreme lethal range."
    if "knife" in fam:
        if feats and feats.get("aspect", 0) > 8.0:
            return "CRITICAL", 3, "HIGH RISK: Long blade/machete style detected."
        return "HIGH", 2, "MEDIUM-HIGH RISK: Edged weapon detected. Close range lethality."
    if "scissor" in fam:
        return "MODERATE", 1, "MODERATE RISK: Pointed tool/scissors detected."
    if feats and feats.get("metal", 0) > 0.4 and feats.get("aspect", 0) > 3.0:
        return "HIGH", 2, "ELEVATED RISK: Metallic rigid potential weapon."
    return "ELEVATED", 2, "ELEVATED RISK: General potential threat detected."

def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0

def tri(x, a, b, c, d):
    if x <= a or x >= d:
        return 0.0
    if b <= x <= c:
        return 1.0
    return (x - a) / (b - a + 1e-9) if x < b else (d - x) / (d - c + 1e-9)

def clip01(x):
    return float(min(1.0, max(0.0, x)))

class ShapeColorAnalyzer:
    MIN_SIDE = 18  

    def segment(self, frame, box):
        H, W = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W - 1, x2), min(H - 1, y2)
        bw, bh = x2 - x1, y2 - y1
        if bw < self.MIN_SIDE or bh < self.MIN_SIDE:
            return None
        pad = int(0.12 * max(bw, bh))
        cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
        cx2, cy2 = min(W, x2 + pad), min(H, y2 + pad)
        crop = frame[cy1:cy2, cx1:cx2]
        rx, ry = x1 - cx1, y1 - cy1
        s = min(1.0, 192.0 / max(crop.shape[:2]))
        if s < 1.0:
            crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        rect = (int(rx * s), int(ry * s), max(4, int(bw * s)), max(4, int(bh * s)))
        rect_area = rect[2] * rect[3]

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        inside = np.zeros(crop.shape[:2], np.uint8)
        inside[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]] = 1

        fg = np.zeros_like(inside)
        ring = (inside == 0)
        if ring.sum() > 40:
            lab_img = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
            bg_med = np.median(lab_img[ring], axis=0)
            dist = np.linalg.norm(lab_img - bg_med, axis=2)
            dist = cv2.GaussianBlur(dist, (3, 3), 0)
            d8 = np.clip(dist / max(dist[inside > 0].max(), 1e-6) * 255, 0, 255).astype(np.uint8)
            thr, _ = cv2.threshold(d8[inside > 0].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            cand = ((d8 > thr) & (inside > 0)).astype(np.uint8)
            h_, s_, v_ = hsv[..., 0], hsv[..., 1], hsv[..., 2]
            skin = (h_ <= 25) & (s_ > 35) & (s_ < 190) & (v_ > 110)
            cand[skin] = 0
            if 0.08 < cand.sum() / rect_area < 0.85:
                fg = cand
        if fg.sum() == 0:
            mask = np.zeros(crop.shape[:2], np.uint8)
            try:
                bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
                m = max(1, int(0.03 * min(rect[2], rect[3])))
                r = (rect[0] + m, rect[1] + m, max(2, rect[2] - 2 * m), max(2, rect[3] - 2 * m))
                cv2.grabCut(crop, mask, r, bgd, fgd, 2, cv2.GC_INIT_WITH_RECT)
                fg = ((mask == 1) | (mask == 3)).astype(np.uint8) * inside
            except cv2.error:
                fg = np.zeros_like(inside)

        frac = fg.sum() / max(rect_area, 1)
        if frac < 0.06 or frac > 0.93:  
            dark = ((hsv[..., 2] < 90).astype(np.uint8)) * inside
            if dark.sum() / rect_area > 0.08:
                fg = dark
            else:
                g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                _, th = cv2.threshold(g, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                if th[inside > 0].mean() > 0.5:
                    th = 1 - th
                fg = th.astype(np.uint8) * inside

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k)
        n, lab, stats, cent = cv2.connectedComponentsWithStats(fg, connectivity=8)
        if n <= 1:
            return None
        ctr = np.array([rect[0] + rect[2] / 2, rect[1] + rect[3] / 2])
        diag = np.hypot(rect[2], rect[3]) + 1e-9
        best, best_score = None, 0
        for i in range(1, n):
            a = stats[i, cv2.CC_STAT_AREA]
            if a < 0.03 * rect_area:
                continue
            sc = a * (1 - 0.6 * np.linalg.norm(cent[i] - ctr) / diag)
            if sc > best_score:
                best, best_score = i, sc
        if best is None:
            return None
        return (lab == best).astype(np.uint8), crop, hsv, rect_area

    def analyze(self, frame, box):
        seg = self.segment(frame, box)
        if seg is None:
            return None
        mask, crop, hsv, rect_area = seg
        area = float(mask.sum())
        if area < 30:
            return None

        h, s, v = hsv[..., 0][mask > 0], hsv[..., 1][mask > 0], hsv[..., 2][mask > 0]
        n = float(len(v))
        dark = float(np.mean(v < 85))
        gray = float(np.mean((s < 50) & (v >= 85) & (v < 140)))
        metal = float(np.mean((s < 50) & (v >= 140)))
        colorful = float(np.mean((s > 90) & (v > 80)))

        cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        outer_ids = [i for i in range(len(cnts)) if hier[0][i][3] == -1]
        oi = max(outer_ids, key=lambda i: cv2.contourArea(cnts[i]))
        outer = cnts[oi]
        holes = sum(1 for i in range(len(cnts))
                    if hier[0][i][3] == oi and cv2.contourArea(cnts[i]) > 0.012 * area)
        (_, _), (rw, rh), _ = cv2.minAreaRect(outer)
        L, Wd = max(rw, rh), max(1.0, min(rw, rh))
        aspect = L / Wd
        rectangularity = area / (L * Wd + 1e-9)
        hull = cv2.convexHull(outer)
        solidity = area / (cv2.contourArea(hull) + 1e-9)

        deep_defects = 0
        try:
            hi = cv2.convexHull(outer, returnPoints=False)
            if hi is not None and len(hi) > 3:
                defects = cv2.convexityDefects(outer, hi)
                if defects is not None:
                    d = np.asarray(defects).reshape(-1, 4)  
                    deep_defects = int(np.sum(d[:, 3] / 256.0 > 0.18 * L))
        except cv2.error:
            pass

        ys, xs = np.nonzero(mask)
        pts = np.stack([xs, ys], 1).astype(np.float64)
        pts -= pts.mean(0)
        evals, evecs = np.linalg.eigh(np.cov(pts.T) + 1e-9 * np.eye(2))
        major = evecs[:, 1]
        minor = evecs[:, 0]
        u, vv = pts @ major, pts @ minor
        ulen = u.max() - u.min() + 1e-9
        nb = 20
        bins = np.minimum(((u - u.min()) / ulen * nb).astype(int), nb - 1)
        widths = np.zeros(nb)
        for b in range(nb):
            sel = vv[bins == b]
            widths[b] = (sel.max() - sel.min()) if len(sel) > 1 else 0.0
        wmax = widths.max() + 1e-9
        wn = widths / wmax
        min_end = float(min(wn[:2].mean(), wn[-2:].mean()))  
        max_step = float(np.max(np.abs(np.diff(wn))))       

        gw = 48
        gu = np.clip(((u - u.min()) / ulen * (gw - 1)).astype(int), 0, gw - 1)
        vspan = max(np.abs(vv).max() * 2, 1e-9)
        gv = np.clip(((vv + vspan / 2) / vspan * (gw - 1)).astype(int), 0, gw - 1)
        grid = np.zeros((gw, gw), np.uint8)
        grid[gu, gv] = 1
        flip = grid[:, ::-1]
        symmetry = float((grid & flip).sum() / max(1, (grid | flip).sum()))

        size_q = clip01(np.sqrt(rect_area) / 70.0)
        area_q = 1.0 if 0.08 < area / rect_area < 0.9 else 0.5
        return dict(dark=dark, gray=gray, metal=metal, colorful=colorful, holes=holes,
                    aspect=float(aspect), rect=float(rectangularity), solidity=float(solidity),
                    deep_defects=deep_defects, min_end=min_end, max_step=max_step,
                    symmetry=symmetry, quality=float(0.25 + 0.75 * size_q * area_q),
                    mask=mask)

    @staticmethod
    def score(f, known_classes):
        blunt = clip01((f["min_end"] - 0.28) / 0.17)
        sharp = clip01((0.34 - f["min_end"]) / 0.16)
        metal_c = clip01(f["metal"] / 0.2)
        gray_c = clip01(f["gray"] / 0.35)
        dark_c = clip01(f["dark"] + 0.5 * f["gray"])
        asym = 1 - f["symmetry"]

        scores = {}
        for c in known_classes:
            cl = c.lower()
            if "gun" in cl:
                scores[c] = (0.18 * gray_c + 0.12 * dark_c + 0.30 * blunt + 0.22 * (1 - metal_c)
                             + 0.08 * asym + 0.10 * tri(f["aspect"], 1.0, 1.3, 6.0, 9.0))
            elif "knife" in cl:
                scores[c] = (0.32 * sharp + 0.32 * metal_c + 0.12 * tri(f["aspect"], 1.0, 1.5, 14, 22)
                             + 0.10 * (1.0 if f["holes"] == 0 else 0.3) + 0.14 * (1 - gray_c))
            elif "scissor" in cl:
                holes_s = 1.0 if f["holes"] >= 2 else (0.2 if f["holes"] == 1 else 0.0)
                scores[c] = (0.30 * holes_s + 0.18 * clip01((f["symmetry"] - 0.55) / 0.3) + 0.14 * sharp
                             + 0.14 * clip01(f["deep_defects"] / 2.0)
                             + 0.14 * clip01(f["metal"] * 1.2 + f["colorful"])
                             + 0.10 * tri(f["aspect"], 1.5, 2.0, 4.0, 6.0))
            else:
                scores[c] = 0.33

        tot = sum(scores.values()) + 1e-9
        return {k: v / tot for k, v in scores.items()}, scores

    @staticmethod
    def cues(f):
        gun_cue = (f["min_end"] >= 0.38 and f["dark"] + f["gray"] + f["metal"] > 0.45
                   and f["holes"] < 2 and f["aspect"] < 6.0)
        knife_cue = (f["min_end"] <= 0.30 and f["metal"] >= 0.15 and f["holes"] == 0)
        return gun_cue, knife_cue

class Track:
    _next = 1

    def __init__(self, box, votes, known_classes):
        self.id = Track._next
        Track._next += 1
        self.box = np.array(box, dtype=float)
        self.known_classes = known_classes
        self.yolo = {c: votes.get(c, 0.0) for c in known_classes}
        self.geo = {c: 1.0 / max(len(known_classes), 1) for c in known_classes}
        self.geo_q = 0.0
        self.hits, self.age = 1, 0
        self.label, self.pending, self.pending_n = None, None, 0
        self.last_conf, self.feats, self.reason = 0.0, None, ""
        self.probs = {c: 1.0 / max(len(known_classes), 1) for c in known_classes}
        self.uncertain, self.yolo_top = False, None
        self.history = deque(maxlen=30)
        self.prob_hist = deque(maxlen=SMOOTH_FRAMES)  
        self.conf_sum, self.conf_n = max(votes.values()) if votes else 0.0, 1
        self.acc = None
        self.family = "Weapon"
        self.wl = 1.0

    @property
    def mean_conf(self):
        return self.conf_sum / max(self.conf_n, 1)

    def update_box(self, box):
        self.box = 0.6 * np.array(box, dtype=float) + 0.4 * self.box

class WeaponVerifier:
    SWITCH_MARGIN, SWITCH_FRAMES = 0.05, 4
    MIN_HITS = 4

    def __init__(self, known_classes=None):
        self.an = ShapeColorAnalyzer()
        self.tracks, self.frame_no = [], 0
        self.override_count = 0
        self.session = defaultdict(int)     
        self._counted = set()
        self.known_classes = known_classes if known_classes else ["Gun", "Knife", "Scissors"]

    def set_known_classes(self, classes):
        for c in classes:
            if c not in self.known_classes:
                self.known_classes.append(c)

    def cluster(self, dets):
        clusters = []
        for d in sorted(dets, key=lambda d: -d["conf"]):
            cls = d["family"]
            for c in clusters:
                if iou(c["box"], d["box"]) >= 0.45:
                    c["votes"][cls] = max(c["votes"].get(cls, 0), d["conf"])
                    break
            else:
                clusters.append({"box": d["box"], "votes": {cls: d["conf"]}})
        return clusters

    def _match(self, clusters):
        pairs = sorted(((iou(t.box, c["box"]), ti, ci) for ti, t in enumerate(self.tracks)
                        for ci, c in enumerate(clusters)), reverse=True)
        used_t, used_c, out = set(), set(), []
        for sc, ti, ci in pairs:
            if sc < 0.2 or ti in used_t or ci in used_c:
                continue
            used_t.add(ti); used_c.add(ci); out.append((ti, ci))
        return out, used_t, used_c

    def update(self, frame, dets):
        self.frame_no += 1
        events = []
        det_classes = [d["family"] for d in dets]
        self.set_known_classes(det_classes)

        clusters = self.cluster(dets)
        matches, used_t, used_c = self._match(clusters)

        for ti, ci in matches:
            t, c = self.tracks[ti], clusters[ci]
            t.update_box(c["box"]); t.hits += 1; t.age = 0
            t.conf_sum += max(c["votes"].values()); t.conf_n += 1
            for k in self.known_classes:
                t.yolo[k] = 0.85 * t.yolo.get(k, 0.0) + 0.15 * c["votes"].get(k, 0.0)
            self._think(frame, t, c["votes"], events)
        for ti, t in enumerate(self.tracks):
            if ti not in used_t:
                t.age += 1
        for ci, c in enumerate(clusters):
            if ci in used_c:
                continue
            t = Track(c["box"], c["votes"], self.known_classes)
            self.tracks.append(t)
            self._think(frame, t, c["votes"], events)
        self.tracks = [t for t in self.tracks if t.age <= 12]
        shown = [t for t in self.tracks if t.hits >= self.MIN_HITS and t.age == 0 and t.label
                 and t.mean_conf >= 0.33
                 and not (t.mean_conf < 0.42 and t.wl < 0.75)]   
        shown.sort(key=lambda t: -t.hits)          
        keep = []
        for t in shown:
            if all(iou(t.box, k.box) < 0.45 for k in keep):
                keep.append(t)
        return keep, events

    def _think(self, frame, t, votes, events):
        classes = self.known_classes
        feats = self.an.analyze(frame, t.box)
        yv_s = np.array([t.yolo.get(k, 0.0) for k in classes])
        t.yolo_top = classes[int(np.argmax(yv_s))] if yv_s.sum() > 0 else (t.yolo_top or classes[0])
        t.last_conf = max(votes.values()) if votes else t.last_conf
        yv = 0.5 * np.array([votes.get(k, 0.0) for k in classes]) + 0.5 * yv_s
        y = (yv + 0.25) / (yv.sum() + 0.75)   
        y = y / y.sum()
        conf_f = max(votes.values()) if votes else 0.0
        reason, w_geo, g = [], 0.0, np.ones(len(classes)) / len(classes)
        if feats is not None:
            geo, _ = self.an.score(feats, classes)
            g = np.array([geo.get(k, 0.0) for k in classes])
            t.geo = {k: float(geo.get(k, 0.0)) for k in classes}
            t.geo_q = 0.7 * t.geo_q + 0.3 * feats["quality"]
            w_geo = (0.55 + 0.30 * max(0.0, 0.65 - conf_f) / 0.65) * feats["quality"]
            t.feats = feats
            wl = 1.0 - clip01((feats["colorful"] - 0.18) / 0.20)
            t.wl = wl if t.hits <= 1 else 0.8 * t.wl + 0.2 * wl
        p = (1 - w_geo) * y + w_geo * g

        gun_i = classes.index("Gun") if "Gun" in classes else -1
        knife_i = classes.index("Knife") if "Knife" in classes else -1
        scis_i = classes.index("Scissors") if "Scissors" in classes else -1

        if feats is not None:
            gun_cue, knife_cue = self.an.cues(feats)
            if gun_cue and not knife_cue and gun_i != -1:
                p[gun_i] += 0.20; reason.append("hong tu + mau duc -> Gun")
            elif knife_cue and not gun_cue and knife_i != -1:
                p[knife_i] += 0.20; reason.append("mui nhon + kim loai sang -> Knife")
            elif gun_cue and knife_cue:
                reason.append("tin hieu mau thuan")
            if scis_i != -1 and p[scis_i] > 0.3 and feats["holes"] < 2 and feats["symmetry"] < 0.6:
                p[scis_i] *= 0.75; reason.append("thieu vong cam -> giam Scissors")
        p = p / p.sum()
        if t.acc is None or len(t.acc) != len(p):
            t.acc = p.copy()
        else:
            t.acc = 0.94 * t.acc + 0.06 * p
        final = t.acc / t.acc.sum()
        t.prob_hist.append(final)
        final = np.mean(t.prob_hist, axis=0)
        final = final / final.sum()
        order = np.argsort(-final)
        lead = classes[order[0]]
        margin = float(final[order[0]] - final[order[1]]) if len(order) > 1 else 1.0
        t.probs = {k: float(final[i]) for i, k in enumerate(classes)}
        t.uncertain = margin < 0.05 and len(order) > 1 and {lead, classes[order[1]]} == {"Gun", "Knife"}
        t.family = lead

        if t.label is None:
            t.label = lead
        elif lead != t.label:
            if t.pending == lead:
                t.pending_n += 1
            else:
                t.pending, t.pending_n = lead, 1
            if t.pending_n >= self.SWITCH_FRAMES and margin > self.SWITCH_MARGIN:
                events.append(f"ID:{t.id} doi nhan {t.label} -> {lead} (margin {margin:.2f})")
                t.label, t.pending, t.pending_n = lead, None, 0
        else:
            t.pending, t.pending_n = None, 0

        if t.label != t.yolo_top and t.hits >= self.MIN_HITS:
            reason.append(f"Verifier sua YOLO:{t.yolo_top}")
            key = (t.id, t.label)
            if key not in self._counted:
                self._counted.add(key); self.override_count += 1
                events.append(f"ID:{t.id} Verifier: YOLO={t.yolo_top} -> {t.label}")
        if t.hits == self.MIN_HITS and t.id not in self._counted:
            self._counted.add(t.id); self.session[t.label] += 1
            events.append(f"ID:{t.id} xac nhan muc tieu: Weapon ({t.label}) ({t.probs[t.label]*100:.0f}%)")
        if feats is not None:
            reason.insert(0, f"dark {feats['dark']*100:.0f}% metal {feats['metal']*100:.0f}% "
                             f"aspect {feats['aspect']:.1f} tip {feats['min_end']:.2f} "
                             f"holes {feats['holes']}")
        t.reason = " | ".join(reason)

def display_name(t):
    return "Uncertain" if t.uncertain else t.label

def draw_track(img, t):
    family = display_name(t)
    col = CLASS_BGR.get(family, (68, 68, 239))
    x1, y1, x2, y2 = [int(v) for v in t.box]
    th = max(2, int(round(min(img.shape[:2]) / 360)))
    cl = max(10, int(min(x2 - x1, y2 - y1) * 0.25))
    for (px, py, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(img, (px, py), (px + dx * cl, py), col, th + 1, cv2.LINE_AA)
        cv2.line(img, (px, py), (px, py + dy * cl), col, th + 1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1), (x2, y2), col, 1, cv2.LINE_AA)
    txt = "Gun/Knife?" if family == "Uncertain" else family
    label = f"ID:{t.id} Weapon ({txt}) {t.probs.get(t.label, 0.0)*100:.0f}%"
    fs = max(0.5, min(img.shape[:2]) / 1100)
    (tw, tht), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
    ty = y1 - 6 if y1 - tht - 12 > 0 else y2 + tht + 10
    cv2.rectangle(img, (x1, ty - tht - 6), (x1 + tw + 10, ty + 4), col, -1)
    cv2.putText(img, label, (x1 + 5, ty - 1), cv2.FONT_HERSHEY_SIMPLEX, fs, (10, 10, 10), 1, cv2.LINE_AA)

class WeaponApp:
    def __init__(self, root):
        from ultralytics import YOLO
        self.root = root
        self.root.title("AegisVision - Advanced Threat Intelligence & Weapon Detection")
        self.root.geometry("1500x900")
        self.root.configure(bg=C["bg"])

        self.model = YOLO(MODEL_PATH)
        self.cap, self.is_running, self.thread = None, False, None
        
        model_classes = list(self.model.names.values()) if hasattr(self.model, "names") else ["Gun", "Knife", "Scissors"]
        known_families = list(set([canon(c) for c in model_classes]))
        self.verifier = WeaponVerifier(known_classes=known_families)
        
        self.current_frame_cv, self.photo_image = None, None
        self._ui_busy = False
        self.prob_rows = {}
        self.setup_styles()
        self.setup_ui()

    def setup_styles(self):
        st = ttk.Style()
        st.theme_use("clam")
        st.configure("TProgressbar", thickness=8, troughcolor=C["panel2"], background=C["accent2"],
                     bd=0, lightcolor=C["accent2"], darkcolor=C["accent2"], bordercolor=C["panel2"])
        st.configure("Weapon.Horizontal.TProgressbar", thickness=8, troughcolor=C["panel2"],
                     background=C["danger"], lightcolor=C["danger"], darkcolor=C["danger"], bordercolor=C["panel2"])
        for name, col in (("Gun", C["danger"]), ("Knife", C["orange"]), ("Scissors", C["yellow"]),
                          ("Conf", C["accent"])):
            st.configure(f"{name}.Horizontal.TProgressbar", thickness=8, troughcolor=C["panel2"],
                         background=col, lightcolor=col, darkcolor=col, bordercolor=C["panel2"])

    def _card(self, parent, row, pady=(6, 6), px=12):
        f = tk.Frame(parent, bg=C["panel"], padx=14, pady=10, highlightbackground=C["border"],
                     highlightthickness=1)
        f.grid(row=row, column=0, sticky="ew", padx=px, pady=pady)
        return f

    def _title(self, parent, text):
        tk.Label(parent, text=text, font=("Segoe UI", 9, "bold"), bg=C["panel"],
                 fg=C["accent"]).pack(anchor="w", pady=(0, 6))

    def setup_ui(self):
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        main = tk.Frame(self.root, bg=C["bg"])
        main.grid(row=0, column=0, sticky="nsew", padx=14, pady=14)
        main.rowconfigure(0, weight=1)
        main.columnconfigure(0, weight=3)
        main.columnconfigure(1, weight=1, minsize=380)

        left = tk.Frame(main, bg=C["panel2"], highlightbackground=C["border"], highlightthickness=1)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 7))
        left.rowconfigure(0, weight=1); left.columnconfigure(0, weight=1)

        right_outer = tk.Frame(main, bg=C["panel2"], highlightbackground=C["border"], highlightthickness=1)
        right_outer.grid(row=0, column=1, sticky="nsew", padx=(7, 0))
        right_outer.rowconfigure(0, weight=1)
        right_outer.columnconfigure(0, weight=1)

        self.right_canvas = tk.Canvas(right_outer, bg=C["panel2"], highlightthickness=0, bd=0)
        self.right_scrollbar = ttk.Scrollbar(right_outer, orient="vertical", command=self.right_canvas.yview)
        right = tk.Frame(self.right_canvas, bg=C["panel2"])

        right.bind("<Configure>", lambda e: self.right_canvas.configure(scrollregion=self.right_canvas.bbox("all")))
        self.right_canvas_window = self.right_canvas.create_window((0, 0), window=right, anchor="nw")
        
        def _on_canvas_configure(e):
            self.right_canvas.itemconfig(self.right_canvas_window, width=e.width)

        self.right_canvas.bind("<Configure>", _on_canvas_configure)
        self.right_canvas.configure(yscrollcommand=self.right_scrollbar.set)

        self.right_canvas.grid(row=0, column=0, sticky="nsew")
        self.right_scrollbar.grid(row=0, column=1, sticky="ns")

        def _on_mousewheel(event):
            self.right_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def _bind_mousewheel(event):
            self.right_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        def _unbind_mousewheel(event):
            self.right_canvas.unbind_all("<MouseWheel>")

        self.right_canvas.bind("<Enter>", _bind_mousewheel)
        self.right_canvas.bind("<Leave>", _unbind_mousewheel)

        right.columnconfigure(0, weight=1)

        self.video_canvas = tk.Canvas(left, bg="#03050a", highlightthickness=0)
        self.video_canvas.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.video_canvas.bind("<Configure>", lambda e: self.current_frame_cv is not None
                               and self.render_frame(self.current_frame_cv))

        ctrl = tk.Frame(left, bg=C["panel2"], height=65)
        ctrl.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))
        bs = {"font": ("Segoe UI", 10, "bold"), "bg": C["accent2"], "fg": "#ffffff",
              "activebackground": "#2563eb", "activeforeground": "#ffffff", "padx": 16, "pady": 8,
              "bd": 0, "cursor": "hand2", "relief": "flat"}
        tk.Button(ctrl, text="Upload Video", command=self.upload_video, **bs).pack(side=tk.LEFT, padx=(0, 8))
        tk.Button(ctrl, text="Live Feed", command=self.open_webcam, **bs).pack(side=tk.LEFT, padx=8)
        tk.Button(ctrl, text="Stop Stream", command=self.stop_stream, bg="#b91c1c", fg="#ffffff",
                  activebackground="#991b1b", activeforeground="#ffffff", font=("Segoe UI", 10, "bold"),
                  padx=16, pady=8, bd=0, cursor="hand2", relief="flat").pack(side=tk.LEFT, padx=8)

        hd = tk.Frame(right, bg="#131c31", height=40)
        hd.grid(row=0, column=0, sticky="ew", padx=1, pady=1)
        tk.Label(hd, text="THREAT INTELLIGENCE CORE", font=("Segoe UI", 11, "bold"),
                 bg="#131c31", fg=C["accent"]).pack(pady=10)

        st = self._card(right, 1, pady=(12, 6))
        self.lbl_status = tk.Label(st, text="System Standby", font=("Segoe UI", 10, "bold"),
                                   bg=C["panel"], fg=C["muted"])
        self.lbl_status.pack(anchor="w", pady=2)
        self.lbl_count = tk.Label(st, text="Active Tracked Targets: 0", font=("Segoe UI", 10),
                                  bg=C["panel"], fg=C["text"])
        self.lbl_count.pack(anchor="w", pady=2)
        self.lbl_class = tk.Label(st, text="Classification Signature: None", font=("Segoe UI", 10),
                                  bg=C["panel"], fg=C["text"])
        self.lbl_class.pack(anchor="w", pady=2)

        an = self._card(right, 2)
        self._title(an, "SECURITY RISK METRICS")
        self.lbl_risk_level = tk.Label(an, text="Risk Assessment: SECURE", font=("Segoe UI", 10, "bold"),
                                       bg=C["panel"], fg=C["ok"])
        self.lbl_risk_level.pack(anchor="w", pady=2)
        self.lbl_confidence_avg = tk.Label(an, text="Confidence Vector: 0.0%", font=("Segoe UI", 10),
                                           bg=C["panel"], fg=C["text"])
        self.lbl_confidence_avg.pack(anchor="w", pady=2)
        self.progress_bar = ttk.Progressbar(an, orient="horizontal", length=270, mode="determinate",
                                            style="Conf.Horizontal.TProgressbar")
        self.progress_bar.pack(anchor="w", fill="x", pady=(8, 4))
        self.lbl_recommendation = tk.Label(an, text="Protocol: Perimeter monitoring active.",
                                           font=("Segoe UI", 9, "italic"), bg=C["panel"], fg=C["muted"],
                                           wraplength=320, justify="left")
        self.lbl_recommendation.pack(anchor="w", pady=(6, 0))

        self.pr_card = self._card(right, 3)
        self._title(self.pr_card, "WEAPON FAMILY PROBABILITY")
        self.prob_bars, self.prob_lbls = {}, {}
        self.update_prob_bars_ui(self.verifier.known_classes)

        vf = self._card(right, 4)
        self._title(vf, "VERIFIER ANALYSIS")
        self.lbl_verdict = tk.Label(vf, text="Verdict: —", font=("Segoe UI", 9, "bold"), bg=C["panel"],
                                    fg=C["text"], anchor="w")
        self.lbl_verdict.pack(anchor="w")
        self.lbl_color = tk.Label(vf, text="Colour : —", font=("Consolas", 9), bg=C["panel"], fg=C["muted"])
        self.lbl_color.pack(anchor="w", pady=(4, 0))
        self.lbl_shape = tk.Label(vf, text="Shape  : —", font=("Consolas", 9), bg=C["panel"], fg=C["muted"])
        self.lbl_shape.pack(anchor="w")
        self.lbl_reason = tk.Label(vf, text="Logic  : —", font=("Segoe UI", 9), bg=C["panel"], fg=C["muted"],
                                   wraplength=330, justify="left")
        self.lbl_reason.pack(anchor="w", pady=(4, 0))
        tl = self._card(right, 5)
        self.lbl_fps = tk.Label(tl, text="Stream FPS: 0.0 | Latency: 0ms", font=("Consolas", 9),
                                bg=C["panel"], fg=C["accent"])
        self.lbl_fps.pack(anchor="w")
        self.lbl_session = tk.Label(tl, text="Session Threats: 0 | Verifier fixes: 0",
                                    font=("Consolas", 9), bg=C["panel"], fg=C["muted"])
        self.lbl_session.pack(anchor="w", pady=(2, 0))

        tk.Label(right, text="Real-time Audit Trail:", font=("Segoe UI", 10, "bold"),
                 bg=C["panel2"], fg=C["text"]).grid(row=6, column=0, sticky="w", padx=12, pady=(10, 4))
        lc = tk.Frame(right, bg=C["log_bg"], highlightbackground=C["border"], highlightthickness=1)
        lc.grid(row=7, column=0, sticky="nsew", padx=12, pady=(0, 12))
        lc.rowconfigure(0, weight=1); lc.columnconfigure(0, weight=1)
        self.log_text = tk.Text(lc, bg=C["log_bg"], fg=C["log_fg"], font=("Consolas", 9), bd=0,
                                state=tk.DISABLED, highlightthickness=0, padx=8, pady=8, height=6)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        self.log_message("System telemetry online. Core neural network + verifier loaded.")

    def update_prob_bars_ui(self, known_classes):
        st = ttk.Style()
        for k in known_classes:
            if k not in self.prob_bars:
                col = CLASS_HEX.get(k, C["danger"])
                st.configure(f"{k}.Horizontal.TProgressbar", thickness=8, troughcolor=C["panel2"],
                             background=col, lightcolor=col, darkcolor=col, bordercolor=C["panel2"])
                row = tk.Frame(self.pr_card, bg=C["panel"])
                row.pack(fill="x", pady=2)
                tk.Label(row, text=k, width=10, anchor="w", font=("Segoe UI", 9, "bold"), bg=C["panel"],
                         fg=col).pack(side=tk.LEFT)
                pb = ttk.Progressbar(row, orient="horizontal", mode="determinate",
                                     style=f"{k}.Horizontal.TProgressbar", maximum=100)
                pb.pack(side=tk.LEFT, fill="x", expand=True, padx=6)
                lb = tk.Label(row, text="0%", width=5, anchor="e", font=("Consolas", 9), bg=C["panel"], fg=C["text"])
                lb.pack(side=tk.LEFT)
                self.prob_bars[k], self.prob_lbls[k] = pb, lb
                self.prob_rows[k] = row

    def log_message(self, message):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, f">> {message}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def start_stream(self, source):
        if self.is_running:
            self.stop_stream()
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            self.log_message("CRITICAL: Failed to bind video stream source!")
            return
        
        model_classes = list(self.model.names.values()) if hasattr(self.model, "names") else ["Gun", "Knife", "Scissors"]
        known_families = list(set([canon(c) for c in model_classes]))
        self.verifier = WeaponVerifier(known_classes=known_families)
        
        self.is_running = True
        self.thread = threading.Thread(target=self.process_video, daemon=True)
        self.thread.start()

    def upload_video(self):
        p = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")])
        if p:
            self.log_message(f"Target stream loaded: {p.split('/')[-1]}")
            self.start_stream(p)

    def open_webcam(self):
        self.log_message("Initializing primary optical sensor array (Webcam)...")
        self.start_stream(0)

    def stop_stream(self):
        self.is_running = False
        if self.thread and self.thread is not threading.current_thread():
            self.thread.join(timeout=1.5)
        if self.cap:
            self.cap.release()
            self.cap = None
        self.video_canvas.delete("all")
        self.current_frame_cv = None
        self.lbl_status.config(text="Stream Terminated", fg=C["warn"])
        self.lbl_risk_level.config(text="Risk Assessment: IDLE", fg=C["muted"])
        self.lbl_confidence_avg.config(text="Confidence Vector: 0.0%")
        self.progress_bar["value"] = 0
        self.lbl_recommendation.config(text="Protocol: Feed offline. Awaiting sequence.")
        self.lbl_fps.config(text="Stream FPS: 0.0 | Latency: 0ms")
        for k in self.prob_bars:
            self.prob_bars[k]["value"] = 0; self.prob_lbls[k].config(text="0%")
        self.log_message("Video pipeline gracefully halted.")

    def render_frame(self, frame):
        self.current_frame_cv = frame
        cw, ch = self.video_canvas.winfo_width(), self.video_canvas.winfo_height()
        if cw < 10 or ch < 10:
            return
        h, w, _ = frame.shape
        sc = min(cw / w, ch / h)
        img = cv2.resize(frame, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
        self.photo_image = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        self.video_canvas.delete("all")
        self.video_canvas.create_image(cw // 2, ch // 2, image=self.photo_image, anchor=tk.CENTER)

    def detect(self, frame):
        res = self.model.predict(frame, conf=0.20, iou=0.6, imgsz=640, agnostic_nms=False, verbose=False)[0]
        out = []
        if res.boxes is not None and len(res.boxes):
            for b, c, k in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(),
                               res.boxes.cls.int().cpu().numpy()):
                raw_name = self.model.names[int(k)]
                out.append({
                    "box": [float(v) for v in b],
                    "conf": float(c),
                    "raw_name": raw_name,
                    "family": canon(raw_name)
                })
        return out

    def process_video(self):
        prev = time.time()
        while self.is_running:
            t0 = time.time()
            ok, frame = self.cap.read()
            if not ok:
                self.root.after(0, lambda: self.log_message("End of video stream sequence reached."))
                break
            tracks, events = self.verifier.update(frame, self.detect(frame))
            annotated = frame.copy()
            for t in tracks:
                draw_track(annotated, t)
            now = time.time()
            fps = 1.0 / max(now - prev, 1e-5); prev = now
            state = self.build_state(tracks, events, fps, int((now - t0) * 1000))
            if not self._ui_busy:  
                self._ui_busy = True
                self.root.after(0, lambda f=annotated, s=state: self.apply_state(f, s))
        self.root.after(0, self.stop_stream)

    def build_state(self, tracks, events, fps, lat):
        s = {"n": len(tracks), "events": events, "fps": fps, "lat": lat,
             "session": dict(self.verifier.session), "fixes": self.verifier.override_count, "t": None}
        if tracks:
            def rank_track(tr):
                lvl, score, _ = analyze_threat_level(tr.family, tr.feats, tr.probs.get(tr.label, 0.0))
                return (score, tr.probs.get(tr.label, 0.0))
            s["t"] = max(tracks, key=rank_track)
            s["family"] = display_name(s["t"])
        return s

    def apply_state(self, frame, s):
        try:
            self.render_frame(frame)
            self.update_prob_bars_ui(self.verifier.known_classes)
            for e in s["events"]:
                self.log_message(e)
            t = s["t"]
            if t is not None:
                family = s["family"]
                risk_label, risk_score, risk_desc = analyze_threat_level(family, t.feats, t.probs.get(t.label, 0.0))
                col = CLASS_HEX.get(family, C["danger"])
                shown_family = "Gun/Knife (uncertain)" if family == "Uncertain" else family
                shown_text = f"Weapon ({shown_family})"

                self.lbl_status.config(text=f"CRITICAL: Weapon Detected", fg=col)
                self.lbl_count.config(text=f"Active Tracked Targets: {s['n']}")
                self.lbl_class.config(text=f"Classification Signature: {shown_text}")
                self.lbl_risk_level.config(text=f"Risk Assessment: {risk_label}", fg=col)
                cv = t.probs.get(t.label, 0.0) * 100
                self.lbl_confidence_avg.config(text=f"Confidence Vector: {cv:.1f}%")
                self.progress_bar["value"] = cv
                self.lbl_recommendation.config(text=f"Protocol: {shown_text} isolated. {risk_desc}")

                for k in self.verifier.known_classes:
                    val = t.probs.get(k, 0.0) * 100
                    if k in self.prob_bars:
                        self.prob_bars[k]["value"] = val
                        self.prob_lbls[k].config(text=f"{val:.0f}%")

                fix = "" if t.label == t.yolo_top else f"  (YOLO: {t.yolo_top} -> sua)"
                self.lbl_verdict.config(text=f"Verdict: {shown_text}{fix}", fg=col)
                f = t.feats
                if f:
                    self.lbl_color.config(text=f"Colour : dark {f['dark']*100:3.0f}% gray {f['gray']*100:3.0f}% "
                                               f"metal {f['metal']*100:3.0f}%")
                    self.lbl_shape.config(text=f"Shape  : aspect {f['aspect']:.1f} rect {f['rect']:.2f} "
                                               f"tip {f['min_end']:.2f} holes {f['holes']} sym {f['symmetry']:.2f}")
                self.lbl_reason.config(text="Logic  : " + (t.reason.split(" | ", 1)[1] if " | " in t.reason
                                                          else "dong thuan YOLO + hinh dang"))
            else:
                self.lbl_status.config(text="Perimeter Secure", fg=C["ok"])
                self.lbl_count.config(text="Active Tracked Targets: 0")
                self.lbl_class.config(text="Classification Signature: None")
                self.lbl_risk_level.config(text="Risk Assessment: SECURE", fg=C["ok"])
                self.lbl_confidence_avg.config(text="Confidence Vector: 0.0%")
                self.progress_bar["value"] = 0
                self.lbl_recommendation.config(text="Protocol: Sector clear. Standard surveillance parameters maintained.")
                for k in self.verifier.known_classes:
                    if k in self.prob_bars:
                        self.prob_bars[k]["value"] = 0; self.prob_lbls[k].config(text="0%")
                self.lbl_verdict.config(text="Verdict: —", fg=C["text"])
            
            ss = s["session"]
            total_threats = sum(ss.values())
            self.lbl_fps.config(text=f"Stream FPS: {s['fps']:.1f} | Latency: {s['lat']}ms")
            self.lbl_session.config(text=f"Session Threats: {total_threats} | Verifier fixes: {s['fixes']}")
        finally:
            self._ui_busy = False

if __name__ == "__main__":
    root = tk.Tk()
    app = WeaponApp(root)
    root.mainloop()