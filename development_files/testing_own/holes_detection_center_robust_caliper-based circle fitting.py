"""
Center Hole Inspector - side-by-side Raw / Center-marked viewer.

Uses hole_center_robust.py (keep both files in the same folder).

pip install opencv-python pillow numpy
python hole_center_ui.py

Keys:  Left/Right = previous/next image (when a folder is open),  Ctrl+O = open image
"""

import os
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

from hole_center_robust import find_hole_center

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# quality limits for OK / CHECK status
MAX_RMS = 2.0          # px
MIN_INLIERS = 0.30     # fraction


def draw_marked(img_bgr, res, show_points=True):
    out = img_bgr.copy()
    H, W = out.shape[:2]
    t = max(2, int(round(min(H, W) / 350)))
    if res is None:
        cv2.putText(out, "NO HOLE FOUND", (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.7, W / 700), (0, 0, 255), t, cv2.LINE_AA)
        return out

    if show_points:
        for (x, y), ok in zip(res["points"], res["inliers"]):
            cv2.circle(out, (int(round(x)), int(round(y))), t,
                       (0, 200, 0) if ok else (0, 140, 255), -1)

    cx, cy = res["center"]
    r = res["radius"]
    c = (int(round(cx)), int(round(cy)))
    cv2.circle(out, c, int(round(r)), (0, 0, 255), t)
    cv2.drawMarker(out, c, (255, 0, 255), cv2.MARKER_CROSS, int(r * 0.6) + 10, t)
    return out


def to_photo(img_bgr, max_w, max_h):
    h, w = img_bgr.shape[:2]
    s = min(max_w / w, max_h / h)
    nw, nh = max(1, int(w * s)), max(1, int(h * s))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    rgb = cv2.cvtColor(cv2.resize(img_bgr, (nw, nh), interpolation=interp), cv2.COLOR_BGR2RGB)
    return ImageTk.PhotoImage(Image.fromarray(rgb))


class App:
    def __init__(self, root):
        self.root = root
        root.title("Center Hole Inspector")
        root.geometry("1400x820")
        root.configure(bg="#1e1f22")

        self.files, self.idx = [], -1
        self.img = self.marked = self.res = None
        self.photos = [None, None]
        self.show_points = tk.BooleanVar(value=True)

        # ---------- toolbar ----------
        bar = tk.Frame(root, bg="#2b2d31", pady=6, padx=8)
        bar.pack(side=tk.TOP, fill=tk.X)
        btn = dict(bg="#3a3d44", fg="white", activebackground="#4a4e57",
                   activeforeground="white", relief=tk.FLAT, padx=12, pady=4)
        tk.Button(bar, text="Open Image", command=self.open_image, **btn).pack(side=tk.LEFT, padx=3)
        tk.Button(bar, text="Open Folder", command=self.open_folder, **btn).pack(side=tk.LEFT, padx=3)
        tk.Button(bar, text="◀ Prev", command=lambda: self.step(-1), **btn).pack(side=tk.LEFT, padx=(15, 3))
        tk.Button(bar, text="Next ▶", command=lambda: self.step(1), **btn).pack(side=tk.LEFT, padx=3)
        tk.Button(bar, text="Save Marked", command=self.save, **btn).pack(side=tk.LEFT, padx=(15, 3))
        tk.Checkbutton(bar, text="Show edge points", variable=self.show_points,
                       command=self.redraw_marked, bg="#2b2d31", fg="white",
                       selectcolor="#2b2d31", activebackground="#2b2d31",
                       activeforeground="white").pack(side=tk.LEFT, padx=15)
        self.file_lbl = tk.Label(bar, text="", bg="#2b2d31", fg="#aaa")
        self.file_lbl.pack(side=tk.RIGHT, padx=5)

        # ---------- result bar ----------
        self.status = tk.Label(root, text="Open an image to start", anchor="w",
                               bg="#1e1f22", fg="#ddd", font=("Consolas", 12), padx=12, pady=6)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

        # ---------- two panels ----------
        body = tk.Frame(root, bg="#1e1f22")
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        body.columnconfigure(0, weight=1, uniform="p")
        body.columnconfigure(1, weight=1, uniform="p")
        body.rowconfigure(1, weight=1)

        self.canvases = []
        for col, title in enumerate(("RAW IMAGE", "CENTER MARKED")):
            tk.Label(body, text=title, bg="#1e1f22", fg="#9aa0a6",
                     font=("Segoe UI", 10, "bold")).grid(row=0, column=col, pady=(0, 4))
            cv = tk.Canvas(body, bg="#111", highlightthickness=1, highlightbackground="#333")
            cv.grid(row=1, column=col, sticky="nsew", padx=4)
            cv.bind("<Configure>", lambda e: self.render())
            self.canvases.append(cv)

        root.bind("<Left>", lambda e: self.step(-1))
        root.bind("<Right>", lambda e: self.step(1))
        root.bind("<Control-o>", lambda e: self.open_image())

    # ---------- file handling ----------
    def open_image(self):
        p = filedialog.askopenfilename(filetypes=[("Images", " ".join("*" + e for e in IMG_EXT)),
                                                  ("All", "*.*")])
        if p:
            folder = os.path.dirname(p)
            self.files = sorted(os.path.join(folder, f) for f in os.listdir(folder)
                                if f.lower().endswith(IMG_EXT))
            self.idx = self.files.index(p) if p in self.files else 0
            if p not in self.files:
                self.files = [p]
            self.load()

    def open_folder(self):
        d = filedialog.askdirectory()
        if d:
            self.files = sorted(os.path.join(d, f) for f in os.listdir(d)
                                if f.lower().endswith(IMG_EXT))
            if not self.files:
                messagebox.showinfo("Folder", "No images found in this folder.")
                return
            self.idx = 0
            self.load()

    def step(self, d):
        if self.files:
            self.idx = (self.idx + d) % len(self.files)
            self.load()

    def load(self):
        path = self.files[self.idx]
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            messagebox.showerror("Error", f"Could not read:\n{path}")
            return
        self.img = img
        self.file_lbl.config(text=f"{os.path.basename(path)}   [{self.idx + 1}/{len(self.files)}]")
        self.root.update_idletasks()

        try:
            self.res = find_hole_center(img)
        except Exception as e:          # never crash the viewer on a bad image
            self.res = None
            print("Detection error:", e)
        self.redraw_marked()
        self.update_status()

    # ---------- drawing ----------
    def redraw_marked(self):
        if self.img is not None:
            self.marked = draw_marked(self.img, self.res, self.show_points.get())
            self.render()

    def render(self):
        if self.img is None:
            return
        for i, (cv, im) in enumerate(zip(self.canvases, (self.img, self.marked))):
            if im is None:
                continue
            w, h = max(cv.winfo_width(), 10), max(cv.winfo_height(), 10)
            self.photos[i] = to_photo(im, w - 4, h - 4)
            cv.delete("all")
            cv.create_image(w // 2, h // 2, image=self.photos[i], anchor="center")

    def update_status(self):
        r = self.res
        if r is None:
            self.status.config(text="  ✖  NO HOLE FOUND", fg="#ff6b6b")
            return
        ok = r["rms"] <= MAX_RMS and r["inlier_ratio"] >= MIN_INLIERS
        cx, cy = r["center"]
        self.status.config(
            fg="#5fd068" if ok else "#ffb347",
            text=(f"  {'✔ OK' if ok else '⚠ CHECK'}    "
                  f"Center X: {cx:.2f}   Center Y: {cy:.2f}   "
                  f"Radius: {r['radius']:.2f} px   "
                  f"Fit RMS: {r['rms']:.2f} px   Inliers: {r['inlier_ratio']:.0%}"))

    def save(self):
        if self.marked is None:
            messagebox.showinfo("Save", "Nothing to save yet.")
            return
        base = os.path.splitext(os.path.basename(self.files[self.idx]))[0]
        p = filedialog.asksaveasfilename(defaultextension=".png", initialfile=f"{base}_marked.png",
                                         filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if p:
            ok, buf = cv2.imencode(os.path.splitext(p)[1] or ".png", self.marked)
            if ok:
                buf.tofile(p)


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
