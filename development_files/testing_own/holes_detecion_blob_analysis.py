"""
Center hole detector with a simple UI.

Requirements:
    pip install opencv-python pillow numpy

Run:
    python hole_detector_ui.py
"""

import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk


# ----------------------------------------------------------------------
# Detection
# ----------------------------------------------------------------------
def detect_hole(img_bgr, thresh=40, min_area=500, min_circ=0.8, blur=7):
    """Find the darkest, most circular blob. Returns dict or None, plus the mask."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = blur if blur % 2 == 1 else blur + 1
    gray = cv2.GaussianBlur(gray, (blur, blur), 0)

    _, bw = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
    k = np.ones((5, 5), np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k)

    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    best, best_area, best_circ = None, 0, 0
    for c in cnts:
        area = cv2.contourArea(c)
        peri = cv2.arcLength(c, True)
        if area < min_area or peri == 0:
            continue
        circ = 4 * np.pi * area / (peri * peri)
        if circ >= min_circ and area > best_area:
            best, best_area, best_circ = c, area, circ

    if best is None:
        return None, bw

    # Center: ellipse fit if enough points, else moments
    if len(best) >= 5:
        (cx, cy), (ma, MA), _ = cv2.fitEllipse(best)
        r = (ma + MA) / 4.0
    else:
        M = cv2.moments(best)
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
        (_, _), r = cv2.minEnclosingCircle(best)

    return {
        "center": (cx, cy),
        "radius": r,
        "area": best_area,
        "circularity": best_circ,
        "contour": best,
    }, bw


def draw_result(img_bgr, res):
    out = img_bgr.copy()
    if res is None:
        return out
    cx, cy = res["center"]
    r = res["radius"]
    t = max(2, int(round(min(out.shape[:2]) / 300)))
    cv2.drawContours(out, [res["contour"]], -1, (0, 255, 0), t)
    cv2.circle(out, (int(round(cx)), int(round(cy))), int(round(r)), (0, 0, 255), t)
    cv2.drawMarker(out, (int(round(cx)), int(round(cy))), (0, 0, 255),
                   cv2.MARKER_CROSS, int(r * 0.6) + 10, t)
    label = f"({cx:.1f}, {cy:.1f}) r={r:.1f}"
    fs = max(0.4, min(out.shape[:2]) / 700)
    ft = max(1, t - 1)
    (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, ft)
    H, W = out.shape[:2]
    # put label below the circle, clamped inside the image
    x = int(min(max(cx - tw / 2, 4), W - tw - 4))
    y = int(cy + r + th + 10)
    if y > H - 4:
        y = int(cy - r - 10)
    y = int(min(max(y, th + 4), H - 4))
    cv2.rectangle(out, (x - 3, y - th - 4), (x + tw + 3, y + base + 2), (255, 255, 255), -1)
    cv2.putText(out, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 255), ft, cv2.LINE_AA)
    return out


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
class App:
    def __init__(self, root):
        self.root = root
        root.title("Center Hole Detector")
        root.geometry("1100x750")

        self.img = None        # original BGR
        self.result_img = None # annotated BGR
        self.mask = None
        self.res = None
        self.tk_img = None

        # --- left: controls ---
        ctrl = tk.Frame(root, padx=10, pady=10)
        ctrl.pack(side=tk.LEFT, fill=tk.Y)

        tk.Button(ctrl, text="Open Image", width=18, command=self.open_image).pack(pady=4)
        tk.Button(ctrl, text="Detect", width=18, command=self.detect).pack(pady=4)
        tk.Button(ctrl, text="Save Result", width=18, command=self.save).pack(pady=4)

        tk.Label(ctrl, text="").pack()
        self.thresh = self._slider(ctrl, "Threshold (dark level)", 0, 255, 40)
        self.min_area = self._slider(ctrl, "Min area (px)", 50, 50000, 500, res=50)
        self.min_circ = self._slider(ctrl, "Min circularity", 0.3, 1.0, 0.8, res=0.01)
        self.blur = self._slider(ctrl, "Blur kernel", 1, 31, 7, res=2)

        self.auto = tk.BooleanVar(value=True)
        tk.Checkbutton(ctrl, text="Live update", variable=self.auto).pack(anchor="w", pady=4)

        self.view = tk.StringVar(value="result")
        tk.Label(ctrl, text="View:").pack(anchor="w", pady=(10, 0))
        for txt, val in [("Original", "original"), ("Result", "result"), ("Mask", "mask")]:
            tk.Radiobutton(ctrl, text=txt, variable=self.view, value=val,
                           command=self.refresh).pack(anchor="w")

        self.info = tk.Label(ctrl, text="No image loaded", justify=tk.LEFT,
                             anchor="w", font=("Consolas", 10), fg="#333")
        self.info.pack(anchor="w", pady=15, fill=tk.X)

        # --- right: image canvas ---
        self.canvas = tk.Canvas(root, bg="#222", highlightthickness=0)
        self.canvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda e: self.refresh())

    def _slider(self, parent, text, lo, hi, val, res=1):
        tk.Label(parent, text=text).pack(anchor="w")
        s = tk.Scale(parent, from_=lo, to=hi, orient=tk.HORIZONTAL, resolution=res,
                     length=200, command=lambda _v: self._on_change())
        s.set(val)
        s.pack(anchor="w")
        return s

    def _on_change(self):
        if self.img is not None and self.auto.get():
            self.detect()

    def open_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("All", "*.*")])
        if not path:
            return
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            messagebox.showerror("Error", "Could not read image.")
            return
        self.img = img
        self.root.title(f"Center Hole Detector - {path}")
        self.detect()

    def detect(self):
        if self.img is None:
            return
        self.res, self.mask = detect_hole(
            self.img,
            thresh=int(self.thresh.get()),
            min_area=int(self.min_area.get()),
            min_circ=float(self.min_circ.get()),
            blur=int(self.blur.get()),
        )
        self.result_img = draw_result(self.img, self.res)

        if self.res:
            cx, cy = self.res["center"]
            self.info.config(fg="#070", text=(
                f"HOLE FOUND\n"
                f"Center X : {cx:.2f}\n"
                f"Center Y : {cy:.2f}\n"
                f"Radius   : {self.res['radius']:.2f}\n"
                f"Area     : {self.res['area']:.0f}\n"
                f"Circular.: {self.res['circularity']:.3f}"))
        else:
            self.info.config(fg="#b00", text="No hole found.\nTry adjusting threshold\nor min circularity.")
        self.refresh()

    def refresh(self):
        if self.img is None:
            return
        v = self.view.get()
        if v == "original":
            disp = self.img
        elif v == "mask" and self.mask is not None:
            disp = cv2.cvtColor(self.mask, cv2.COLOR_GRAY2BGR)
        else:
            disp = self.result_img if self.result_img is not None else self.img

        cw = max(self.canvas.winfo_width(), 1)
        ch = max(self.canvas.winfo_height(), 1)
        h, w = disp.shape[:2]
        scale = min(cw / w, ch / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_NEAREST
        rgb = cv2.cvtColor(cv2.resize(disp, (nw, nh), interpolation=interp), cv2.COLOR_BGR2RGB)

        self.tk_img = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, image=self.tk_img, anchor="center")

    def save(self):
        if self.result_img is None:
            messagebox.showinfo("Save", "Nothing to save yet.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if path:
            ext = "." + path.rsplit(".", 1)[-1]
            ok, buf = cv2.imencode(ext, self.result_img)
            if ok:
                buf.tofile(path)


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
