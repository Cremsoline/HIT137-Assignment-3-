import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import cv2
import numpy as np
from PIL import Image, ImageTk

#  THEME COLORS
BTN_RED = "#c0392b"
BTN_ORANGE = "#e67e22"
BTN_HOVER = "#ff6f3c"
PANEL_BG = "#2b2b2b"
FRAME_BG = "#3c3c3c"
TEXT_COLOR = "white"


# IMAGE PROCESSOR
class ImageProcessor:
    """
    Handles image processing operations using OpenCV.
    """

    def __init__(self):
        self.current_cv_image = None
        self.original_reference = None

    def load(self, path):
        self.current_cv_image = cv2.imread(path)
        if self.current_cv_image is None:
            raise ValueError("Failed to load image. Please choose a valid image file.")
        self.original_reference = self.current_cv_image.copy()
        return self.current_cv_image

    def grayscale(self):
        gray = cv2.cvtColor(self.current_cv_image, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def blur(self, value):
        k = max(1, int(value) * 2 + 1)
        return cv2.GaussianBlur(self.current_cv_image, (k, k), 0)

    def edges(self):
        gray = cv2.cvtColor(self.current_cv_image, cv2.COLOR_BGR2GRAY)
        edge_map = cv2.Canny(gray, 100, 200)
        return cv2.cvtColor(edge_map, cv2.COLOR_GRAY2BGR)

    def brightness_contrast(self, brightness, contrast):
        return cv2.convertScaleAbs(self.current_cv_image, alpha=float(contrast), beta=int(brightness))

    def rotate(self, angle_code):
        return cv2.rotate(self.current_cv_image, angle_code)

    def flip(self, axis):
        return cv2.flip(self.current_cv_image, axis)

    def resize(self, scale_percent):
        h, w = self.original_reference.shape[:2]
        new_dim = (int(w * scale_percent / 100), int(h * scale_percent / 100))
        return cv2.resize(self.original_reference, new_dim, interpolation=cv2.INTER_AREA)


# IMAGE HISTORY
class ImageHistory:
    def __init__(self):
        self.undo_stack = []
        self.redo_stack = []

    def reset(self):
        self.undo_stack.clear()
        self.redo_stack.clear()

    def save_state(self, img):
        if img is not None:
            self.undo_stack.append(img.copy())
            self.redo_stack.clear()

    def undo(self, current_img):
        if self.undo_stack:
            self.redo_stack.append(current_img.copy())
            return self.undo_stack.pop()
        return current_img

    def redo(self, current_img):
        if self.redo_stack:
            self.undo_stack.append(current_img.copy())
            return self.redo_stack.pop()
        return current_img


# LOADING SCREEN 
class LoadingScreen:
    def __init__(self, parent):
        self.parent = parent
        self.loading_window = None
        self.is_showing = False

    def show(self):
        if self.is_showing:
            return

        self.is_showing = True
        self.loading_window = tk.Toplevel(self.parent)
        self.loading_window.overrideredirect(True)
        self.loading_window.transient(self.parent)

        self.parent.update_idletasks()
        x = self.parent.winfo_x() + (self.parent.winfo_width() - 300) // 2
        y = self.parent.winfo_y() + (self.parent.winfo_height() - 150) // 2

        self.loading_window.geometry(f"300x150+{x}+{y}")
        self.loading_window.configure(bg=PANEL_BG)

        frame = tk.Frame(self.loading_window, bg=PANEL_BG)
        frame.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)

        self.loading_window.update()

    def hide(self):
        if self.loading_window:
            self.is_showing = False
            self.loading_window.destroy()
            self.loading_window = None


# IMAGE EDITOR APP
class ImageEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🔥 OOP Image Editor")
        self.root.geometry("1200x800")
        self.root.configure(bg=PANEL_BG)

        self.processor = ImageProcessor()
        self.history = ImageHistory()
        self.loading = LoadingScreen(self.root)

        self.current_path = None  # for Save
        self.tk_img = None

        self.create_menu()
        self.create_layout()
        self.set_status("Ready")

    #  BUTTON STYLE (READABLE TEXT) 
    def mk_btn(self, parent, text, cmd):
        return tk.Button(
            parent,
            text=text,
            command=cmd,
            bg="#d9d9d9",
            fg="black",
            activebackground="#bfbfbf",
            activeforeground="black",
            font=("Arial", 11, "bold"),
            relief=tk.RAISED,
            bd=1,
            padx=8,
            pady=6
        )

    #  STATUS HELPERS 
    def get_filename(self):
        return self.current_path.split("/")[-1] if self.current_path else "Unsaved"

    def set_status(self, prefix=""):
        """Update status bar text with prefix + filename + dimensions."""
        if self.processor.current_cv_image is None:
            self.status.config(text=prefix if prefix else "Ready")
            return
        h, w = self.processor.current_cv_image.shape[:2]
        self.status.config(text=f"{prefix}{self.get_filename()} | Dimensions: {w}x{h}")

    #  MENU 
    def create_menu(self):
        menubar = tk.Menu(self.root)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open", command=self.open_image)
        file_menu.add_command(label="Save", command=self.save)
        file_menu.add_command(label="Save As", command=self.save_as)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        edit_menu = tk.Menu(menubar, tearoff=0)
        edit_menu.add_command(label="Undo", command=self.undo)
        edit_menu.add_command(label="Redo", command=self.redo)

        menubar.add_cascade(label="File", menu=file_menu)
        menubar.add_cascade(label="Edit", menu=edit_menu)

        self.root.config(menu=menubar)

    #  LAYOUT 
    def create_layout(self):
        #  TOP TOOLBAR 
        toolbar = tk.Frame(self.root, bg=FRAME_BG, relief=tk.RAISED, bd=2)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        self.mk_btn(toolbar, "Open", self.open_image).pack(side=tk.LEFT, padx=6, pady=6)
        self.mk_btn(toolbar, "Save", self.save).pack(side=tk.LEFT, padx=6, pady=6)
        self.mk_btn(toolbar, "Save As", self.save_as).pack(side=tk.LEFT, padx=6, pady=6)
        self.mk_btn(toolbar, "Undo", self.undo).pack(side=tk.LEFT, padx=12, pady=6)
        self.mk_btn(toolbar, "Redo", self.redo).pack(side=tk.LEFT, padx=6, pady=6)
        self.mk_btn(toolbar, "Exit", self.root.quit).pack(side=tk.RIGHT, padx=6, pady=6)

        #  LEFT SIDEBAR 
        sidebar = tk.Frame(self.root, bg=FRAME_BG, width=280)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(sidebar, text="Filters", bg=FRAME_BG, fg=TEXT_COLOR,
                 font=("Arial", 12, "bold")).pack(pady=(15, 10))

        self.mk_btn(sidebar, "Grayscale",
                    lambda: self.process_with_loading(self.processor.grayscale, "Edited: ")
                    ).pack(fill="x", padx=10, pady=4)

        self.mk_btn(sidebar, "Edge Detection",
                    lambda: self.process_with_loading(self.processor.edges, "Edited: ")
                    ).pack(fill="x", padx=10, pady=4)

        # Blur slider + apply
        tk.Label(sidebar, text="Blur Intensity", bg=FRAME_BG, fg=TEXT_COLOR,
                 font=("Arial", 11, "bold")).pack(pady=(12, 4))

        self.blur_slider = tk.Scale(sidebar, from_=0, to=15, orient=tk.HORIZONTAL)
        self.blur_slider.set(0)
        self.blur_slider.pack(fill="x", padx=10)

        self.mk_btn(sidebar, "Apply Blur",
                    lambda: self.process_with_loading(
                        lambda: self.processor.blur(self.blur_slider.get()), "Edited: "
                    )).pack(fill="x", padx=10, pady=6)

        # Brightness / Contrast sliders + apply
        tk.Label(sidebar, text="Brightness", bg=FRAME_BG, fg=TEXT_COLOR,
                 font=("Arial", 11, "bold")).pack(pady=(10, 4))

        self.bright_slider = tk.Scale(sidebar, from_=-100, to=100, orient=tk.HORIZONTAL)
        self.bright_slider.set(0)
        self.bright_slider.pack(fill="x", padx=10)

        tk.Label(sidebar, text="Contrast", bg=FRAME_BG, fg=TEXT_COLOR,
                 font=("Arial", 11, "bold")).pack(pady=(10, 4))

        self.contrast_slider = tk.Scale(sidebar, from_=1.0, to=3.0, resolution=0.1, orient=tk.HORIZONTAL)
        self.contrast_slider.set(1.0)
        self.contrast_slider.pack(fill="x", padx=10)

        self.mk_btn(sidebar, "Apply Adjustments",
                    lambda: self.process_with_loading(
                        lambda: self.processor.brightness_contrast(
                            self.bright_slider.get(), self.contrast_slider.get()
                        ),
                        "Edited: "
                    )).pack(fill="x", padx=10, pady=6)

        # Transforms
        tk.Label(sidebar, text="Transforms", bg=FRAME_BG, fg=TEXT_COLOR,
                 font=("Arial", 12, "bold")).pack(pady=(16, 10))

        self.mk_btn(sidebar, "Rotate 90°",
                    lambda: self.process_with_loading(
                        lambda: self.processor.rotate(cv2.ROTATE_90_CLOCKWISE), "Edited: "
                    )).pack(fill="x", padx=10, pady=3)

        self.mk_btn(sidebar, "Rotate 180°",
                    lambda: self.process_with_loading(
                        lambda: self.processor.rotate(cv2.ROTATE_180), "Edited: "
                    )).pack(fill="x", padx=10, pady=3)

        self.mk_btn(sidebar, "Rotate 270°",
                    lambda: self.process_with_loading(
                        lambda: self.processor.rotate(cv2.ROTATE_90_COUNTERCLOCKWISE), "Edited: "
                    )).pack(fill="x", padx=10, pady=3)

        self.mk_btn(sidebar, "Flip Horizontal",
                    lambda: self.process_with_loading(lambda: self.processor.flip(1), "Edited: ")
                    ).pack(fill="x", padx=10, pady=3)

        self.mk_btn(sidebar, "Flip Vertical",
                    lambda: self.process_with_loading(lambda: self.processor.flip(0), "Edited: ")
                    ).pack(fill="x", padx=10, pady=3)

        self.mk_btn(sidebar, "Resize Image", self.resize_popup).pack(fill="x", padx=10, pady=(10, 12))

        # DISPLAY AREA 
        display_frame = tk.Frame(self.root, bg=PANEL_BG)
        display_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        self.display_area = tk.Label(display_frame, bg=PANEL_BG)
        self.display_area.pack(expand=True)

        # STATUS BAR 
        self.status = tk.Label(self.root, text="Ready", bg="#1f1f1f", fg="white", anchor=tk.W)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

    # IMAGE DISPLAY
    def display(self, img):
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        pil_img.thumbnail((1100, 650))
        self.tk_img = ImageTk.PhotoImage(pil_img)
        self.display_area.config(image=self.tk_img)

    # PROCESSING
    def process_with_loading(self, operation, status_prefix="Edited: "):
        if self.processor.current_cv_image is None:
            messagebox.showerror("No image", "Please open an image first.")
            return

        self.loading.show()
        try:
            new_img = operation()
            self.history.save_state(self.processor.current_cv_image)
            self.processor.current_cv_image = new_img
            self.display(new_img)
            self.set_status(status_prefix)
        finally:
            self.loading.hide()

    # FILE OPERATIONS
    def open_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg *.bmp")])
        if not path:
            return

        self.loading.show()
        try:
            img = self.processor.load(path)
            self.current_path = path
            self.history.reset()
            self.display(img)
            self.set_status("Loaded: ")
        except Exception as e:
            messagebox.showerror("Open failed", str(e))
        finally:
            self.loading.hide()

    def save(self):
        if self.processor.current_cv_image is None:
            messagebox.showerror("No image", "Please open an image first.")
            return

        if not self.current_path:
            self.save_as()
            return

        cv2.imwrite(self.current_path, self.processor.current_cv_image)
        messagebox.showinfo("Saved", "Image saved successfully!")
        self.set_status("Saved: ")

    def save_as(self):
        if self.processor.current_cv_image is not None:
            path = filedialog.asksaveasfilename(defaultextension=".jpg")
            if path:
                cv2.imwrite(path, self.processor.current_cv_image)
                self.current_path = path
                messagebox.showinfo("Saved", "Image saved successfully!")
                self.set_status("Saved: ")

    def resize_popup(self):
        if self.processor.current_cv_image is None:
            messagebox.showerror("No image", "Please open an image first.")
            return
        percent = simpledialog.askinteger("Resize", "Enter scale percentage (50-200):", initialvalue=100)
        if percent:
            self.process_with_loading(lambda: self.processor.resize(percent), "Edited: ")

    # EDIT OPERATIONS
    def undo(self):
        if self.processor.current_cv_image is None:
            return
        self.loading.show()
        try:
            self.processor.current_cv_image = self.history.undo(self.processor.current_cv_image)
            self.display(self.processor.current_cv_image)
            self.set_status("Undo: ")
        finally:
            self.loading.hide()

    def redo(self):
        if self.processor.current_cv_image is None:
            return
        self.loading.show()
        try:
            self.processor.current_cv_image = self.history.redo(self.processor.current_cv_image)
            self.display(self.processor.current_cv_image)
            self.set_status("Redo: ")
        finally:
            self.loading.hide()


# RUN APP
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageEditorApp(root)
    root.mainloop()
