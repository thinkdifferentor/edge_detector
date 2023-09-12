from PIL import Image, ImageTk
from tkinter import Label, Tk, TOP
import cv2
import sys
import math
import os
import json
import screeninfo


class Edges:
    class Operator(object):
        def __init__(self, img_path, thresh):
            self.img_path = img_path
            self.img = Image.open(self.img_path).convert("L")
            self.img_dat = self.img.load()
            self.w = self.img.width
            self.h = self.img.height
            self.max_x = self.w - 1
            self.max_y = self.h - 1
            self.thresh = thresh

        def on_max_exceeded(self, x, y):
            is_exceeded = x == self.max_x or y == self.max_y
            if x == self.max_x or y == self.max_y:
                self.img_dat[x, y] = 0

            return is_exceeded

        def gradient(self, x, y):
            return math.sqrt(pow(x, 2) + pow(y, 2))

        def threshold_cutoff(self, grad_x, grad_y):
            grad_mod = self.gradient(grad_x, grad_y)
            if grad_mod > self.thresh:
                return 255
            else:
                return 0

        def get_thresh(self):
            return self.thresh

            
    class Convolution(Operator):
        def __init__(self, kern_x, kern_y, *args):
            super(Edges.Convolution, self).__init__(*args)
            self.kern_x = kern_x
            self.kern_y = kern_y

            
        def apply_gradient(self, grad_x, grad_y):
            for x in range(0, self.w):
                for y in range(0, self.h):
                    pixel = self.threshold_cutoff(grad_x[x][y], grad_y[x][y])
                    self.img_dat[x, y] = pixel

                    
        def apply_kernel(self, x, y, kern):
            dim = len(kern) // 2
            total = 0
            for i in range(-dim, dim + 1):
                for j in range(-dim, dim + 1):
                    tempX = x + i
                    tempY = y + j
                    kernX = dim + i
                    kernY = dim + j
                    if tempX >= 0 and tempX < self.w and tempY >= 0 and tempY < self.h:
                        total += self.img_dat[tempX, tempY] * kern[kernX][kernY]

            return total

            
        def __call__(self):
            grad_x = [[0 for y in range(0, self.h)] for x in range(0, self.w)]
            grad_y = [[0 for y in range(0, self.h)] for x in range(0, self.w)]
            for x in range(0, self.w):
                for y in range(0, self.h):
                    grad_x[x][y] = self.apply_kernel(x, y, self.kern_x)

            for y in range(0, self.h):
                for x in range(0, self.w):
                    grad_y[x][y] = self.apply_kernel(x, y, self.kern_y)

            self.apply_gradient(grad_x, grad_y)

            return self.img
            
            
    class Roberts(Operator):
        def __init__(self, *args):
            super(Edges.Roberts, self).__init__(*args)

            
        def __call__(self):
            for x in range(0, self.w):
                for y in range(0, self.h):
                    if self.on_max_exceeded(x, y):
                        continue

                    grad_x = self.img_dat[x, y] - self.img_dat[x + 1, y + 1]
                    grad_y = self.img_dat[x + 1, y] - self.img_dat[x, y + 1]

                    self.img_dat[x, y] = self.threshold_cutoff(grad_x, grad_y)

            return self.img

                    
    class Prewitt(Convolution):
        def __init__(self, *args):
            kern_x = [
                [-1, 0, +1],
                [-1, 0, +1],
                [-1, 0, +1]
            ]
            kern_y = [
                [-1, -1, -1],
                [0, 0, 0],
                [1, 1, 1]
            ]
            super(Edges.Prewitt, self).__init__(kern_x, kern_y, *args)


    class Sobel(Convolution):
        def __init__(self, *args):
            kern_x = [
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]
            ]
            kern_y = [
                [-1, -2, -1],
                [0, 0, 0],
                [1, 2, 1]
            ]
            super(Edges.Sobel, self).__init__(kern_x, kern_y, *args)


    class Scharr(Convolution):
        def __init__(self, *args):
            kern_x = [
                [-3, 0, 3],
                [-10, 0, 10],
                [-3, 0, 3]
            ]
            kern_y = [
                [-3, -10, -3],
                [0, 0, 0],
                [3, 10, 3]
            ]
            super(Edges.Scharr, self).__init__(kern_x, kern_y, *args)

            
    class Laplace(Convolution):
        class Gauss:
            def gauss_2d(self, x, y, sig):
                sig = (2 * pow(sig, 2))
                return pow(math.e, -(pow(x, 2) + pow(y, 2)) / sig) / (math.pi * sig)

            def normalize(self, kernel):
                total = 0
                for row in kernel:
                    for val in row:
                        total += val
                        
                dim = len(kernel)
                for x in range(dim):
                    for y in range(dim):
                        kdat = kernel[x][y]
                        kernel[x][y] = kdat / total

            def make_kernel(self, dim, sig):
                kern = list()
                r1 = dim // 2
                r2 = r1 + 1
                r1 = -r1
                for x in range(r1, r2):
                    row = list()
                    for y in range(r1, r2):
                        row.append(self.gauss_2d(x, y, sig))
                    kern.append(row)

                self.normalize(kern)
                return kern


        def __init__(self, *args):
            self.kern = [
                [0, 1, 0],
                [1, -4, 1],
                [0, 1, 0]
            ]
            super(Edges.Laplace, self).__init__(None, None, *args)


        def is_edge(self, a, b):
            if (a * b) < 0 and abs(a - b) > self.thresh:
                    return True
            else:
                return False

            
        def __call__(self):
            gauss = self.Gauss()
            gauss_kern = gauss.make_kernel(3, 1)
            laplace = [[0 for x in range(0, self.h)] for x in range(0, self.w)]

            for x in range(0, self.w):
                for y in range(0, self.h):
                    self.img_dat[x, y] = int(self.apply_kernel(x, y, gauss_kern))

            lapmax = 0
            for x in range(0, self.w):
                for y in range(0, self.h):
                    lap = self.apply_kernel(x, y, self.kern)
                    laplace[x][y] = lap
                    lap_abs = abs(lap)
                    lapmax = lap_abs if lap_abs > lapmax else lapmax

            sign_changed = 0
            for x in range(1, self.max_x):
                for y in range(1, self.max_y):
                    if self.is_edge(laplace[x - 1][y], laplace[x + 1][y]):
                        sign_changed += 1
                    if self.is_edge(laplace[x][y + 1], laplace[x][y - 1]):
                        sign_changed += 1
                    if self.is_edge(laplace[x + 1][y + 1], laplace[x - 1][y - 1]):
                        sign_changed += 1
                    if self.is_edge(laplace[x + 1][y - 1], laplace[x - 1][y + 1]):
                        sign_changed += 1

                    if sign_changed >= 2:
                        self.img_dat[x, y] = 255
                    else:
                        self.img_dat[x, y] = 0

                    sign_changed = 0

            return self.img

    class Canny:
        def __init__(self, img_path, thresh):
            self.thresh_low, self.thresh_high = thresh
            self.img_path = img_path

        def get_thresh(self):
            return (self.thresh_low, self.thresh_high)

        def __call__(self):
            cv_image = cv2.imread(self.img_path, 0)
            canny_img = cv2.Canny(cv_image, self.thresh_low, self.thresh_high)
            return Image.fromarray(canny_img)
            
            

            
    def __init__(self):
        self.img_path = None
        self.img = None
        self.img_dat = None
        self.out_img = None
        self.orig_img = None
        self.tk_labels = []

        
    def apply_to_image(self, img_path, operator_type, *args):
        self.img_path = img_path
        self.out_img = os.path.splitext(self.img_path)[0] + "_edged" + os.path.splitext(self.img_path)[1]
        self.op_name = operator_type.__name__
        operator = operator_type(self.img_path, *args)
        processed_img = operator()
        print("Calculated edges for {} operator".format(operator_type.__name__))
        return processed_img, "{}, thresh={}".format(operator_type.__name__, operator.get_thresh())

        
    def save(self):
        self.img.save(self.out_img, "jpeg")

        
    def __enter__(self):
        return self

    
    def __exit__(self, e_type, e_val, traceback):
        self.save()



class GUIGrid:
    def __init__(self, rows = 2, columns = 3, initial_scale = (0.9, 0.8)):
        self.tk_root = Tk()
        self.max_w, self.max_h = self._get_window_wh(initial_scale)
        self.tk_root.geometry("{}x{}".format(self.max_w, self.max_h))
        self.tk_images = []
        self.rows = rows
        self.columns = columns
        self.cols_nb = 0


    def _get_window_wh(self, scale):
        scale_w, scale_h = scale
        monitors = screeninfo.get_monitors()
        x = self.tk_root.winfo_x()
        y = self.tk_root.winfo_y()
        monitor = None
        for m in reversed(monitors):
            if m.x <= x <= m.width + m.x and m.y <= y <= m.height + m.y:
                monitor = m
                break
        monitor = monitors[0] if monitor is None else monitor
        return int(monitor.width * scale_w), int(monitor.height * scale_h)


    def add_image(self, img, text = None, rowspan = 1, colspan = 1, height_fix = 45):
        height_fix = 0 if text is None else height_fix
        img = img.resize((int(self.max_w / self.columns) * colspan,
                          int(self.max_h / self.rows) * rowspan - height_fix))
        img_tk = ImageTk.PhotoImage(img)
        label = Label(self.tk_root, text=text,
                      image=img_tk, compound='bottom', font=("Arial", 25), bg="#000", fg='#fff')
        label.grid(row=int(self.cols_nb / self.columns + 1), column=(self.cols_nb % self.columns) + 1,
                   rowspan=rowspan, columnspan=colspan)
        self.tk_images.append((img_tk, label))
        self.cols_nb += colspan


    def render_loop(self):
        self.tk_root.mainloop()
        

if __name__ == "__main__":
    edges = Edges()
    gui = GUIGrid(2, 4)
    ops1 = [ (Edges.Canny, (50,100)),
             (Edges.Roberts, 20),
             (Edges.Prewitt, 60) ]
    ops2 = [ (Edges.Sobel, 70),
             (Edges.Scharr, 300),
             (Edges.Laplace, 12) ]

    orig_img = Image.open(sys.argv[1])
    gui.add_image(orig_img, None)
    for op, thresh in ops1:
        img, text = edges.apply_to_image(sys.argv[1], op, thresh)
        gui.add_image(img, text)

    gui.add_image(Image.new('RGB', (1, 1)))
    for op, thresh in ops2:
        img, text = edges.apply_to_image(sys.argv[1], op, thresh)
        gui.add_image(img, text) 


    gui.render_loop()
