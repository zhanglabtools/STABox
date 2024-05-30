from tkinter import Label, Canvas, StringVar, Frame


class ProgressBar(Frame):
    def __init__(self, parent, width=300, height=30, border_width=2):
        super(ProgressBar, self).__init__(parent)
        self.width = width
        self.height = height
        self.border_width = border_width
        self.canvas = Canvas(self, width=self.width, height=self.height, bg="white")
        self.progress_text = StringVar()

        self.x1 = self.border_width + 1
        self.y1 = self.border_width + 1
        self.x2 = self.width - 1
        self.y2 = self.height - 1
        self.x11 = self.x1 + self.border_width / 2
        self.y11 = self.y1 + self.border_width / 2
        self.progress_width = self.width - self.x11 - self.border_width / 2 - 1
        print(self.x1, self.y1, self.x2, self.y2)
        self.init()

    def init(self):
        self.canvas.grid(row=0, column=0)
        # 进度条背景框
        self.canvas.create_rectangle(self.x1, self.y1, self.x2, self.y2, outline="blue", width=self.border_width)
        # 进度条进度
        self.fill_rec = self.canvas.create_rectangle(self.x11, self.y11, self.x11, self.y2 - self.border_width / 2,
                                                     outline="", width=0, fill="blue")

        Label(self, textvariable=self.progress_text, width=5).grid(row=0, column=1)

    def progress(self, current, total):
        self.canvas.coords(self.fill_rec,
                           (self.x11, self.y11,
                            self.x11 + (current / total) * self.progress_width,
                            self.y2 - self.border_width / 2))
        f = round(current / total * 100, 2)
        self.progress_text.set(str(f) + '%')
        self.update()
        if f == 100.00:
            self.progress_text.set("完成")
