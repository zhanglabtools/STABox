from queue import Queue


class MainQueue(Queue):
    def __init__(self):
        super(MainQueue, self).__init__()


class Message(object):
    def __init__(self, what, obj):
        super(Message, self).__init__()
        self.what = what
        self.obj = obj


class Handler(object):
    def __init__(self, view, handle_msg):
        self.queue = MainQueue()
        self.view = view
        self.handle_msg = handle_msg
        self.loop()

    def loop(self):
        while not self.queue.empty():
            content = self.queue.get()
            self.handle_msg(content)
        self.view.after(30, self.loop)

    def send_msg(self, msg):
        self.queue.put(msg)

    def send_obj(self, what, obj):
        self.queue.put(Message(what, obj))
