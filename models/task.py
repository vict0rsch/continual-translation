from collections import OrderedDict


class BaseTask:
    def __init__(self):
        super().__init__()
        self.key = ""
        self.loss_names = []
        self.needs_D = False
        self.needs_lr = False
        self.needs_z = False
        self.module_name = ""
        self.lambda_key = ""
        self.loss_function = ""
        self.priority = 0
        self.metrics_key = ""
        self.threshold_key = ""
        self.no_G = False
        self.target_key = None
        self.has_target = False
        self.eval_visuals_pred = False
        self.eval_visuals_target = False
        self.eval_acc = False
        self.log_type = "acc"
        self.output_dim = 0
        self.loader_resize_target = True
        self.loader_resize_input = True
        self.loader_flip = True
        self.input_key = ""

    def setup(self):

        assert self.key
        assert self.key not in {"idt", "z", "fake", "rec"}
        assert self.lambda_key not in {"idt", "A", "B"}
        assert self.threshold_type in {"acc", "loss"}
        assert self.log_type in {"acc", "vis"}

        if not self.module_name:
            self.module_name = self.key

        if self.needs_lr and not self.module_name:
            raise ValueError("needs_lr and module_name")

        if not self.no_G:
            self.loss_names.append(f"G_A_{self.key}")
            self.loss_names.append(f"G_B_{self.key}")
        if self.needs_D:
            self.loss_names.append(f"D_A_{self.key}")
            self.loss_names.append(f"D_B_{self.key}")

        if self.has_target and self.target_key is None:
            self.target_key = self.key + "_target"

        if self.log_type == "acc":
            assert self.output_dim > 0

        self.threshold_key = f"{self.key}_{self.threshold_type}_threshold"

        if not self.input_key:
            self.input_key = self.key

    def __str__(self):
        s = self.__class__.__name__ + ":\n"
        for d in dir(self):
            if not d.startswith("__"):
                attr = getattr(self, d)
                if not callable(attr):
                    s += "   {:15}: {}\n".format(d, attr)
        return s


class GrayTask(BaseTask):
    def __init__(self):
        super().__init__()
        self.key = "gray"
        self.needs_D = True
        self.needs_lr = True
        self.has_target = True
        self.target_key = "real"
        self.threshold_type = "acc"
        self.priority = 2
        self.needs_z = True
        self.lambda_key = "G"
        self.eval_visuals_pred = True
        self.eval_visuals_target = False
        self.eval_acc = False
        self.log_type = "vis"


class RotationTask(BaseTask):
    def __init__(self):
        super().__init__()
        self.key = "rotation"
        self.has_target = True
        self.threshold_type = "loss"
        self.priority = 0
        self.needs_z = True
        self.lambda_key = "R"
        self.eval_visuals_pred = False
        self.eval_visuals_target = False
        self.eval_acc = True
        self.log_type = "acc"
        self.loader_resize_target = False
        self.loader_resize_input = True
        self.output_dim = 4
        self.loader_flip = False


class DepthTask(BaseTask):
    def __init__(self):
        super().__init__()
        self.key = "depth"
        self.needs_lr = True
        self.has_target = True
        self.threshold_type = "loss"
        self.priority = 1
        self.needs_z = False
        self.lambda_key = "D"
        self.eval_visuals_pred = True
        self.eval_visuals_target = True
        self.eval_acc = False
        self.log_type = "vis"
        self.input_key = "real"


class AuxiliaryTasks:
    def __init__(self, keys=[]):
        super().__init__()
        self._index = 0
        tasks = []
        for k in keys:
            if k == "gray":
                tasks += [(k, GrayTask())]
            elif k == "rotation":
                tasks += [(k, RotationTask())]
            elif k == "depth":
                tasks += [(k, DepthTask())]
            else:
                raise ValueError("Unknown Auxiliary task {}".format(k))
        tasks = sorted(tasks, key=lambda x: x[1].priority)
        self.tasks = OrderedDict(tasks)
        self.keys = list(self.tasks.keys())

        for t in self.tasks:
            self.tasks[t].setup()

    def task_before(self, k):
        if k not in self.tasks:
            return None
        index = self.keys.index(k)
        if index == 0:
            return None
        return self.keys[index - 1]

    def task_after(self, k):
        if k not in self.tasks:
            return None
        index = self.keys.index(k)
        if index >= len(self.tasks) - 1:
            return None
        return self.keys[index + 1]

    def __str__(self):
        return "AuxiliaryTasks: " + str(self.tasks)

    def __iter__(self):
        return self

    def __next__(self):
        if self._index == len(self.keys):
            self._index = 0
            raise StopIteration
        t = self.tasks[self.keys[self._index]]
        self._index += 1
        return t

    def __getitem__(self, k):
        if isinstance(k, int):
            return self.tasks[self.keys[k]]
        return self.tasks[k]


class T:
    def __init__(self, ts):
        self.tasks = OrderedDict([(t, t + 10) for t in ts])
        self._index = 0
        self._keys = list(self.tasks.keys())

    def task_before(self, k):
        if k not in self.tasks:
            return None
        keys = list(self.tasks.keys())
        index = keys.index(k)
        if index == 0:
            return None
        return keys[index - 1]

    def task_after(self, k):
        if k not in self.tasks:
            return None

        keys = list(self.tasks.keys())
        index = keys.index(k)
        if index >= len(self.tasks) - 1:
            return None
        return keys[index + 1]

    def __str__(self):
        return "AuxiliaryTasks: " + str(self.tasks)

    def __iter__(self):
        return self

    def __next__(self):
        if self._index == len(self._keys):
            self._index = 0
            raise StopIteration
        t = self.tasks[self._keys[self._index]]
        self._index += 1
        return t

    def __getitem__(self, i):
        return self.tasks[self._keys[i]]
