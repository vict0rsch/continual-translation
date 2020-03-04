from collections import OrderedDict


class BaseTask:
    def __init__(self):
        super().__init__()
        self.key = ""
        self.loss_names = []
        self.needs_D = False
        self.needs_lr = False
        self.needs_z = False
        self.model_name = ""
        self.input_key = ""
        self.lambda_key = ""
        self.loss_function = ""
        self.priority = 0
        self.metrics_key = ""
        self.threshold_key = ""

    def setup(self):

        assert self.key

        if self.needs_lr and not self.model_name:
            raise ValueError("needs_lr and model_name")

        if self.needs_D:
            self.D_A = f"netD_A_{self.key}"
            self.D_B = f"netD_B_{self.key}"


class GrayTask(BaseTask):
    def __init__(self):
        super().__init__()


class RotationTask(BaseTask):
    def __init__(self):
        super().__init__()


class DepthTask(BaseTask):
    def __init__(self):
        super().__init__()


class AuxiliaryTasks:
    def __init__(self, keys=[]):
        super().__init__()

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

        for t in self.tasks:
            t.setup()


class T:
    def __init__(self, ts):
        self.tasks = OrderedDict([(t, t + 10) for t in ts])

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
