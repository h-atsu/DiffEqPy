class Optimizer:
    def __init__(self):
        self.target = None
        self.hooks = []

    def setup(self, target):
        self.target = target
        return self

    def update(self):
        # None 以外のパラメータをリストにまとめる
        params = [p for p in self.target.params() if p.grad is not None]

        # optionalな前処理
        for f in self.hooks:
            f(params)

        # パラメータの更新
        # 新しくoptimizerを定義するときはupdate_oneのみを定義しておけば良い
        for param in params:
            self.update_one(param)

    def update_one(self, param):
        raise NotImplementedError()

    def add_hook(self, f):
        self.hooks.append(f)


class SGD(Optimizer):
    def __init__(self, lr=0.01):
        super().__init__()
        self.lr = lr

    def update_one(self, param):
        param.data -= self.lr * param.grad.data
