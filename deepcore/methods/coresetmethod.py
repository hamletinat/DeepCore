class CoresetMethod(object):
    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, **kwargs):
        # if fraction <= 0.0 or fraction > 1.0:
        #     raise ValueError("Illegal Coreset Size.")
        not_accept = [0 for frc in fraction if frc <= 0.0 or frc > 1.0]  ### change for multiple fractions
        if 0 in not_accept:
            raise ValueError("Illegal Coreset Size.")
        self.dst_train = dst_train
        self.num_classes = len(dst_train.classes)
        self.fraction = fraction
        self.random_seed = random_seed
        self.index = []
        self.args = args

        self.n_train = len(dst_train)
        # self.coreset_size = round(self.n_train * fraction)
        self.coreset_size = [round(self.n_train * fct) for fct in fraction] ### change for multiple fractions

    def select(self, **kwargs):
        return

