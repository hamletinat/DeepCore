from .earlytrain import EarlyTrain
import torch, time
import numpy as np
from ..nets.nets_utils import MyDataParallel


class GraNd(EarlyTrain):
    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, epochs=200, repeat=1,
                 specific_model=None, balance=False, **kwargs):
        super().__init__(dst_train, args, fraction, random_seed, epochs, specific_model)
        self.epochs = epochs
        self.n_train = len(dst_train)
        self.coreset_size = round(self.n_train * fraction)
        self.specific_model = specific_model
        self.repeat = repeat

        self.balance = balance

    def while_update(self, outputs, loss, targets, epoch, batch_idx, batch_size):
        if batch_idx % self.args.print_freq == 0:
            print('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f' % (
                epoch, self.epochs, batch_idx + 1, (self.n_train // batch_size) + 1, loss.item()))

    def before_run(self):
        if isinstance(self.model, MyDataParallel):
            self.model = self.model.module

    def finish_run(self):
        self.model.embedding_recorder.record_embedding = True  # recording embedding vector

        self.model.eval()

        embedding_dim = self.model.get_last_layer().in_features
        batch_loader = torch.utils.data.DataLoader(
            self.dst_train, batch_size=self.args.selection_batch, num_workers=self.args.workers)
        sample_num = self.n_train

        for i, (input, targets) in enumerate(batch_loader):
            self.model_optimizer.zero_grad()
            outputs = self.model(input.to(self.args.device))
            loss = self.criterion(outputs.requires_grad_(True),
                                  targets.to(self.args.device)).sum()
            batch_num = targets.shape[0]
            with torch.no_grad():
                bias_parameters_grads = torch.autograd.grad(loss, outputs)[0]
                self.norm_matrix[i * self.args.selection_batch:min((i + 1) * self.args.selection_batch, sample_num),
                self.cur_repeat] = torch.norm(torch.cat([bias_parameters_grads, (
                        self.model.embedding_recorder.embedding.view(batch_num, 1, embedding_dim).repeat(1,
                                             self.args.num_classes, 1) * bias_parameters_grads.view(
                                             batch_num, self.args.num_classes, 1).repeat(1, 1, embedding_dim)).
                                             view(batch_num, -1)], dim=1), dim=1, p=2)

        self.model.train()

        self.model.embedding_recorder.record_embedding = False

    def select(self, **kwargs):
        # Initialize a matrix to save norms of each sample on idependent runs
        self.norm_matrix = torch.zeros([self.n_train, self.repeat], requires_grad=False).to(self.args.device)

        for self.cur_repeat in range(self.repeat):
            self.run()
            self.random_seed = self.random_seed + 5

        self.norm_mean = torch.mean(self.norm_matrix, dim=1).cpu().detach().numpy()
        selection_results_list = [] ### changes for multipple fractions
        if not self.balance:
            for current_coreset_size in self.coreset_size: ### changes for multipple fractions
                top_examples = self.train_indx[np.argsort(self.norm_mean)][::-1][:current_coreset_size] # self.coreset_size -> current_coreset_size
                selection_results_list.append(top_examples)
        else:
            for current_fruction in self.fraction: ### changes for multipple fractions
                print("\n ***** current fraction= ", current_fruction)
                top_examples = np.array([], dtype=np.int64)
                for c in range(self.num_classes):
                    c_indx = self.train_indx[np.array(self.dst_train.targets) == c]
                    budget = round(current_fruction * len(c_indx)) #self.fraction
                    top_examples = np.append(top_examples, c_indx[np.argsort(self.norm_mean[c_indx])[::-1][:budget]])
                selection_results_list.append(top_examples) ### changes for multipple fractions
        return {"indices": selection_results_list, "scores": self.norm_mean} # top_examples -> selection_results_list  ### changes for multipple fractions
