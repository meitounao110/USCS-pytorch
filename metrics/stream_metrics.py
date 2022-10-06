import numpy as np
from sklearn.metrics import confusion_matrix
import torch.distributed as dist
import torch


class _StreamMetrics(object):
    def __init__(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def update(self, gt, pred):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def get_results(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def to_str(self, metrics):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def reset(self):
        """ Overridden by subclasses """
        raise NotImplementedError()


class StreamSegMetrics(_StreamMetrics):
    """
    Stream Metrics for Semantic Segmentation Task
    """

    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten())

    @staticmethod
    def to_str(results):
        string = "\n"
        for k, v in results.items():
            if k != "Class IoU":
                string += "%s: %f\n" % (k, v)

        # string+='Class IoU:\n'
        # for k, v in results['Class IoU'].items():
        #    string += "\tclass %d: %f\n"%(k, v)
        return string

    def _fast_hist(self, label_true, label_pred):
        mask = (label_true >= 0) & (label_true < self.n_classes)
        hist = np.bincount(
            self.n_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def get_results(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return {
            "Overall Acc": acc,
            "Mean Acc": acc_cls,
            "FreqW Acc": fwavacc,
            "Mean IoU": mean_iu,
        }

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class AverageMeter(object):
    """Computes average values"""

    def __init__(self):
        self.book = dict()

    def reset_all(self):
        self.book.clear()

    def reset(self, id):
        item = self.book.get(id, None)
        if item is not None:
            item[0] = 0
            item[1] = 0

    def update(self, id, val):
        record = self.book.get(id, None)
        if record is None:
            self.book[id] = [val, 1]
        else:
            record[0] += val
            record[1] += 1

    def get_results(self, id):
        record = self.book.get(id, None)
        assert record is not None
        return record[0] / record[1]


class Metrics(_StreamMetrics):
    """
    for mimo
    """

    def __init__(self, n_classes, num_members):
        self.metrics = {}
        self.num_members = num_members
        for i in range(0, num_members):
            self.metrics["out_" + str(i)] = StreamSegMetrics(n_classes)
        self.metrics["out"] = StreamSegMetrics(n_classes)

    def reset(self):
        for i in range(0, self.num_members):
            self.metrics["out_" + str(i)].reset()
        self.metrics["out"].reset()

    def update(self, label_trues, label_preds):
        for i in range(0, self.num_members):
            self.metrics["out_" + str(i)].update(label_trues, label_preds["final_" + str(i)])
        self.metrics["out"].update(label_trues, label_preds["final"])

    def get_results(self):
        results = {}
        for i in range(0, self.num_members):
            results["out_" + str(i)] = self.metrics["out_" + str(i)].get_results()
        results["out"] = self.metrics["out"].get_results()
        return results

    def to_str(self, results):
        string = "\n"
        for i in range(0, self.num_members):
            string += f"Metrics_{i}:\n"
            for k, v in results["out_" + str(i)].items():
                if k != "Class IoU":
                    string += "\t%s: %f\n" % (k, v)
        string += f"Metrics_Mean:\n"
        for k, v in results["out"].items():
            if k != "Class IoU":
                string += "\t%s: %f\n" % (k, v)
        return string

    def max_results(self, results):
        result = []
        for i in range(0, self.num_members):
            result.append(results[f"out_{i}"]['Mean IoU'])
        result.append(results["out"]['Mean IoU'])
        result.sort()
        max_result = result[len(result) - 1]
        return max_result
