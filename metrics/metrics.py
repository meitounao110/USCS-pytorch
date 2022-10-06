import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = np.multiply(val, weight)
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum = np.add(self.sum, np.multiply(val, weight))
        self.count = self.count + weight
        self.avg = self.sum / self.count

    @property
    def value(self):
        return self.val

    @property
    def average(self):
        return np.round(self.avg, 5)


def batch_pix_accuracy(output, target):
    _, predict = torch.max(output, 1)

    predict = predict.int() + 1
    target = target.int() + 1

    pixel_labeled = (target > 0).sum()
    pixel_correct = ((predict == target) * (target > 0)).sum()
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    # if distributed:
    #     dist.all_reduce(pixel_labeled), dist.all_reduce(pixel_correct)
    dist.all_reduce(pixel_labeled), dist.all_reduce(pixel_correct)
    return pixel_correct.cpu().numpy(), pixel_labeled.cpu().numpy()


def batch_intersection_union(output, target, num_class):
    _, predict = torch.max(output, 1)
    predict = predict + 1
    target = target + 1

    predict = predict * (target > 0).long()
    intersection = predict * (predict == target).long()

    area_inter = torch.histc(intersection.float(), bins=num_class, max=num_class, min=1)
    area_pred = torch.histc(predict.float(), bins=num_class, max=num_class, min=1)
    area_lab = torch.histc(target.float(), bins=num_class, max=num_class, min=1)
    area_union = area_pred + area_lab - area_inter
    assert (area_inter <= area_union).all(), "Intersection area should be smaller than Union area"

    # if distributed:
    #     dist.all_reduce(area_inter), dist.all_reduce(area_union)

    dist.all_reduce(area_inter), dist.all_reduce(area_union)
    return area_inter.cpu().numpy(), area_union.cpu().numpy()


def eval_metrics(output, target, num_classes, ignore_index):
    target = target.clone()
    target[target == ignore_index] = -1
    correct, labeled = batch_pix_accuracy(output.data, target)
    inter, union = batch_intersection_union(output.data, target, num_classes)
    return [np.round(correct, 5), np.round(labeled, 5), np.round(inter, 5), np.round(union, 5)]


# ref https://github.com/CSAILVision/sceneparsing/blob/master/evaluationCode/utils_eval.py
def pixel_accuracy(output, target):
    output = np.asarray(output)
    target = np.asarray(target)
    pixel_labeled = np.sum(target > 0)
    pixel_correct = np.sum((output == target) * (target > 0))
    return pixel_correct, pixel_labeled


def inter_over_union(output, target, num_class):
    output = np.asarray(output) + 1
    target = np.asarray(target) + 1
    output = output * (target > 0)

    intersection = output * (output == target)
    area_inter, _ = np.histogram(intersection, bins=num_class, range=(1, num_class))
    area_pred, _ = np.histogram(output, bins=num_class, range=(1, num_class))
    area_lab, _ = np.histogram(target, bins=num_class, range=(1, num_class))
    area_union = area_pred + area_lab - area_inter
    return area_inter, area_union


class _DistMetrics(object):
    def __init__(self, n_classes, ignore_index):
        self.total_inter, self.total_union = 0, 0
        self.total_correct, self.total_label = 0, 0
        self.num_classes = n_classes
        self.ignore_index = ignore_index

    def reset(self):
        self.total_inter, self.total_union = 0, 0
        self.total_correct, self.total_label = 0, 0

    def update(self, outputs, labels):
        correct, labeled, inter, union = eval_metrics(outputs, labels, self.num_classes, self.ignore_index)
        self.total_inter, self.total_union = self.total_inter + inter, self.total_union + union
        self.total_correct, self.total_label = self.total_correct + correct, self.total_label + labeled

    def get_results(self):
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return {"Overall Acc": pixAcc, "Mean IoU": mIoU}

    @staticmethod
    def to_str(results):
        string = "\n"
        for k, v in results.items():
            string += "%s: %f\n" % (k, v)
        return string


class DistMetrics(object):
    """
    for mimo
    """

    def __init__(self, n_classes, ignore_index, num_members):
        self.metrics = {}
        self.num_members = num_members
        for i in range(0, num_members):
            self.metrics["out_" + str(i)] = _DistMetrics(n_classes, ignore_index)
        self.metrics["out"] = _DistMetrics(n_classes, ignore_index)

    def reset(self):
        for i in range(0, self.num_members):
            self.metrics["out_" + str(i)].reset()
        self.metrics["out"].reset()

    def update(self, outputs, labels):
        for i in range(0, self.num_members):
            self.metrics["out_" + str(i)].update(outputs["final_" + str(i)], labels)
        self.metrics["out"].update(outputs["final"], labels)

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
                string += "\t%s: %f\n" % (k, v)
        string += f"Metrics_Mean:\n"
        for k, v in results["out"].items():
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
