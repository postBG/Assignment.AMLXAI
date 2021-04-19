import os

class ResultData(object):
    def __init__(self, results, lengths):
        assert len(results) == len(lengths)
        self.results = results
        self.lengths = lengths

    def averages(self):
        return [sum(result) / l for result, l in zip(self.results, self.lengths)]

    def average(self, idx):
        return sum(self.results[idx]) / self.lengths[idx]

def read_result(path):
    rel_path = os.path.join('result_data', path)
    with open(rel_path, 'r') as f:
        lines = f.readlines()
        results = [[float(x) for x in l.strip().split()] for l in lines]
        lengths = list(range(1, len(lines) + 1))
        return ResultData(results, lengths)
