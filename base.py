class BaseLearner(object): pass


class Mapper(object):
    def __init__(self, y=None):
        if not y:
            self.label_map = None
            self.reverse_map = None
        else:
            self.set_label_maps(y)

    def set_label_maps(self, y):
        """FUNCTION TO SET MAPPERS FOR CLASS LABELS"""
        self.label_map = list(set(y))
        self.label_map = {self.label_map[i]: i for i in range(len(self.label_map))}
        self.reverse_map = {self.label_map[key]: key for key in self.label_map}

    def map_labels(self, y, margin_map=False):
        """FUNCTION TO MAP CLASS LABELS TO INTEGERS FROM 0 TO n_class-1"""
        if not self.label_map:
            self.set_label_maps(y)
        if margin_map:
            print(self.label_map)
            return [(-1) ** (self.label_map[y_i] + 1) for y_i in y]
        return [self.label_map[y_i] for y_i in y]

    def map_reverse(self, y):
        """FUNCTION TO MAP INTEGERS BACK TO CLASS LABELS"""
        return [self.reverse_map[y_i] for y_i in y]
