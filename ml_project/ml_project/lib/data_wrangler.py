class DataWrangler:
    @classmethod
    def _get_data(cls, file, separator, conversion):
        data = []
        for line in file:
            if conversion:
                data.append([conversion(item.strip()) for item in line.strip().split()])
            else:
                data.append([item.strip() for item in line.strip().split(separator)])

        return data

    @classmethod
    def read_from_file(cls, filepath, conversion = None, filetype = ""):
        data = []
        with open(filepath, 'r') as f:
            if filetype == "":
                data = cls._get_data(f, ' ', conversion)
            elif filetype == "csv":
                data = cls._get_data(f, ',', conversion)
            elif filetype == "tsv":
                data = cls._get_data(f, '\t', conversion)
            else:
                print("Filetype not supported")

        return data

    @classmethod
    def write_to_file(cls, filepath, predictions):
        text_to_write = "\n".join(["{} {}".format(i, predictions[i]) for i in range(len(predictions))])

        with open(filepath, 'w') as f:
            f.write(text_to_write)