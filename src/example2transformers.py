from datasets import load_dataset
import transformers

# dataset = load_dataset("olivierdehaene/xkcd")


# write a class that gets the dataset and runs certain pipelines
class DataAnalysis:
    def __init__(self, dataset_url=None):
        self.dataset_url = dataset_url

    def get_dataset(self):
        self.dataset = load_dataset(self.dataset_url)

    # print basic information about the dataset
    def print_info(self):
        print(self.dataset["train"].features)
        print(self.dataset["train"].info)
        return self.dataset["train"].info, self.dataset["train"].features

    def set_pipeline(self, pipeline_name):
        self.pipe = transformers.pipeline(pipeline_name)

    def analyze_fragment(self, fragment, column_name):
        analyze_output = self.pipe(self.dataset["train"][:fragment][column_name])
        return analyze_output

    def analyze_all(self, column_name):
        analyze_output = self.pipe(self.dataset["train"][column_name])
        return analyze_output
