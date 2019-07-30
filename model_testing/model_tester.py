import torch
from torch.utils.data import DataLoader

from .collectors import MultiCollector


class Test:
    """
    Base test class that mimics the unittest.TestCase class
    """

    def setUp(self):
        """
        This method is called before Test.test(), it is used to prepare the test
        """
        pass

    def run(self):
        self.setUp()
        self.test()
        self.tearDown()

    def test(self):
        """
        Override this method with your test code
        """
        pass

    def tearDown(self):
        """
        This method is called after Test.test(), it is used to clean-up.
        """
        pass


class ModelTester:
    """
    Model testing class.
    Given a model, a dataset and tests to execute ModelTester.run()
    takes care of executing all the given tests on the model. The user should use
    ModelTester.add_batch_computing() to provide the details on
    how to feed data into the model.
    See method documentation for details.

    Fields accessible from TrainingCallback:
        batch_size
        device
    """

    def __init__(self, batch_size, device):
        self.batch_size = batch_size
        self.device = device
        self._collectors = MultiCollector()

        class StubBatchComputer:
            def __call__(self, data):
                raise NotImplementedError('You should specify a batch computing callback ' +
                                          'using ModelTester.add_batch_computing()')
        self._batch_processor = None
        self.add_batch_processor(StubBatchComputer())

    def add_collector(self, name, collector):
        collector.tester = self
        self._collectors.add_collector(name, collector)
        return self

    def add_batch_processor(self, batch_processor):
        batch_processor.tester = self
        self._batch_processor = batch_processor
        return self

    @torch.no_grad()
    def run(self, model, test_set, batch_processor=None):
        model.eval()

        if batch_processor:
            self.add_batch_processor(batch_processor)

        testing_data = DataLoader(test_set, self.batch_size, drop_last=True)
        
        self._collectors.on_test_begin()

        for data in testing_data:
            self._collectors.on_batch_begin()

            if torch.is_tensor(data):
                data = data.to(self.device)
            else:
                data = [d.to(self.device) for d in data]
            
            output, target = self._batch_processor(model, data)
            self._collectors(output.cpu(), target.cpu())
    
            self._collectors.on_batch_end()

        results = self._collectors.collect_results()
        
        self._collectors.on_test_end()

        return results
