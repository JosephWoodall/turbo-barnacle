import unittest
from data_pipeline.source_data_retrieval import SourceDataRetrieval


class SourceDataRetrievalTest(unittest.TestCase):
    """
    The SourceDataRetrievalTest class tests the SourceDataRetrieval class for execution of functions.

    """

    def __init__(self):
        self.SourceDataRetrieval = SourceDataRetrieval()
        print("------------------------------SOURCE_DATA_RETRIEVAL_TEST_INITIALIZED")


if __name__ == '__main__':
    unittest.main()
