import unittest
from prepare_data.extract import extract_pathes
from prepare_data.extract import extract

class TestPathExtraction(unittest.TestCase):
    def setUp(self):
        self.train_pathes = extract_pathes('train')
        self.dev_pathes = extract_pathes('dev')
        self.test_pathes = extract_pathes('test')

    def test_return_tuple(self):
        self.assertEqual(len(self.train_pathes), 3)
        self.assertEqual(len(self.dev_pathes), 3)
        self.assertEqual(len(self.test_pathes), 3)

    def test_return_imgs(self):
        self.assertEqual(len(self.train_pathes[0]), 712)
        self.assertEqual(len(self.dev_pathes[0]), 35)
        self.assertEqual(len(self.test_pathes[0]), 63)

class TestExtraction(unittest.TestCase):
    def setUp(self):
        self.train_result = extract('train')
        self.dev_result = extract('dev')
        self.test_result = extract('test')

    def test_shapes(self):
        self.assertEqual(self.train_result[0].shape, (712, 3, 512, 512))
        self.assertEqual(self.dev_result[0].shape, (35, 3, 512, 512))
        self.assertEqual(self.test_result[0].shape, (63, 3, 512, 512))

        self.assertEqual(self.train_result[1].shape, (712, 1, 512, 512))
        self.assertEqual(self.dev_result[1].shape, (35, 1, 512, 512))
        self.assertEqual(self.test_result[1].shape, (63, 1, 512, 512))

