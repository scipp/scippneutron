import unittest
import numpy as np
import scipp as sc

from neutron.imaging import operations as operations


class OperationsTest(unittest.TestCase):
    @staticmethod
    def _run_test(val_in):
        test_input = np.array(val_in)

        # Flatten to 1D input for the moment
        test_input = sc.Variable(["y", "x"], values=test_input)
        return operations.mask_from_adj_pixels(test_input)

    def test_center_true(self):
        input_data = [[True, True, True], [True, False, True],
                      [True, True, True]]

        returned = self._run_test(input_data)
        self.assertTrue(sc.all(sc.all(returned, "x"), "y").value)

    def test_center_false(self):
        input_data = [[False, False, False], [False, True, False],
                      [False, False, False]]

        returned = self._run_test(input_data)
        self.assertFalse(sc.all(sc.all(returned, "x"), "y").value)

    def test_center_not_changed(self):
        input_list = [[[True, True, True], [True, False, False],
                       [True, True, False]],
                      [[False, False, False], [True, True, False],
                       [True, False, False]]]

        for i in input_list:
            self.assertEqual(i, self._run_test(i).values.tolist())

    def test_edges_handle_correctly(self):
        test_input = [[False, True, False], [False, True, True],
                      [True, True, True]]

        # Top left should flip, all others should not
        expected = [[False, True, True], [False, True, True],
                    [True, True, True]]

        self.assertEqual(expected, self._run_test(test_input).values.tolist())

    def test_5d_works(self):
        test_input = [
            [True, True, True, True, True],
            [True, False, True, True, False],  # Should all -> True
            [True, True, True, True, True],
            [False, False, False, False, False],
            [False, True, False, False, False]
        ]  # Should -> False

        expected = [[True] * 5, [True] * 5, [True] * 5, [False] * 5,
                    [False] * 5]

        self.assertEqual(expected, self._run_test(test_input).values.tolist())

    def test_mean_filter(self):
        bulk_value = 1
        test_value = 4
        data = np.array([bulk_value] * 9 * 9 * 4,
                        dtype=np.float64).reshape(9, 9, 4)
        data = sc.Variable(['y', 'x', 'z'], values=data)
        data['z', 1]['x', 4]['y', 4].value = test_value  # centre at z == 1
        data['z', 2]['x', 4]['y', 0].value = test_value  # edge at z == 3
        data['z', 3]['x', 0]['y', 0].value = test_value  # corner at z == 2

        centre_mean = (bulk_value * 8 + test_value * 1) / 9
        corner_mean = (bulk_value * 3 + test_value * 1) / 4
        edge_mean = (bulk_value * 5 + test_value * 1) / 6

        mean = operations.mean_from_adj_pixels(data)
        assert sc.is_equal(mean['z', 0],
                           data['z',
                                0])  # mean of 1 everywhere same as original

        assert sc.is_equal(
            mean['z', 1]['y', 3:6]['x', 3:6],
            sc.Variable(['y', 'x'],
                        values=np.array([centre_mean] * 9).reshape(3, 3)))

        assert sc.is_equal(
            mean['z', 2]['y', 0:1]['x', 3:6],
            sc.Variable(['y', 'x'],
                        values=np.array([edge_mean] * 3).reshape(1, 3)))
        assert sc.is_equal(
            mean['z', 2]['y', 1:2]['x', 3:6],
            sc.Variable(['y', 'x'],
                        values=np.array([centre_mean] * 3).reshape(1, 3)))

        assert mean['z', 3]['y', 0]['x', 0].value == corner_mean

    def test_median_filter(self):
        bulk_value = 1.0
        test_value = 4.0
        data = np.array([bulk_value] * 9 * 9 * 4,
                        dtype=np.float64).reshape(9, 9, 4)
        data = sc.Variable(['y', 'x', 'z'], values=data)
        data['z', 1]['x', 4]['y', 4].value = test_value  # centre at z == 1

        data['z', 2]['x', 3]['y', 0].value = test_value  # edge at z == 3
        data['z', 2]['x', 4]['y', 0].value = test_value  # edge at z == 3
        data['z', 2]['x', 5]['y', 0].value = test_value  # edge at z == 3

        data['z', 3]['x', 0]['y', 0].value = test_value  # corner at z == 2
        data['z', 3]['x', 0]['y', 1].value = test_value  # corner at z == 2

        centre_median = bulk_value
        corner_median = np.median(
            [bulk_value, bulk_value, test_value, test_value])
        edge_median = np.median([
            bulk_value, bulk_value, bulk_value, test_value, test_value,
            test_value
        ])

        median = operations.median_from_adj_pixels(data)
        assert sc.is_equal(median['z', 0],
                           data['z',
                                0])  # median of 1 everywhere same as original

        assert sc.is_equal(
            median['z', 1]['y', 3:6]['x', 3:6],
            sc.Variable(['y', 'x'],
                        values=np.array([centre_median] * 9).reshape(3, 3)))

        assert sc.is_equal(
            median['z', 2]['y', 0:1]['x', 3:6],
            sc.Variable(['y', 'x'],
                        values=np.array([bulk_value, edge_median,
                                         bulk_value]).reshape(1, 3)))

        assert median['z', 3]['y', 0]['x', 0].value == corner_median


if __name__ == '__main__':
    unittest.main()
