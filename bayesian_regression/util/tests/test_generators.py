import unittest
from bayesian_regression.util.generators import band_table, block_table

from biom import Table
import numpy.testing as npt
import pandas.util.testing as pdt
import numpy as np
import pandas as pd
from sklearn.utils import check_random_state


class TestGenerator(unittest.TestCase):

    def setUp(self):
        pass

    def test_band_table(self):
        res = band_table(5, 6, alpha=3)
        res_table, res_md, res_fmd, res_beta, res_theta, res_gamma = res
        mat = np.array(
            [[17.0, 17.0, 4.0, 1.0, 0.0],
             [0.0, 1.0, 3.0, 4.0, 1.0],
             [1.0, 1.0, 5.0, 10.0, 4.0],
             [1.0, 0.0, 1.0, 5.0, 8.0],
             [0.0, 0.0, 5.0, 3.0, 6.0],
             [0.0, 0.0, 2.0, 5.0, 4.0]]
        )

        samp_ids = ['S0', 'S1',	'S2', 'S3', 'S4']
        feat_ids = ['F0', 'F1', 'F2', 'F3', 'F4', 'F5']
        exp_fmd = pd.DataFrame({'mu': [3.045444, 5.800314, 6.957476,
                                       8.528105, 8.735116, 9.481786]},
                               index=feat_ids)
        exp_table = Table(mat, feat_ids, samp_ids)
        exp_md = pd.DataFrame({'G': [2., 4., 6., 8., 10.]},
                              index=samp_ids)

        exp_beta = np.array(
            [[-0.48699685, -0.51737237, -0.70588861, -0.59306809, -0.6546415 ]]
        )
        exp_theta = np.array(
            [-1.76471591, 0.09173837, 1.11383944, 0.95193505, 0.30415688]
        )
        exp_gamma = np.array(
            [[3.1040167, 2.59893833, 4.4744781, 4.35601705, 4.91757565]]
        )

        self.assertEqual(exp_table, res_table)
        pdt.assert_frame_equal(exp_md, res_md)
        pdt.assert_frame_equal(exp_fmd, res_fmd)
        npt.assert_allclose(exp_beta, res_beta, atol=1e-7)
        npt.assert_allclose(exp_theta, res_theta, atol=1e-7)

    def test_block_table(self):
        res = block_table(6, 8)
        res_table, res_md, res_fmd, res_beta, res_theta, res_gamma = res
        mat = np.array(
            [[0.0, 0.0, 38.0, 190.0, 45.0, 2.0],
             [1.0, 50.0, 7.0, 132.0, 266.0, 0.0],
             [0.0, 8.0, 3.0, 76.0, 63.0, 0.0],
             [0.0, 0.0, 118.0, 31.0, 0.0, 391.0],
             [15.0, 8.0, 91.0, 0.0, 0.0, 0.0],
             [196.0, 54.0, 23.0, 1.0, 0.0, 0.0],
             [59.0, 196.0, 26.0, 1.0, 0.0, 0.0],
             [140.0, 85.0, 99.0, 0.0, 0.0, 0.0]]
        )

        samp_ids = ['S0', 'S1',	'S2', 'S3', 'S4', 'S5']
        feat_ids = ['F0', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7']

        exp_fmd = pd.DataFrame({
            'class': [1, 1, 1, 1, -1, -1, -1, -1],
            'mu': [6.882026, 6.200079, 6.489369, 7.120447,
                   4.933779, 3.511361, 4.475044, 3.924321]
            }, index=feat_ids
        )

        exp_table = Table(mat, feat_ids, samp_ids)
        exp_md = pd.DataFrame({'G': [0., 0., 1., 1., 1., 1.]},
                              index=samp_ids)
        exp_beta = np.array([[0.48220975, 0.04219932, -0.51668982,
                              1.55558875, 2.56861688, 1.27868026, 1.6225236]])
        exp_theta = np.array([0.50094173, 2.40521398, 3.48880514,
                              2.10665223, -1.66846398, -5.71559097])
        exp_gamma = np.array([[-3.25737809, 0.08819684, 3.63515226,
                               -7.62944273, -12.13872507, -7.21789007,
                               -8.07587006]])

        self.assertEqual(exp_table, res_table)
        pdt.assert_frame_equal(exp_md, res_md)
        pdt.assert_frame_equal(exp_fmd, res_fmd)
        npt.assert_allclose(exp_beta, res_beta, atol=1e-7)
        npt.assert_allclose(exp_theta, res_theta, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
