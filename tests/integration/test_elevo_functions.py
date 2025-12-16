import pytest
import numpy as np
from utils.pred_utils import Prediction_ELEvo
import os
import datetime

@pytest.fixture
def load_test_pre_conjunction_Prediction_ELEvo():
    
    pre_conj_input_file = 'tests/fixtures/test_elevo_input_2022_05_10T14_48_00_CME_001.npy'
    pre_conj_output_file = 'tests/fixtures/test_elevo_output_2022_05_10T14_48_00_CME_001.npy'

    pre_conj_input = np.load(pre_conj_input_file, allow_pickle=True).item()
    pre_conj_output = np.load(pre_conj_output_file, allow_pickle=True).item()

    return pre_conj_input, pre_conj_output

@pytest.fixture
def load_test_post_conjunction_Prediction_ELEvo():
    post_conj_input_file = 'tests/fixtures/test_elevo_input_2025_05_12T23_48_00_CME_001.npy'
    post_conj_output_file = 'tests/fixtures/test_elevo_output_2025_05_12T23_48_00_CME_001.npy'

    post_conj_input = np.load(post_conj_input_file, allow_pickle=True).item()
    post_conj_output = np.load(post_conj_output_file, allow_pickle=True).item()

    return post_conj_input, post_conj_output

@pytest.mark.parametrize("input", [
    'load_test_pre_conjunction_Prediction_ELEvo',
    'load_test_post_conjunction_Prediction_ELEvo'
])
def test_Prediction_ELEvo(input, request):
    test_input, output = request.getfixturevalue(input)

    (time2_num, cme_r, cme_lat,
     cme_lon, cme_a, cme_b,
     cme_c, cme_id, cme_v,
     halfwidth, arr_time_fin, arr_time_err0,
     arr_time_err1, arr_id, arr_hit,
     arr_speed_list, arr_speed_err_list) = Prediction_ELEvo(**test_input)

    test_output = {}
    test_output["hc_time_num1"]= time2_num
    test_output["hc_r1" ]= cme_r
    test_output["hc_lat1" ]= cme_lat
    test_output["hc_lon1" ]= cme_lon
    test_output["a1_ell" ]= cme_a
    test_output["b1_ell" ]= cme_b
    test_output["c1_ell" ]= cme_c
    test_output["hc_id1" ]= cme_id
    test_output["hc_v1" ]= cme_v
    test_output["halfwidth" ]= halfwidth
    test_output["hc_arr_time1"] = arr_time_fin
    test_output["hc_err_arr_time_min1"] = arr_time_err0
    test_output["hc_err_arr_time_max1"] = arr_time_err1
    test_output["hc_arr_id1"] = arr_id
    test_output["hc_arr_hit1"] = arr_hit
    test_output["hc_arr_speed1"] = arr_speed_list
    test_output["hc_err_arr_speed1"] = arr_speed_err_list

    rtol = 1e-5
    atol = 1e-8

    for key in output:

        if isinstance(output[key], list) or isinstance(output[key], np.ndarray):
            type_inst = type(output[key][0])
        else:
            type_inst = type(output[key])

        if type_inst == datetime.datetime:
            test_date_array = np.array([d.timestamp() for d in test_output[key]])
            output_date_array = np.array([d.timestamp() for d in output[key]])
            np.testing.assert_allclose(test_date_array, output_date_array, rtol=rtol, atol=atol)

        elif type_inst == bytes or type_inst == str:
            np.testing.assert_array_equal(test_output[key], output[key])
            
        else:
            np.testing.assert_allclose(test_output[key], output[key], rtol=rtol, atol=atol)
