import whaletext

def test_load_waimai():
    waimai_data = whaletext.datasets.load_waimai()
    assert waimai_data.shape[0] > 0

def test_load_lcqmc():
    lcqmc_train, lcqmc_valid, lcqmc_test = whaletext.datasets.load_lcqmc()
    assert lcqmc_train.shape[0] > 0
