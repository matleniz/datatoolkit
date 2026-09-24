from dtk_engine import run_key


def test_hello_n5():
    res = run_key("hello", {"n": 5})
    assert res["metrics"]["sum"] == 30
    assert len(res["tables"]) == 1
    assert len(res["tables"][0]["records"]) == 5
    assert len(res["figures"]) == 1
