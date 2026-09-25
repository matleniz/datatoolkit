from dtk_engine import api, run_key
from dtk_engine.demo_data import TEST_CSV, TRAIN_CSV
from dtk_engine.transform_registry import get_transform


def _tables(res):
    return {t["title"]: t["records"] for t in res["tables"]}


def test_defaults_on_demo_data():
    res = run_key("preprocessing_advisor", {})
    assert res["metrics"]["model_family"] == "any"
    assert res["metrics"]["n_recommendations"] >= 1
    assert {"recommendations", "columns"} <= set(_tables(res))


def test_json_steps_with_test_family_and_target():
    res = run_key(
        "preprocessing_advisor",
        {
            "source": {"kind": "csv", "path": TRAIN_CSV},
            "test": {"kind": "csv", "path": TEST_CSV},
            "model_family": "linear",
            "target": "Survived",
        },
    )
    recs = _tables(res)["recommendations"]
    assert res["metrics"]["n_leak_warnings"] >= 1  # PassengerId
    for rec in recs:  # plain JSON params, each a valid step
        assert isinstance(rec["params"], dict)
        get_transform(rec["op"]).parse(rec["params"])
    assert "Survived" not in {r["column"] for r in recs}
    assert "PassengerId" in res["text"]


def test_api_door_renders():
    train, test = api.load(TRAIN_CSV), api.load(TEST_CSV)
    result = api.advise(train, test, model_family="tree", target="Survived")
    assert result.metrics["model_family"] == "tree"
    assert "recommendations" in result._repr_html_()
