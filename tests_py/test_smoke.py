import pytest

import fky_lapsim as fl


def test_project_simulate():
    p = fl.example_project()
    assert fl.validate(p)["valid"] is True
    m = fl.defaults()["motion"]
    assert fl.simulate(p, m)


def test_lap_validation():
    assert fl.validate_lap_vehicle(fl.lap_vehicle_demo())["valid"] is True
    assert fl.validate_track(fl.track_demo())["valid"] is True


def test_session_rejects_targetless_request():
    demo = fl.formula_car_demo()
    with pytest.raises(ValueError):
        fl.OptimizationSession(demo["project"], fl.defaults()["optimization"])


def test_run_lap():
    r = fl.run_lap(fl.lap_vehicle_demo(), fl.track_demo(), {})
    assert r


def test_error_is_valueerror():
    with pytest.raises(ValueError):
        fl.call("nope")
