from gym_d2d.conversion import dB_to_linear, linear_to_dB, dBm_to_W, W_to_dBm

from pytest import approx


def test_dB_to_linear():
    assert dB_to_linear(0) == approx(1)
    assert dB_to_linear(1) == approx(1.258925)
    assert dB_to_linear(2) == approx(1.584893)
    assert dB_to_linear(10) == approx(10)
    assert dB_to_linear(30) == approx(1000)
    assert dB_to_linear(100) == approx(10000000000)


def test_linear_to_dB():
    assert linear_to_dB(1) == approx(0)
    assert linear_to_dB(2) == approx(3.0103)
    assert linear_to_dB(3) == approx(4.771213)
    assert linear_to_dB(4) == approx(6.0206)
    assert linear_to_dB(5) == approx(6.9897)
    assert linear_to_dB(30) == approx(14.771213)
    assert linear_to_dB(100) == approx(20)
    assert linear_to_dB(1000) == approx(30)


def test_dBm_to_W():
    assert dBm_to_W(0) == approx(0.001)
    assert dBm_to_W(1) == approx(0.001258925)
    assert dBm_to_W(2) == approx(0.001584893)
    assert dBm_to_W(10) == approx(0.01)
    assert dBm_to_W(30) == approx(1)
    assert dBm_to_W(100) == approx(10000000)


def test_W_to_dBm():
    assert W_to_dBm(0.1) == approx(20)
    assert W_to_dBm(0.2) == approx(23.0103)
    assert W_to_dBm(1) == approx(30)
    assert W_to_dBm(2) == approx(33.0103)
    assert W_to_dBm(5) == approx(36.9897)
    assert W_to_dBm(100) == approx(50)
    assert W_to_dBm(1000) == approx(60)
