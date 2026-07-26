import sparrow_engine


def test_trt_unsupported_hardware_is_public() -> None:
    assert "TrtUnsupportedHardware" in sparrow_engine.__all__
    assert issubclass(
        sparrow_engine.TrtUnsupportedHardware,
        sparrow_engine.SparrowEngineError,
    )
