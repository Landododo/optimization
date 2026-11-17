def test_imports_and_segmentize():
    import osmnx  # noqa: F401
    import highspy  # noqa: F401
    import shapely
    from shapely.geometry import LineString

    line = LineString([(0, 0), (10, 0)])
    seg = shapely.segmentize(line, max_segment_length=2.5)
    assert len(seg.coords) == 5  # 10 m split into four 2.5 m pieces