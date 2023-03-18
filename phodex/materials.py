def n_si(lbda):
    """https://refractiveindex.info/?shelf=main&book=Si&page=Salzberg"""
    if lbda < 1.357 or lbda > 11.04:
        raise ValueError(f"Material model invalid for {lbda=}!")
    l2 = lbda**2
    n2 = (
        1
        + 10.6684293 * l2 / (l2 - 0.301516485**2)
        + 0.0030434748 * l2 / (l2 - 1.13475115**2)
        + 1.54133408 * l2 / (l2 - 1104**2)
    )
    return n2**0.5


def n_sio2(lbda):
    """https://refractiveindex.info/?shelf=main&book=SiO2&page=Malitson"""
    if lbda < 0.21 or lbda > 6.7:
        raise ValueError(f"Material model invalid for {lbda=}!")
    l2 = lbda**2
    n2 = (
        1
        + 0.6961663 * l2 / (l2 - 0.0684043**2)
        + 0.4079426 * l2 / (l2 - 0.1162414**2)
        + 0.8974794 * l2 / (l2 - 9.896161**2)
    )
    return n2**0.5


def n_si3n4(lbda):
    """https://refractiveindex.info/?shelf=main&book=Si3N4&page=Luke"""
    if lbda < 0.31 or lbda > 5.504:
        raise ValueError(f"Material model invalid for {lbda=}!")
    l2 = lbda**2
    n2 = 1 + 3.0249 * l2 / (l2 - 0.1353406**2) + 40314 * l2 / (l2 - 1239.842**2)
    return n2**0.5
