from bilby.gw.detector import InterferometerList, get_empty_interferometer


def build_interferometer_list(detector_names):
    """InterferometerList mit Unterstützung für ET-1/ET-2/ET-3.

    Die Reihenfolge von `detector_names` bleibt erhalten (die Detektor-Blöcke
    des Netzes sind positionsabhängig)."""
    if any(d.startswith("ET-") for d in detector_names):
        triangle = {ifo.name: ifo for ifo in InterferometerList(["ET-"])}
    else:
        triangle = {}
    ifos = [
        triangle[d] if d in triangle else get_empty_interferometer(d)
        for d in detector_names
    ]
    return InterferometerList(ifos)