import typing

def split_in_circles(positions: List[]):
    """ Given continous one dim list of agent positions
        Split it to the circles.  

    Args:
        positions (List[]): [description]
    """
    start = positions[0]
    circles = [start]
    for i, pos in enumerate(positions):
        if i == 0:
            continue
        pass