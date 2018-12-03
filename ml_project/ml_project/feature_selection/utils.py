
def get_intersection(selections, selected_features, reject):
    commons = []
    for feature in selected_features:
        common = True
        for key in selections:
            if key == reject:
                continue
            if feature not in selections[key]:
                common = False
                break
        if common:
            commons.append(feature)

    return commons