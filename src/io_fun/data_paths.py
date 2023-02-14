def get_path(home, dataset):
    """
    :param home: part of path that is specific to user, e.g. /Users/..../
    :param dataset: dataset name that needs to be collected. Note that this can either be a target or source dataset
    :return: complete path to storage location of target dataset
    """

    if dataset == "isic":
        img_dir = f"{home}/ISIC2018/ISIC2018_Task3_Training_Input"
        label_dir = f"{home}/ISIC2018/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv"

        return img_dir, label_dir

    if dataset == "chest":
        data_dir = f"{home}/chest_xray"

        return data_dir

    if dataset == "breast":
        data_dir = f"{home}/breast/"

        return data_dir

    if dataset == "thyroid":
        data_dir = f"{home}/thyroid/"

        return data_dir

    if (dataset == "pcam-small") | (dataset == "pcam-middle"):
        data_dir = f"{home}/PCam/png_images"

        return data_dir

    if dataset == "knee":
        data_dir = f"{home}/knee/"

        return data_dir

    if dataset == "kimia":
        data_dir = f"{home}/kimia_path_960"

        return data_dir

    if dataset == "mammograms":
        data_dir = f"{home}/mammograms"

        return data_dir

    if dataset == "imagenet":
        data_dir = f"{home}/ImageNet/images"

        return data_dir
