def CreateDataset(args):
    if args.dataset_name:
        from data.dental import DentalSegDataset
        dataset = DentalSegDataset
    else:
        assert 0

    return dataset
