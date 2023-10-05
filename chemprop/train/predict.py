from typing import List

import torch
from tqdm import tqdm

from chemprop.data import MoleculeDataLoader, MoleculeDataset, StandardScaler
from chemprop.models import InteractionModel
import numpy as np
from chemprop.args import TrainArgs

def predict(model: InteractionModel,
            data_loader: MoleculeDataLoader,
            args: TrainArgs,
            disable_progress_bar: bool = False,
            scaler: StandardScaler = None,
            tokenizer=None) -> List[List[float]]:
    """
    Makes predictions on a dataset using an ensemble of models.

    :param model: A :class:`~chemprop.models.model.InteractionModel`.
    :param data_loader: A :class:`~chemprop.data.data.MoleculeDataLoader`.
    :param disable_progress_bar: Whether to disable the progress bar.
    :param scaler: A :class:`~chemprop.features.scaler.StandardScaler` object fit on the training targets.
    :return: A list of lists of predictions. The outer list is molecules while the inner list is tasks.
    """
    model.eval()

    preds = []

    for batch in tqdm(data_loader, disable=disable_progress_bar, leave=False):
        # Prepare batch
        batch: MoleculeDataset
        mol_batch, features_batch, atom_descriptors_batch, atom_features_batch, bond_features_batch, add_feature = \
            batch.batch_graph(), batch.features(), batch.atom_descriptors(), batch.atom_features(), batch.bond_features(), batch.add_features()

        add_feature = torch.Tensor(add_feature)

        # Make predictions

        with torch.no_grad():
            batch_preds  = model(mol_batch, add_feature, features_batch, atom_descriptors_batch, atom_features_batch, bond_features_batch)
        batch_preds = batch_preds.data.cpu().numpy()
        # Inverse scale if regression
        if scaler is not None:
            batch_preds = scaler.inverse_transform(batch_preds)

        # Collect vectors
        batch_preds = batch_preds.tolist()
        preds.extend(batch_preds)

    return preds
