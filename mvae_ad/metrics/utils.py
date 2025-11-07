"""Utility functions for computing metrics."""

import torch
from multivae.data.utils import set_inputs_to_device
from torch.utils.data import DataLoader


def get_embeddings_and_id_dict(model, dataset, batch_size, device):
    """Get all the embeddings and ids from the dataset"""
    model = model.eval().to(device)
    embeddings = []
    logvars = []
    participants = []
    sessions = []
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=batch_size):
            batch = set_inputs_to_device(batch, device)
            output = model.encoder(batch.data)
            embeddings.append(output["embedding"])
            logvars.append(output["log_covariance"])
            participants.extend(list(batch["participant"]))
            sessions.extend(list(batch["session"]))

    embeddings = torch.cat(embeddings, dim=0)
    logvars = torch.cat(logvars, dim=0)

    return {
        "embeddings": embeddings,
        "logvars": logvars,
        "participant": participants,
        "session": sessions,
    }
