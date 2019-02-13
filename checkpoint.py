import torch

import nnetwork


def save(save_dir, model, arch, hidden_units, train_data):
    model.class_to_idx = train_data.class_to_idx
    model.cpu
    torch.save(
        {
            'arch': arch,
            'hidden_units': hidden_units,
            'state_dict': model.state_dict(),
            'class_to_idx': model.class_to_idx
        },
        save_dir,
    )


def load(checkpoint_path, gpu):
    checkpoint = torch.load(checkpoint_path)
    model = nnetwork.setup(
        arch=checkpoint['arch'],
        hidden_units=checkpoint['hidden_units'],
        gpu=gpu
    )

    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])

    return model
