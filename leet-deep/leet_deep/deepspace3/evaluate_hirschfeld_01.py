import torch

from leet_deep.deepspace3 import run_training_ds3_hirshfeld_diffusion_01
from leet_deep.deepspace3.run_training_ds3_hirshfeld_diffusion_01 import ConfigurationHirshfeldData

import torch
import matplotlib.pyplot as plt


def run_diffusion(base_model, diffusion_model, device, smiles_input, num_atoms, num_steps):
    """
    Generate conformers using the trained diffusion model.

    Parameters:
    - model: The trained diffusion model.
    - num_atoms: Number of atoms in the molecule.
    - num_steps: Number of diffusion steps to reverse.
    - num_conformers: Number of conformers to generate.

    Returns:
    - A tensor of shape (num_conformers, num_atoms, 3) representing the generated conformers.
    """
    smiles_input = torch.unsqueeze(smiles_input,0)

    base_model.eval()
    diffusion_model.eval()

    with torch.no_grad():
        # Initialize with noise
        current_state = 0.0+torch.randn(1,64, 1) * 0.15
        current_state = current_state.to(device)

        all_hirshs = torch.zeros(num_steps+1,64,1)
        all_hirshs[0, :,:] = current_state

        for step in range(num_steps):
            # Apply the model to refine the current_state
            # This is a placeholder for how your model updates the state;
            # you'll need to adjust it based on your model's specific interface
            timestep = num_steps - step
            # Assuming timestep decreases in reverse diffusion
            base_model_out = base_model(smiles_input, smiles_input)
            predicted_noise = diffusion_model(base_model_out[0],current_state)
            predicted_noise_scaled = predicted_noise[:,:num_atoms]
            current_state = current_state - predicted_noise

            norms = predicted_noise_scaled[0,:num_atoms,0]
            print(norms)
            all_hirshs[step+1,:] = current_state

        return current_state , all_hirshs


if __name__ == "__main__":
    # Assuming `model` is your trained model
    # Example parameters
    num_atoms = 64   # Adjust based on your molecule
    num_steps = 100  # Total diffusion steps in reverse


    #base_model, diffusion_model, conf = run_training_3d_diffusion_01.create_model("C:\\dev\\leet-chem\\leet-deep\\configs\\diffusion_3d_dm\\conf_diffusion_3d_a_02.json")
    #base_model, diffusion_model, conf = run_training_3d_diffusion_01.create_model("C:\\dev\\leet-chem\\leet-deep\\configs\\diffusion_3d_dm\\conf_diffusion_3d_a_02_new.json")

    conf = ConfigurationHirshfeldData("C:\\dev\\leet-chem\\leet-deep\\configs\\ds3_diffusion_hirshfeld\\diffusion_hirshfeld_conf_01_run_01.json")
    conf.load_config()
    base_model, diffusion_model = run_training_ds3_hirshfeld_diffusion_01.create_model(conf)

    train_loader , val_loader = run_training_ds3_hirshfeld_diffusion_01.create_dataset(conf)

    base_model = base_model.to(conf.device)
    diffusion_model = diffusion_model.to(conf.device)

    base_model = base_model.to(conf.device)
    diffusion_model = diffusion_model.to(conf.device)


    batch = next(iter(train_loader))
    inputs_smiles = batch['smiles_enc'].to(conf.device)
    inputs_num_atoms = batch['num_atoms_with_hydrogen'].to(conf.device)
    target_hirshfelds = batch['hirshfeld']
    num_atoms_i = batch['num_atoms_with_hydrogen']
    idx_mol = 12

    final_state , all_states = run_diffusion(base_model,diffusion_model,conf.device,inputs_smiles[idx_mol,:],inputs_num_atoms[idx_mol,:],20)

    # Plotting each line
    all_states = torch.squeeze(all_states)
    plt.cla()
    for i in range(num_atoms_i[idx_mol]):
        plt.plot(all_states[:, i], label=f'Line {i + 1}')
        plt.axhline( y=target_hirshfelds[idx_mol,i] , linestyle='--' )

    # Adding labels and legend
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.title('Plotting 16 Lines from a PyTorch Tensor')
    plt.legend()

    # Display the plot
    plt.show()

    print("mkay..")

    multiple_molecules_conformers = torch.zeros((16,num_atoms,3),device=conf.device)
    multiple_molecules_num_atoms  = torch.zeros((16,1))
    multiple_molecules_idxmol     = range(16)
    for idx_mol in multiple_molecules_idxmol:
        inputs = torch.unsqueeze(inputs_pre[idx_mol, :], 0)
        num_atoms_i = batch['num_atoms'][idx_mol]

        print('\n\n\n\n\n\n--------------------------------------------------------------')
        generated_conformers , all_conformers = run_diffusion(base_model, diffusion_model, conf.device, inputs, num_atoms_i, num_steps, num_conformers)