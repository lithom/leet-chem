import random

import torch
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem

import run_training_3d_diffusion_01
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

from leet_deep.deepspace3 import run_training_ds3_diffusion_01


def generate_conformers(base_model, diffusion_model, device, smiles_input, num_atoms, num_steps, num_conformers):
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
    base_model.eval()
    diffusion_model.eval()

    with torch.no_grad():
        # Initialize with noise
        current_state = torch.randn(num_conformers, 32, 3) * 2.0
        current_state = current_state.to(device)

        all_conformers = torch.zeros(num_steps+1,num_conformers,32,3)
        all_conformers[0, :, :, :] = current_state

        for step in range(num_steps):
            # Apply the model to refine the current_state
            # This is a placeholder for how your model updates the state;
            # you'll need to adjust it based on your model's specific interface
            timestep = num_steps - step  # Assuming timestep decreases in reverse diffusion
            base_model_out = base_model(smiles_input, smiles_input)
            base_model_out_a = base_model_out[0].repeat(num_conformers,1,1)
            predicted_noise = diffusion_model(base_model_out_a,current_state)
            predicted_noise_scaled = predicted_noise[:,:32,:] * 25
            current_state = current_state - predicted_noise_scaled

            if True: # manual centering..
                #hmm.. manually stabilize, was not needed for deepsspace2..
                # Compute the mean of each batch
                batch_means = torch.mean(current_state, dim=1, keepdim=True)
                # Subtract the mean from each point in the corresponding batch
                current_state = current_state - batch_means

            norms = torch.sum( torch.norm( predicted_noise_scaled[:,:num_atoms,:] , p=2 , dim=2 ) , dim=1  )
            print(norms)
            all_conformers[step+1,:,:,:] = current_state

        return current_state , all_conformers



# Visualization of the first generated conformer
def plot_conformer(conformer, num_atoms):
    """
    Plot a single conformer using matplotlib.

    Parameters:
    - conformer: A tensor of shape (num_atoms, 3) representing the conformer's coordinates.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(conformer[:num_atoms, 0], conformer[:num_atoms, 1], conformer[:num_atoms, 2])

    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    plt.show()




def plot_conformers_2(conformers):
    """
    Plot multiple conformers using matplotlib, each with a different color.

    Parameters:
    - conformers: A tensor of shape (num_conformers, num_atoms, 3) representing the conformers' coordinates.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Define a colormap to select colors from
    colormap = plt.cm.get_cmap('hsv', conformers.size(0))

    for i, conformer in enumerate(conformers):
        ax.scatter(conformer[:, 0], conformer[:, 1], conformer[:, 2], color=colormap(i), label=f'Conformer {i + 1}')

    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    plt.title('Generated Conformers')
    plt.legend()
    plt.show()



def plot_conformer_with_bonds_from_rdkit_mol(ax, conformer_a, mol):
    conformer = conformer_a[:mol.GetNumAtoms(),:]
    # Plotting atoms
    atom_colors = {'C': 'black', 'H': 'gray', 'O': 'red', 'N': 'blue' ,'F': 'green','S': 'yellow', 'Cl': 'YellowGreen', 'Br': 'orange'}  # Extend this dictionary based on your needs
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        pos = conformer[idx]
        #print(atom.GetSymbol())
        color = atom_colors.get(atom.GetSymbol(), 'cyan')  # Default color if element not in dict
        ax.scatter(pos[0], pos[1], pos[2], color=color, s=60, alpha=0.6)  # Adjust size (s) as needed

    # Plotting bonds
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        start_pos, end_pos = conformer[start], conformer[end]
        ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], [start_pos[2], end_pos[2]], color='gray')
def plot_conformer_with_bonds_from_smiles(ax, conformer, smiles, create_figure_and_axes = False):
    """
    Plots a conformer with atoms represented as spheres and bonds as lines.
    Atoms are colored differently based on the element.

    Parameters:
    - conformer: A tensor or array of shape (num_atoms, 3) representing atomic positions.
    - smiles: SMILES string for the molecule.
    """
    # Parse the SMILES string with RDKit
    mol = Chem.MolFromSmiles(smiles.decode().replace('y',''))
    #mol = Chem.AddHs(mol)
    # Use RDKit to generate 3D coordinates
    #AllChem.EmbedMolecule(mol, AllChem.ETKDG())

    # Align the RDKit-generated coordinates with the conformer using a simple RMSD alignment
    # This step assumes that both sets of coordinates are in a compatible order
    #AllChem.AlignMolConformers(mol)

    if(create_figure_and_axes):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(-16, 16)
        ax.set_ylim(-16, 16)
        ax.set_zlim(-16, 16)

    plot_conformer_with_bonds_from_rdkit_mol(ax,conformer,mol)

    if(create_figure_and_axes):
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')
        plt.title('Molecular Conformation with Bonds')
        plt.show()


def animate_diffusion_process(diffusion_steps, smiles):
    """
    Creates an animation showing the diffusion process for a single conformer.

    Parameters:
    - diffusion_steps: A tensor of shape (steps, num_atoms, 3) representing the conformer's coordinates at each step.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-6,6)
    ax.set_ylim(-6,6)
    ax.set_zlim(-6,6)

    # Parse the SMILES string with RDKit
    mol = Chem.MolFromSmiles(smiles.decode().replace('y',''))
    # Use RDKit to generate 3D coordinates
    #AllChem.EmbedMolecule(mol, AllChem.ETKDG())

    # Align the RDKit-generated coordinates with the conformer using a simple RMSD alignment
    # This step assumes that both sets of coordinates are in a compatible order
    #AllChem.AlignMolConformers(mol)

    def init():
        """Initializes the plot with the first frame."""
        ax.clear()
        conformer = diffusion_steps[0]
        plot_conformer_with_bonds_from_rdkit_mol(ax,conformer,mol)
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.set_zlim(-6, 6)
        return fig,

    def update(step):
        """Updates the plot for each frame."""
        ax.clear()
        conformer = diffusion_steps[step]
        plot_conformer_with_bonds_from_rdkit_mol(ax,conformer,mol)
        ax.set_title(f'Step {step+1}')
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.set_zlim(-6, 6)
        return fig,

    anim = FuncAnimation(fig, update, frames=range(diffusion_steps.shape[0]), init_func=init, blit=True)
    #plt.show()
    plt.close()  # Prevents the final frame from being shown statically below the animation.

    return anim


def animate_diffusion_process_multiple(diffusion_steps, smiles):
    """
    Creates an animation showing the diffusion process for a single conformer.

    Parameters:
    - diffusion_steps: A tensor of shape (steps, num_atoms, 3) representing the conformer's coordinates at each step.
    """
    plots_x = 4
    plots_y = 4

    # Create a 4x4 grid of subplots
    fig, axes = plt.subplots(4, 4, figsize=(12, 12), subplot_kw={'projection': '3d'})

    # Adjust layout to prevent overlap of titles
    plt.tight_layout()

    # Loop through each subplot and set some dummy data
    mols = [None] * (plots_x*plots_y)

    for i in range(4):
        for j in range(4):
            idx = i*plots_x+j
            mol = Chem.MolFromSmiles(smiles[idx].decode().replace('y', ''))
            mols[idx] = mol
            ax = axes[i, j]
            ax.plot([0, 1, 2, 3, 4], [0, 1, 4, 9, 16])  # Dummy data
            ax.set_title(f'Subplot {i + 1},{j + 1}')  # Set subplot title
            ax.set_xlim(-6,6)
            ax.set_ylim(-6, 6)
            ax.set_zlim(-6, 6)

    #fig = plt.figure(figsize=(10, 8))
    #ax = fig.add_subplot(111, projection='3d')



    # Parse the SMILES string with RDKit

    # Use RDKit to generate 3D coordinates
    #AllChem.EmbedMolecule(mol, AllChem.ETKDG())

    # Align the RDKit-generated coordinates with the conformer using a simple RMSD alignment
    # This step assumes that both sets of coordinates are in a compatible order
    #AllChem.AlignMolConformers(mol)

    def init():
        for i in range(plots_x):
            for j in range(plots_y):
                idx = i * plots_x + j
                #print(idx)
                ax = axes[i,j]
                mol = mols[idx]
                """Initializes the plot with the first frame."""
                ax.clear()
                conformer = diffusion_steps[idx,0,:,:]
                plot_conformer_with_bonds_from_rdkit_mol(ax,conformer,mol)
                ax.set_xlabel('X Axis')
                ax.set_ylabel('Y Axis')
                ax.set_zlabel('Z Axis')
                ax.set_xlim(-6, 6)
                ax.set_ylim(-6, 6)
                ax.set_zlim(-6, 6)
        return fig,

    def update(step):
        print(f"frame {step}")
        for i in range(plots_x):
            for j in range(plots_y):
                idx = i * plots_x + j
                ax = axes[i,j]
                mol = mols[idx]
                print(idx)
                """Updates the plot for each frame."""
                ax.clear()
                conformer = diffusion_steps[idx,step,:,:]
                plot_conformer_with_bonds_from_rdkit_mol(ax,conformer,mol)
                ax.set_title(f'Step {step+1}')
                ax.set_xlabel('X Axis')
                ax.set_ylabel('Y Axis')
                ax.set_zlabel('Z Axis')
                ax.set_xlim(-6, 6)
                ax.set_ylim(-6, 6)
                ax.set_zlim(-6, 6)
        return fig,

    anim = FuncAnimation(fig, update, frames=range(diffusion_steps.shape[1]), init_func=init, blit=True)
    #plt.show()
    plt.close()  # Prevents the final frame from being shown statically below the animation.
    return anim





if __name__ == "__main__":
    # Assuming `model` is your trained model
    # Example parameters
    num_atoms = 32  # Adjust based on your molecule
    num_steps = 40  # Total diffusion steps in reverse
    num_conformers = 16  # Number of conformers to generate

    for iteration in range(32):
        #base_model, diffusion_model, conf = run_training_3d_diffusion_01.create_model("C:\\dev\\leet-chem\\leet-deep\\configs\\diffusion_3d_dm\\conf_diffusion_3d_a_02.json")
        #base_model, diffusion_model, conf = run_training_3d_diffusion_01.create_model("C:\\dev\\leet-chem\\leet-deep\\configs\\diffusion_3d_dm\\conf_diffusion_3d_a_02_new.json")
        base_model, diffusion_model, conf = run_training_3d_diffusion_01.create_model("C:\\dev\\leet-chem\\leet-deep\\configs\\diffusion_3d_dm\\conf_diffusion_3d_a_03_smaller_a.json")
        base_model_3ds, diffusion_model_3ds, conf_3ds = run_training_ds3_diffusion_01.create_model("C:\\dev\\leet-chem\\leet-deep\\configs\\ds3_diffusion3d\\diffusion3d_conf_01_run_01.json")

        train_loader , val_loader = run_training_3d_diffusion_01.create_dataset(conf)

        base_model = base_model.to(conf.device)
        diffusion_model = diffusion_model.to(conf.device)

        base_model_3ds = base_model_3ds.to(conf.device)
        diffusion_model_3ds = diffusion_model_3ds.to(conf.device)


        batch = next(iter(train_loader))
        inputs_pre = batch['smiles_enc'].to(conf.device)
        #idx_mol = 0


        multiple_molecules_conformers = torch.zeros((16,num_steps,num_atoms,3),device=conf.device)
        multiple_molecules_num_atoms  = torch.zeros((16,1))
        multiple_molecules_idxmol     = range(16)
        for idx_mol in multiple_molecules_idxmol:
            inputs = torch.unsqueeze(inputs_pre[idx_mol, :], 0)
            num_atoms_i = batch['num_atoms'][idx_mol]

            print('\n\n\n\n\n\n--------------------------------------------------------------')
            generated_conformers , all_conformers = generate_conformers(base_model, diffusion_model, conf.device, inputs, num_atoms_i, num_steps, num_conformers)
            print('\n\n\n\n\n\n--------------------------------------------------------------')
            generated_conformers_3ds, all_conformers_3ds = generate_conformers(base_model_3ds, diffusion_model_3ds, conf.device, inputs,num_atoms_i, num_steps, num_conformers)
            print('\n\n\n\n\n\n--------------------------------------------------------------')

            #multiple_molecules_conformers[idx_mol,:,:,:] = all_conformers[0:num_steps,0,:,:]
            multiple_molecules_conformers[idx_mol, :, :, :] = all_conformers_3ds[0:num_steps, 0, :, :]

        print("Conformers generated")


        #plot_conformer_with_bonds_from_smiles( None,all_conformers[38,0,:num_atoms,:].cpu() , batch['smiles'][idx_mol],create_figure_and_axes=True)

        # Assuming generated_conformers is the tensor of shape (num_conformers, num_atoms, 3)
        plot_conformers_2(generated_conformers.cpu())
        matplotlib.use('TkAgg')  # Use the TkAgg backend for interactive plots

        #anim_a = animate_diffusion_process_multiple(all_conformers[0:40, 0:16, :num_atoms, :].cpu(), batch['smiles'][idx_mol])
        #anim_a = animate_diffusion_process_multiple(multiple_molecules_conformers[0:16,0:40,:,:].cpu(), batch['smiles'])
        anim_a = animate_diffusion_process_multiple(multiple_molecules_conformers[0:16, 0:40, :, :].cpu(),batch['smiles'])
        anim_a.save(f'diffusion_ds3_x16_{iteration}.gif', writer='pillow')

        # for zi in range(20):
        #     anim_a = animate_diffusion_process( all_conformers[0:40,zi,:num_atoms,:].cpu(),batch['smiles'][idx_mol])
        #     anim_a.save(f'diffusion_animation_x_{zi}.gif', writer='pillow')
            #anim_a.save(f'diffusion_animation_{random.randint(0,100000)}.gif', writer='pillow')


