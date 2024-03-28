package tech.molecules.leet.chem.virtualspaces.gui.task;

import com.actelion.research.chem.reaction.Reaction;
import com.idorsia.research.chem.hyperspace.importer.MultiStepImporter;
import com.idorsia.research.chem.hyperspace.io.SynthonSpaceParser2;
import tech.molecules.leet.chem.virtualspaces.ChemicalSpaceCreator2;
import tech.molecules.leet.chem.virtualspaces.SpaceCreation_A;
import tech.molecules.leet.chem.virtualspaces.gui.BuildingBlockFile;
import tech.molecules.leet.chem.virtualspaces.gui.LoadedBB;
import tech.molecules.leet.chem.virtualspaces.gui.ReactionMechanism;
import tech.molecules.leet.chem.virtualspaces.gui.SpaceCreatorController;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

public class CreateSpaceTask extends AbstractAction {

    private SpaceCreatorController controller;


    private File fileOutputDirectory = null;
    private int maxAtoms = 0;


    public CreateSpaceTask(SpaceCreatorController controller) {
        super("Create Space");
        this.controller = controller;
    }

    @Override
    public void actionPerformed(ActionEvent e) {
        Map<BuildingBlockFile, List<LoadedBB>> loadedBBs = this.controller.getModel().getBuildingBlockFileTableModel().getLoadedBBs();
        List<LoadedBB> bbs = loadedBBs.values().stream().flatMap(xi -> xi.stream()).collect(Collectors.toList());
        List<ReactionMechanism> reactionMechanisms = this.controller.getModel().getReactionMechanismTableModel().getSelectedReactionMechanisms();
        List<Reaction> reactions = reactionMechanisms.stream().map(xi -> xi.getRxn()).collect(Collectors.toList());

        if(bbs.isEmpty()) {
            JOptionPane.showMessageDialog(controller.getView().getFrame(), "No Building Blocks selected.", "Warning", JOptionPane.WARNING_MESSAGE);
            return;
        }
        if(reactions.isEmpty()) {
            JOptionPane.showMessageDialog(controller.getView().getFrame(), "No Reactions selected.", "Warning", JOptionPane.WARNING_MESSAGE);
            return;
        }

        JDialog dialog = createConfigureSpaceCreationDialog(controller.getView().getFrame());
        dialog.setVisible(true);
        List<Function<LoadedBB,Boolean>> filters = new ArrayList<>();
        if(maxAtoms > 0) {
            filters.add( (x) -> x.getNumAtoms() <= maxAtoms );
        }

        System.out.println("Space creation: Start!");
        ChemicalSpaceCreator2 spaceCreator = SpaceCreation_A.createSpaceCreator(bbs,reactions,fileOutputDirectory,filters);
        spaceCreator.create();
        System.out.println("Space creation: DONE!");


        System.out.println("Hyperspace Import: Start!");
        String synthonSpaceFile = fileOutputDirectory.getAbsolutePath()+File.separator+"synthonSpace.tsv";
        String[] importerArguments = new String[]{"2split",
                fileOutputDirectory.getAbsolutePath()+File.separator+"CombinatorialLibraries",
                synthonSpaceFile,
                "BB-ID"};
        MultiStepImporter.main(importerArguments);

        System.out.println("Hyperspace Import: DONE!");

        System.out.println("Hyperspace Space Creation: Start!");
        int numProcessors = Math.max( 1 , Runtime.getRuntime().availableProcessors()-1 );
        SynthonSpaceParser2.initREALSpace(synthonSpaceFile,"Hyperspace","FragFp", numProcessors);
        System.out.println("Hyperspace Space Creation: DONE!");
    }



    private JDialog createConfigureSpaceCreationDialog(JFrame parentFrame) {
        // Create the directory selection dialog
        JDialog configureSpaceCreationDialog = new JDialog(parentFrame, "Create Space", true);
        configureSpaceCreationDialog.setDefaultCloseOperation(JDialog.DISPOSE_ON_CLOSE);

        // Create a panel to hold the components
        JPanel panel = new JPanel(new GridLayout(3, 1));

        // Create the "Max Atoms" label and text field
        JLabel maxAtomsLabel = new JLabel("Max Atoms:");
        JTextField maxAtomsTextField = new JTextField();

        // Add the label and text field to the panel
        panel.add(maxAtomsLabel);
        panel.add(maxAtomsTextField);

        // Create the "Select Directory" button
        JButton selectDirectoryButton = new JButton("Select Directory");
        selectDirectoryButton.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                // Create the file chooser
                JFileChooser fileChooser = new JFileChooser();
                fileChooser.setDialogTitle("Select Directory");
                fileChooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);

                // Show the file chooser dialog
                int userSelection = fileChooser.showDialog(configureSpaceCreationDialog, "Select");

                // Handle the selected directory
                if (userSelection == JFileChooser.APPROVE_OPTION) {
                    File selectedDirectory = fileChooser.getSelectedFile();
                    fileOutputDirectory = selectedDirectory;
                    //JOptionPane.showMessageDialog(configureSpaceCreationDialog, "Selected directory: " + selectedDirectory.getAbsolutePath());
                }
            }
        });

        // Create the "Start Compute" button
        JButton startComputeButton = new JButton("Start Space Creation");
        startComputeButton.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                // Get the value from the text field
                String maxAtomsText = maxAtomsTextField.getText();
                if(!maxAtomsText.isEmpty()) {
                    // Check if it's a valid integer
                    try {
                        maxAtoms = Integer.parseInt(maxAtomsText);
                        JOptionPane.showMessageDialog(configureSpaceCreationDialog, "Max Atoms: " + maxAtoms);
                        configureSpaceCreationDialog.dispose();
                    } catch (NumberFormatException ex) {
                        JOptionPane.showMessageDialog(configureSpaceCreationDialog, "Please enter a valid integer for Max Atoms.", "Error", JOptionPane.ERROR_MESSAGE);
                    }
                }
            }
        });

        // Add the "Select Directory" and "Start Compute" buttons to the panel
        panel.add(selectDirectoryButton);
        panel.add(startComputeButton);

        // Add the panel to the directory selection dialog
        configureSpaceCreationDialog.getContentPane().add(panel);

        // Set the size and location of the directory selection dialog
        configureSpaceCreationDialog.pack();
        configureSpaceCreationDialog.setLocationRelativeTo(parentFrame);

        return configureSpaceCreationDialog;
    }

}
