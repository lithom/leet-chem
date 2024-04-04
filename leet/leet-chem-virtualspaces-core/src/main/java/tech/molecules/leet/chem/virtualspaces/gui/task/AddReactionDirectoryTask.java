package tech.molecules.leet.chem.virtualspaces.gui.task;

import tech.molecules.leet.chem.virtualspaces.gui.SpaceCreatorController;

import javax.swing.*;
import java.awt.event.ActionEvent;
import java.io.File;

public class AddReactionDirectoryTask extends AbstractAction {

    private SpaceCreatorController controller;

    public AddReactionDirectoryTask(SpaceCreatorController controller) {
        super("Add Reactions Directory..");
        this.controller = controller;
    }


    @Override
    public void actionPerformed(ActionEvent e) {
        JFileChooser directoryChooser = new JFileChooser();
        directoryChooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY); // Set to directories only

        int returnValue = directoryChooser.showOpenDialog(controller.getView().getFrame()); // 'frame' is the parent component
        if (returnValue == JFileChooser.APPROVE_OPTION) {
            File directory = directoryChooser.getSelectedFile();
            File[] files = directory.listFiles((dir, name) -> name.toLowerCase().endsWith(".rxn"));
            if (files != null) {
                for (File file : files) {
                    // Here, you would create a new ReactionMechanism object for each .rxn file
                    // and add it to your list or model. This example does not detail extracting
                    // information from the files or creating the ReactionMechanism object,
                    // as it will depend on your specific requirements.
                    //String filepath = file.getAbsolutePath();
                    try {
                        controller.addReaction(file);
                    } catch (Exception ex) {
                        System.out.println("Error loading reaction file..");
                        ex.printStackTrace();
                    }

                    //Reaction rxn = new Reaction(); // Placeholder - you need to define how to parse .rxn files
                    //rxn.setSomeDataBasedOnFile(file); // Placeholder for setting data
                    //ReactionMechanism newMechanism = new ReactionMechanism(filepath, rxn);
                    //modelB.addReactionMechanism(newMechanism);
                }
            }
        }
    }
}
