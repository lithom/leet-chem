package tech.molecules.leet.chem.virtualspaces.gui.task;

import tech.molecules.leet.chem.virtualspaces.gui.BuildingBlockFile;
import tech.molecules.leet.chem.virtualspaces.gui.SpaceCreatorController;

import javax.swing.*;
import javax.swing.filechooser.FileNameExtensionFilter;
import java.awt.event.ActionEvent;
import java.io.File;

public class AddBuildingBlockFilesTask extends AbstractAction  {

    private SpaceCreatorController controller;

    public AddBuildingBlockFilesTask(SpaceCreatorController controller) {
        super("Add Building Block Files..");
        this.controller = controller;
    }

    @Override
    public void actionPerformed(ActionEvent e) {
        JFileChooser fileChooser = new JFileChooser();
        fileChooser.setMultiSelectionEnabled(true); // Enable multiple file selection
        fileChooser.setFileSelectionMode(JFileChooser.FILES_ONLY);
        FileNameExtensionFilter filter = new FileNameExtensionFilter("BuildingBlock Files", "dwar", "sdf", "tsv");
        fileChooser.setFileFilter(filter);

        int returnValue = fileChooser.showOpenDialog(controller.getView().getFrame()); // 'frame' is the parent component
        if (returnValue == JFileChooser.APPROVE_OPTION) {
            File[] files = fileChooser.getSelectedFiles();
            // Process the selected files
            for (File file : files) {
                // Here, you would create a new BuildingBlockFile object for each file
                // and add it to your model. This example does not detail extracting
                // information from the files or updating the model, as it will depend
                // on your application's specific requirements.
                String filepath = file.getAbsolutePath();
                // Assuming a method in your controller or directly in your view for adding:
                // Note: You need to implement the logic for creating a BuildingBlockFile instance
                // and updating the table model.
                BuildingBlockFile newFile = new BuildingBlockFile(filepath,".xxx","Structure", "BB-ID",-1);
                controller.addBuildingBlockFile(newFile);
            }
        }
    }
}
