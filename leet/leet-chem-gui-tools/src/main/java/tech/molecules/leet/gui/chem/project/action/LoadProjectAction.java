package tech.molecules.leet.gui.chem.project.action;

import tech.molecules.leet.chem.LeetSerialization;
import tech.molecules.leet.chem.sar.SimpleSARProject;

import javax.swing.*;
import java.awt.event.ActionEvent;
import java.io.File;
import java.io.IOException;
import java.util.function.Consumer;

public class LoadProjectAction extends AbstractAction {

    private Consumer<SimpleSARProject> projectConsumer;

    public LoadProjectAction(Consumer<SimpleSARProject> projectConsumer) {
        super("Load project..");
        this.projectConsumer = projectConsumer;
    }

    @Override
    public void actionPerformed(ActionEvent e) {
        // Create a file chooser dialog
        JFileChooser fileChooser = new JFileChooser();

        // Show the dialog to pick a file
        int choice = fileChooser.showOpenDialog(null);

        if (choice == JFileChooser.APPROVE_OPTION) {
            // Get the selected file
            File file = fileChooser.getSelectedFile();

            // Load the project from the selected file
            loadProject(file);
        }
    }

    private void loadProject(File file) {
        try {
            SimpleSARProject project = (SimpleSARProject) LeetSerialization.OBJECT_MAPPER.reader().readValue( file , SimpleSARProject.class );
            this.projectConsumer.accept(project);
        } catch (IOException e) {
            e.printStackTrace();
            //throw new RuntimeException(e);
        }


        // Display a success message
        JOptionPane.showMessageDialog(null, "Project loaded successfully!");
    }

}
