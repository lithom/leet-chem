package tech.molecules.leet.gui.chem.project.action;

import com.fasterxml.jackson.core.exc.StreamWriteException;
import com.fasterxml.jackson.databind.DatabindException;
import tech.molecules.leet.chem.LeetSerialization;
import tech.molecules.leet.chem.sar.SimpleSARProject;

import javax.swing.*;
import java.awt.event.ActionEvent;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

public class SaveProjectAction extends AbstractAction implements ObjectSpecific {

    public SaveProjectAction() {
        super("Save project..");
    }

    @Override
    public void actionPerformed(ActionEvent e) {
        // Create a file chooser dialog
        JFileChooser fileChooser = new JFileChooser();

        // Show the dialog to pick a filename
        int choice = fileChooser.showSaveDialog(null);

        if (choice == JFileChooser.APPROVE_OPTION) {
            // Get the selected file
            File file = fileChooser.getSelectedFile();

            // Save the project to the selected file
            saveProject(file);
        }
    }

    private void saveProject(File file) {
        if(this.project!=null) {
            try {
                BufferedWriter out = new BufferedWriter(new FileWriter(file));
                LeetSerialization.OBJECT_MAPPER.writer().writeValue(out, this.project);
            } catch (StreamWriteException e) {
                e.printStackTrace();
                //throw new RuntimeException(e);
            } catch (DatabindException e) {
                e.printStackTrace();
                //throw new RuntimeException(e);
            } catch (IOException e) {
                e.printStackTrace();
                //throw new RuntimeException(e);
            }
        }

        // Display a success message
        JOptionPane.showMessageDialog(null, "Project saved successfully!");
    }

    private SimpleSARProject project;

    @Override
    public void setObject(Object obj) {
        if(obj instanceof SimpleSARProject) {
            this.project = (SimpleSARProject)obj;
        }
    }

    @Override
    public Object getObject() {
        return this.project;
    }
}
