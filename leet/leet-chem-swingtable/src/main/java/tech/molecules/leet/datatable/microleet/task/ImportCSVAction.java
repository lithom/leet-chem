package tech.molecules.leet.datatable.microleet.task;

import javax.swing.*;
import javax.swing.filechooser.FileNameExtensionFilter;
import java.awt.event.ActionEvent;

public class ImportCSVAction extends AbstractAction {
    public ImportCSVAction(String name) {
        super(name);
    }

    @Override
    public void actionPerformed(ActionEvent e) {
        JFileChooser fileChooser = new JFileChooser();
        // Create a filter for CSV files
        FileNameExtensionFilter csvFilter = new FileNameExtensionFilter("CSV Files", "csv");
        fileChooser.setFileFilter(csvFilter);

        int result = fileChooser.showOpenDialog(null);
        if (result == JFileChooser.APPROVE_OPTION) {

            // 1.

        }
    }



}
