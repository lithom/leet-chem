package tech.molecules.leet.gui.chem.action;

import com.actelion.research.chem.io.DWARFileParser;
import tech.molecules.leet.chem.ChemUtils;
import tech.molecules.leet.chem.StructureWithID;

import javax.swing.*;
import java.awt.event.ActionEvent;
import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class LoadStructureWithIDFromDWARAction extends AbstractAction {

    public LoadStructureWithIDFromDWARAction() {
        super("Load Structures..");
    }

    private List<StructureWithID> structures = new ArrayList<>();

    public List<StructureWithID> getStructures() {
        return this.structures;
    }

    @Override
    public void actionPerformed(ActionEvent e) {
        JFileChooser fileChooser = new JFileChooser();
        fileChooser.setDialogTitle("Select a File");

        int selection = fileChooser.showOpenDialog(null);

        if (selection == JFileChooser.APPROVE_OPTION) {
            File selectedFile = fileChooser.getSelectedFile();
            System.out.println("Selected File: " + selectedFile.getAbsolutePath());

            DWARFileParser pi = new DWARFileParser(selectedFile);
            int si_structure = pi.getSpecialFieldIndex("Structure");
            int si_molid     = pi.getFieldIndex("Idorsia No");

            structures = new ArrayList<>();
            while(pi.next()) {
                String idc = pi.getSpecialFieldData(si_structure);
                String mid = pi.getFieldData(si_molid);
                structures.add(new StructureWithID(mid,"",ChemUtils.parseIDCode(idc)));
            }
        } else {
            System.out.println("No file selected.");
        }
    }
}
