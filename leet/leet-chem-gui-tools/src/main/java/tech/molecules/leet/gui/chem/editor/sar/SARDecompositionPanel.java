package tech.molecules.leet.gui.chem.editor.sar;

import tech.molecules.leet.gui.chem.editor.SARDecompositionEditor;

import javax.swing.*;
import java.awt.*;

public class SARDecompositionPanel extends JPanel {

    private SARDecompositionFragmentListModel model;

    private SARDecompositionEditor editor = new SARDecompositionEditor();

    private SARDecompositionFragmentListPanel decompPanel;


    public SARDecompositionPanel(SARDecompositionFragmentListModel model) {
        this.model = model;
        this.decompPanel = new SARDecompositionFragmentListPanel(model);
        this.reinit();
    }

    private void reinit() {
        this.removeAll();
        this.setLayout(new BorderLayout());

        this.add(editor,BorderLayout.NORTH);
        this.add(decompPanel);
    }

}
