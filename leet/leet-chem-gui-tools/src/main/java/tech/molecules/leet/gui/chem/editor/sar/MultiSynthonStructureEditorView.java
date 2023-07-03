package tech.molecules.leet.gui.chem.editor.sar;

import com.actelion.research.gui.editor.SwingEditorPanel;
import com.formdev.flatlaf.icons.FlatWindowAbstractIcon;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

public class MultiSynthonStructureEditorView extends JPanel {

    private MultiSynthonStructureEditorModel model;


    private JPanel panelTop;
    private SwingEditorPanel editor;

    public MultiSynthonStructureEditorView(MultiSynthonStructureEditorModel model) {
        this.model = model;

        this.reinit();
    }

    private void reinit() {
        this.removeAll();
        this.setLayout(new BorderLayout());

        this.panelTop = new JPanel(); this.panelTop.setLayout(new FlowLayout(FlowLayout.LEFT));
        this.add(panelTop,BorderLayout.NORTH);

        if(!model.getSynthonEditedVersion().isFragment()) {
            model.getSynthonEditedVersion().setFragment(true);
        }
        this.editor = new SwingEditorPanel(model.getSynthonEditedVersion());
        this.editor.getDrawArea().setAllowQueryFeatures(true);
        this.add(editor,BorderLayout.CENTER);

        JButton buttonConfirmChange = new JButton("Confirm change");
        JButton buttonRevert        = new JButton("Revert to old");

        this.panelTop.add(buttonConfirmChange);
        this.panelTop.add(buttonRevert);

        buttonConfirmChange.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                model.confirmChangeToSynthon();
            }
        });
        buttonRevert.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                model.revertEditedSynthon();
                editor.getSwingDrawArea().repaint();
            }
        });
    }



}
