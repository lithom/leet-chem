package tech.molecules.leet.gui.chem.editor.sar;

import com.actelion.research.chem.Canonizer;
import com.actelion.research.chem.StereoMolecule;
import com.actelion.research.gui.editor.SwingEditorPanel;
import com.formdev.flatlaf.icons.FlatWindowAbstractIcon;

import javax.swing.*;
import java.awt.*;
import java.awt.datatransfer.Clipboard;
import java.awt.datatransfer.StringSelection;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.function.Consumer;

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
        JButton buttonSetLabel      = new JButton("Set Label..");
        JButton jbExport            = new JButton("Export");

        this.panelTop.add(buttonConfirmChange);
        this.panelTop.add(buttonRevert);
        this.panelTop.add(buttonSetLabel);
        this.panelTop.add(jbExport);

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

        buttonSetLabel.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                LabelSetterDialog dialog = new LabelSetterDialog(null);
                dialog.setVisible(true);
            }
        });
        jbExport.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                Canonizer ca = new Canonizer(editor.getDrawArea().getMolecule(),Canonizer.ENCODE_ATOM_CUSTOM_LABELS);
                String theString = ca.getIDCode();
                StringSelection selection = new StringSelection(theString);
                Clipboard clipboard = Toolkit.getDefaultToolkit().getSystemClipboard();
                clipboard.setContents(selection, selection);
            }
        });

    }




    public class LabelSetterDialog extends JDialog {

        private JTextField textField;

        public LabelSetterDialog(JFrame parent) {
            super(parent, "Label Setter", true); // The 'true' argument makes it a modal dialog
            setSize(300, 150);
            setLayout(new BorderLayout());

            JPanel panel = new JPanel();
            panel.setLayout(new FlowLayout());

            JLabel label = new JLabel("Label:");
            textField = new JTextField(20);

            JButton setButton = new JButton("Set Label");

            setButton.addActionListener(new ActionListener() {
                @Override
                public void actionPerformed(ActionEvent e) {
                    String labelText = textField.getText();
                    setLabel(labelText);
                    dispose(); // Close the dialog
                }
            });

            panel.add(label);
            panel.add(textField);
            panel.add(setButton);

            add(panel, BorderLayout.CENTER);
        }

        private void setLabel(String labelText) {
            System.out.println("Label set to: " + labelText);
            StereoMolecule mi = editor.getDrawArea().getMolecule();
            for(int zi=0;zi<mi.getAtoms();zi++) {
                if(mi.isSelectedAtom(zi)) {
                    mi.setAtomCustomLabel(zi,labelText);
                }
            }
            editor.getDrawArea().setMolecule(mi);
            editor.getSwingDrawArea().repaint();
        }
    }


    // idcode_A: dcMH@DdDfVulUZ`BH@Mapj[aW_wPg~op

}
